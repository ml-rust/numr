//! CPU runtime implementation

use super::client::{CpuAllocator, CpuClient};
use super::device::CpuDevice;
use crate::runtime::{
    FloatArithmeticContract, FusedMulAddContract, NoOpGraph, Runtime, SquareRootContract,
};
use std::alloc::{Layout as AllocLayout, alloc, dealloc};

/// Alignment of a buffer at or above [`CACHE_LINE_MIN_BYTES`].
const CACHE_LINE_ALIGN: usize = 64;

/// Buffer size from which an allocation is aligned to a cache line.
///
/// Below this size the system allocator's native alignment applies. An
/// over-aligned request takes the allocator's slower aligned path, and that
/// cost dominates a small tensor. Large buffers keep cache-line alignment so
/// SIMD rows do not straddle lines and streaming stores stay eligible.
const CACHE_LINE_MIN_BYTES: usize = 4096;

/// Largest alignment a buffer below [`CACHE_LINE_MIN_BYTES`] requests.
///
/// It covers every element type numr stores. The system allocator serves it
/// without the aligned path on 64-bit targets.
const SMALL_BUFFER_MAX_ALIGN: usize = 16;

/// Address returned for a zero-byte allocation: non-null and aligned for any
/// element type.
const ZERO_SIZE_ADDRESS: u64 = CACHE_LINE_ALIGN as u64;

/// Allocation layout of a CPU buffer of `size_bytes` bytes, or `None` when no
/// layout of that size exists.
///
/// The alignment is a pure function of `size_bytes`, so `deallocate` rebuilds
/// the exact layout `allocate` used. A buffer below [`CACHE_LINE_MIN_BYTES`]
/// gets the largest power of two that is at most `size_bytes` and at most
/// [`SMALL_BUFFER_MAX_ALIGN`]. A value of type `T` that fits in the buffer has
/// `align_of::<T>() <= size_of::<T>() <= size_bytes`, so that alignment holds
/// it. No kernel's result depends on the buffer address: SIMD loads and stores
/// are unaligned, and the only aligned stores check the address first.
///
/// `size_bytes` must be nonzero. Callers handle the zero case first.
pub(crate) fn buffer_layout(size_bytes: usize) -> Option<AllocLayout> {
    debug_assert!(size_bytes > 0, "a zero-byte buffer has no layout");
    let align = if size_bytes >= CACHE_LINE_MIN_BYTES {
        CACHE_LINE_ALIGN
    } else {
        let floor_pow2 = 1usize << (usize::BITS - 1 - size_bytes.leading_zeros());
        floor_pow2.min(SMALL_BUFFER_MAX_ALIGN)
    };
    AllocLayout::from_size_align(size_bytes, align).ok()
}

/// CPU compute runtime
///
/// This is the default runtime that works on any platform.
/// Memory is allocated on the heap using the system allocator.
#[derive(Clone, Debug, Default)]
pub struct CpuRuntime;

impl Runtime for CpuRuntime {
    type Device = CpuDevice;
    type Client = CpuClient;
    type Allocator = CpuAllocator;
    type Graph = NoOpGraph;
    type RawHandle = ();
    type DType = crate::dtype::DType;

    fn name() -> &'static str {
        "cpu"
    }

    fn float_arithmetic_contract(dtype: Self::DType) -> FloatArithmeticContract {
        match dtype {
            crate::dtype::DType::F32 | crate::dtype::DType::F64 => {
                FloatArithmeticContract::Ieee754Basic
            }
            _ => FloatArithmeticContract::BackendDefined,
        }
    }

    fn fused_mul_add_contract(dtype: Self::DType) -> FusedMulAddContract {
        match dtype {
            crate::dtype::DType::F32 | crate::dtype::DType::F64 => {
                FusedMulAddContract::Ieee754Fused
            }
            _ => FusedMulAddContract::BackendDefined,
        }
    }

    fn square_root_contract(dtype: Self::DType) -> SquareRootContract {
        match dtype {
            crate::dtype::DType::F32 | crate::dtype::DType::F64 => {
                SquareRootContract::Ieee754RoundToNearest
            }
            _ => SquareRootContract::BackendDefined,
        }
    }

    fn allocate(size_bytes: usize, _device: &Self::Device) -> crate::error::Result<u64> {
        // A zero-byte allocation still needs a NON-NULL, aligned address.
        // `slice::from_raw_parts(ptr, 0)` is undefined behaviour on a null
        // pointer, and the kernels build slices from a tensor's pointer
        // unconditionally, so returning 0 here made every empty tensor UB.
        // `ZERO_SIZE_ADDRESS` is aligned for every element type. `deallocate`
        // skips `size_bytes == 0`, so this address is never passed to the
        // allocator.
        if size_bytes == 0 {
            return Ok(ZERO_SIZE_ADDRESS);
        }

        let layout = buffer_layout(size_bytes)
            .ok_or_else(|| crate::error::Error::OutOfMemory { size: size_bytes })?;

        // Use alloc (not alloc_zeroed) — Tensor::empty is explicitly uninitialized.
        // Operations that need zeroed memory (e.g. Tensor::zeros) handle zeroing themselves.
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(crate::error::Error::OutOfMemory { size: size_bytes });
        }

        Ok(ptr as u64)
    }

    fn deallocate(ptr: u64, size_bytes: usize, _device: &Self::Device) {
        if ptr == 0 || size_bytes == 0 {
            return;
        }

        // A layout this size could never have been allocated (allocate() rejects
        // it), so there is nothing to free. Returning is the only sound option —
        // panicking in a Drop path would abort the process.
        let Some(layout) = buffer_layout(size_bytes) else {
            return;
        };

        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
    }

    fn copy_to_device(src: &[u8], dst: u64, _device: &Self::Device) -> crate::error::Result<()> {
        if src.is_empty() || dst == 0 {
            return Ok(());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        }
        Ok(())
    }

    fn copy_from_device(
        src: u64,
        dst: &mut [u8],
        _device: &Self::Device,
    ) -> crate::error::Result<()> {
        if dst.is_empty() || src == 0 {
            return Ok(());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_within_device(
        src: u64,
        dst: u64,
        size_bytes: usize,
        _device: &Self::Device,
    ) -> crate::error::Result<()> {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return Ok(());
        }

        unsafe {
            // Use copy (not copy_nonoverlapping) in case src and dst overlap
            std::ptr::copy(src as *const u8, dst as *mut u8, size_bytes);
        }
        Ok(())
    }

    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        _device: &Self::Device,
    ) -> crate::error::Result<()> {
        if src_handle == 0 || dst_handle == 0 || shape.is_empty() {
            return Ok(());
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(());
        }

        // For CPU, we can use pointer arithmetic directly. Folding the byte
        // offset into the base pointer is what lets a contiguous-but-offset
        // view (every `narrow()` result) take the single-memcpy fast path.
        let src_base = (src_handle as usize + src_byte_offset) as *const u8;
        let dst_base = dst_handle as *mut u8;

        // The destination is freshly allocated row-major storage, so it cannot
        // alias the source. See `strided_copy` for the tier structure.
        unsafe {
            super::strided_copy::copy_strided_impl(src_base, dst_base, shape, strides, elem_size);
        }
        Ok(())
    }

    fn default_device() -> Self::Device {
        CpuDevice::new()
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        CpuClient::new(device.clone())
    }

    fn raw_handle(_client: &Self::Client) -> &Self::RawHandle {
        &()
    }

    fn capture_graph_into<F>(
        client: &Self::Client,
        inputs: &[&crate::tensor::Tensor<Self>],
        outputs: &[&crate::tensor::Tensor<Self>],
        f: F,
    ) -> crate::error::Result<crate::runtime::CapturedGraph<Self>>
    where
        F: FnOnce(&Self::Client) -> crate::error::Result<()>,
    {
        f(client)?;
        let owned_inputs: Vec<_> = inputs.iter().map(|t| (*t).clone()).collect();
        let owned_outputs: Vec<_> = outputs.iter().map(|t| (*t).clone()).collect();
        Ok(crate::runtime::CapturedGraph::new(
            crate::runtime::NoOpGraph,
            owned_inputs,
            owned_outputs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    #[test]
    fn test_cpu_supports_graph_capture() {
        assert!(!CpuRuntime::supports_graph_capture());
    }

    #[test]
    fn test_cpu_capture_graph_into_executes_eagerly() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Allocate a 4-element f32 output tensor.
        let out =
            crate::tensor::Tensor::<CpuRuntime>::zeros(&[4], crate::dtype::DType::F32, &device)
                .unwrap();
        let input = crate::tensor::Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[4],
            &device,
        )
        .unwrap();

        let mut closure_ran = false;
        let graph = CpuRuntime::capture_graph_into(&client, &[&input], &[&out], |_c| {
            closure_ran = true;
            // Copy input bytes into `out` in-place — visible after capture.
            let size = input.numel() * input.dtype().size_in_bytes();
            CpuRuntime::copy_within_device(input.ptr(), out.ptr(), size, input.device())
        })
        .expect("cpu capture_graph_into should succeed");

        assert!(closure_ran, "closure must run eagerly");

        // The copy performed inside the closure must be visible.
        let result = out.to_vec::<f32>();
        assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);

        // launch() is a no-op on CPU — should not error.
        graph.launch().expect("launch should be a no-op on CPU");
    }

    #[test]
    fn test_cpu_capture_graph_into_propagates_error() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let result = CpuRuntime::capture_graph_into(&client, &[], &[], |_c| {
            Err(crate::error::Error::Backend(
                "sentinel error from closure".into(),
            ))
        });

        assert!(result.is_err(), "error from the closure must be propagated");
    }

    #[test]
    fn buffer_layout_small_sizes_take_floor_power_of_two_capped_at_16() {
        let cases = [
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 4),
            (7, 4),
            (8, 8),
            (12, 8),
            (16, 16),
            (24, 16),
            (512, 16),
            (CACHE_LINE_MIN_BYTES - 1, 16),
        ];
        for (size, align) in cases {
            let layout = buffer_layout(size).expect("representable size");
            assert_eq!(layout.size(), size);
            assert_eq!(layout.align(), align, "alignment for {size} bytes");
        }
    }

    #[test]
    fn buffer_layout_large_sizes_are_cache_line_aligned() {
        for size in [CACHE_LINE_MIN_BYTES, CACHE_LINE_MIN_BYTES + 8, 1 << 20] {
            let layout = buffer_layout(size).expect("representable size");
            assert_eq!(
                layout.align(),
                CACHE_LINE_ALIGN,
                "alignment for {size} bytes"
            );
        }
    }

    #[test]
    fn buffer_layout_holds_every_element_type_that_fits() {
        use crate::dtype::{DType, DataType};
        let dtypes = [
            DType::F64,
            DType::F32,
            DType::I64,
            DType::I32,
            DType::U8,
            DType::Bool,
            DType::Complex64,
            DType::Complex128,
        ];
        for dtype in dtypes {
            for len in 1..=64usize {
                let size = dtype.storage_bytes(len);
                let ptr = CpuRuntime::allocate(size, &CpuDevice::new()).expect("allocate");
                assert_eq!(
                    ptr % dtype.size_in_bytes().min(SMALL_BUFFER_MAX_ALIGN) as u64,
                    0,
                    "{dtype:?} x {len} is misaligned"
                );
                CpuRuntime::deallocate(ptr, size, &CpuDevice::new());
            }
        }
    }

    #[test]
    fn large_allocation_is_cache_line_aligned() {
        let device = CpuDevice::new();
        let ptr = CpuRuntime::allocate(CACHE_LINE_MIN_BYTES, &device).expect("allocate");
        assert_eq!(ptr % CACHE_LINE_ALIGN as u64, 0);
        CpuRuntime::deallocate(ptr, CACHE_LINE_MIN_BYTES, &device);
    }
}
