//! CPU runtime implementation
//!
//! The CPU runtime uses standard heap allocation and provides a reference
//! implementation for all tensor operations.
//!
//! # Broadcasting
//!
//! NumPy-style broadcasting is fully supported for binary arithmetic operations
//! (add, sub, mul, div, pow, max, min). Shapes are broadcast according to standard
//! rules: dimensions are right-aligned and expanded where one operand has size 1.
//!
//! Comparison operations (eq, ne, lt, le, gt, ge) also support broadcasting.
//!
//! # Non-contiguous Tensors
//!
//! Operations handle non-contiguous tensors via strided memory access. For
//! broadcasting, a strided kernel is used that correctly handles stride-0
//! dimensions (where a single value is broadcast across the dimension).

mod fft;
mod helpers;
pub mod jacobi;
mod kernels;
mod linalg;
mod ops;
mod sort;
#[cfg(feature = "sparse")]
pub(crate) mod sparse;
pub mod special;
mod statistics;

use super::{DefaultAllocator, Device, Runtime, RuntimeClient};
use crate::dtype::Element;
use crate::ops::{BinaryOp, Kernel, ReduceOp, UnaryOp};
use std::alloc::{Layout as AllocLayout, alloc_zeroed, dealloc};

// Re-export TensorOps, ScalarOps, CompareOps, SparseOps traits for convenience
pub use crate::ops::{CompareOps, ScalarOps, TensorOps};
#[cfg(feature = "sparse")]
pub use crate::sparse::SparseOps;

// Re-export algorithm traits
pub use crate::algorithm::LinearAlgebraAlgorithms;
pub use crate::algorithm::SpecialFunctions;
pub use crate::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};

// Re-export Tensor for tests
pub use crate::tensor::Tensor;

/// CPU compute runtime
///
/// This is the default runtime that works on any platform.
/// Memory is allocated on the heap using the system allocator.
#[derive(Clone, Debug, Default)]
pub struct CpuRuntime;

/// CPU device (there's only one: the host CPU)
#[derive(Clone, Debug, Default)]
pub struct CpuDevice {
    id: usize,
}

impl CpuDevice {
    /// Create a new CPU device
    pub fn new() -> Self {
        Self { id: 0 }
    }
}

impl Device for CpuDevice {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> String {
        "cpu".to_string()
    }
}

/// CPU client for operation dispatch
#[derive(Clone, Debug)]
pub struct CpuClient {
    pub(crate) device: CpuDevice,
    allocator: CpuAllocator,
}

impl CpuClient {
    /// Create a new CPU client
    pub fn new(device: CpuDevice) -> Self {
        let allocator = create_cpu_allocator(device.clone());
        Self { device, allocator }
    }
}

impl RuntimeClient<CpuRuntime> for CpuClient {
    fn device(&self) -> &CpuDevice {
        &self.device
    }

    fn synchronize(&self) {
        // CPU operations are synchronous, nothing to do
    }

    fn allocator(&self) -> &CpuAllocator {
        &self.allocator
    }
}

/// CPU-specific allocator type alias
pub type CpuAllocator = DefaultAllocator<CpuDevice>;

/// Create a CPU allocator for the given device
fn create_cpu_allocator(device: CpuDevice) -> CpuAllocator {
    DefaultAllocator::new(
        device,
        |size, _dev| {
            if size == 0 {
                return 0;
            }
            let align = 64; // AVX-512 alignment
            let layout =
                AllocLayout::from_size_align(size, align).expect("Invalid allocation layout");
            let ptr = unsafe { alloc_zeroed(layout) };
            if ptr.is_null() {
                panic!("Failed to allocate {} bytes", size);
            }
            ptr as u64
        },
        |ptr, size, _dev| {
            if ptr == 0 || size == 0 {
                return;
            }
            let align = 64;
            let layout =
                AllocLayout::from_size_align(size, align).expect("Invalid allocation layout");
            unsafe {
                dealloc(ptr as *mut u8, layout);
            }
        },
    )
}

impl Runtime for CpuRuntime {
    type Device = CpuDevice;
    type Client = CpuClient;
    type Allocator = CpuAllocator;
    type RawHandle = (); // CPU has no special handle needed

    fn name() -> &'static str {
        "cpu"
    }

    fn allocate(size_bytes: usize, _device: &Self::Device) -> u64 {
        if size_bytes == 0 {
            return 0;
        }

        // Use aligned allocation for SIMD compatibility
        let align = 64; // AVX-512 alignment
        let layout =
            AllocLayout::from_size_align(size_bytes, align).expect("Invalid allocation layout");

        let ptr = unsafe { alloc_zeroed(layout) };

        if ptr.is_null() {
            panic!("Failed to allocate {} bytes", size_bytes);
        }

        ptr as u64
    }

    fn deallocate(ptr: u64, size_bytes: usize, _device: &Self::Device) {
        if ptr == 0 || size_bytes == 0 {
            return;
        }

        let align = 64;
        let layout =
            AllocLayout::from_size_align(size_bytes, align).expect("Invalid allocation layout");

        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
    }

    fn copy_to_device(src: &[u8], dst: u64, _device: &Self::Device) {
        if src.is_empty() || dst == 0 {
            return;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        }
    }

    fn copy_from_device(src: u64, dst: &mut [u8], _device: &Self::Device) {
        if dst.is_empty() || src == 0 {
            return;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len());
        }
    }

    fn copy_within_device(src: u64, dst: u64, size_bytes: usize, _device: &Self::Device) {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return;
        }

        unsafe {
            // Use copy (not copy_nonoverlapping) in case src and dst overlap
            std::ptr::copy(src as *const u8, dst as *mut u8, size_bytes);
        }
    }

    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        _device: &Self::Device,
    ) {
        if src_handle == 0 || dst_handle == 0 || shape.is_empty() {
            return;
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return;
        }

        // For CPU, we can use pointer arithmetic directly
        let src_base = (src_handle as usize + src_byte_offset) as *const u8;
        let dst_base = dst_handle as *mut u8;

        // Iterate over all elements using indices
        let mut indices = vec![0usize; shape.len()];

        for dst_offset in 0..numel {
            // Calculate source byte offset for current indices
            let mut src_elem_offset: isize = 0;
            for (i, &idx) in indices.iter().enumerate() {
                src_elem_offset += (idx as isize) * strides[i];
            }

            // Copy element
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_base.offset(src_elem_offset * elem_size as isize),
                    dst_base.add(dst_offset * elem_size),
                    elem_size,
                );
            }

            // Increment indices (row-major order)
            for dim in (0..shape.len()).rev() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }
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
}

// ============================================================================
// Kernel Trait Implementation
// ============================================================================

#[allow(unsafe_op_in_unsafe_fn)] // Kernels are already marked unsafe
impl Kernel<CpuRuntime> for CpuClient {
    unsafe fn binary_op<T: Element>(
        &self,
        op: BinaryOp,
        a: *const T,
        b: *const T,
        out: *mut T,
        len: usize,
    ) {
        kernels::binary_op_kernel(op, a, b, out, len);
    }

    unsafe fn unary_op<T: Element>(&self, op: UnaryOp, a: *const T, out: *mut T, len: usize) {
        kernels::unary_op_kernel(op, a, out, len);
    }

    unsafe fn matmul<T: Element>(
        &self,
        a: *const T,
        b: *const T,
        out: *mut T,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    ) {
        kernels::matmul_kernel(a, b, out, m, n, k, lda, ldb, ldc);
    }

    unsafe fn reduce<T: Element>(
        &self,
        op: ReduceOp,
        a: *const T,
        out: *mut T,
        reduce_size: usize,
        outer_size: usize,
    ) {
        kernels::reduce_kernel(op, a, out, reduce_size, outer_size);
    }

    unsafe fn fill<T: Element>(&self, out: *mut T, value: T, len: usize) {
        kernels::fill_kernel(out, value, len);
    }

    unsafe fn copy<T: Element>(&self, src: *const T, dst: *mut T, len: usize) {
        kernels::copy_kernel(src, dst, len);
    }
}
