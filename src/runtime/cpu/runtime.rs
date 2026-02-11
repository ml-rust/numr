//! CPU runtime implementation

use super::client::{CpuAllocator, CpuClient};
use super::device::CpuDevice;
use crate::runtime::Runtime;
use std::alloc::{Layout as AllocLayout, alloc, dealloc};

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
    type RawHandle = (); // CPU has no special handle needed

    fn name() -> &'static str {
        "cpu"
    }

    fn allocate(size_bytes: usize, _device: &Self::Device) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // Use aligned allocation for SIMD compatibility
        let align = 64; // AVX-512 alignment
        let layout = AllocLayout::from_size_align(size_bytes, align)
            .map_err(|_| crate::error::Error::OutOfMemory { size: size_bytes })?;

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

        let align = 64;
        let layout =
            AllocLayout::from_size_align(size_bytes, align).expect("Invalid allocation layout");

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
}
