//! Storage: device memory management with Arc-based sharing

use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::Runtime;
use std::sync::Arc;

/// Storage for tensor data on a device
///
/// Storage wraps device memory with reference counting, enabling zero-copy
/// views (transpose, slice, etc.) that share the underlying buffer.
///
/// Memory is automatically deallocated when the last reference is dropped.
pub struct Storage<R: Runtime> {
    inner: Arc<StorageInner<R>>,
}

struct StorageInner<R: Runtime> {
    /// Raw device pointer (GPU address or CPU ptr cast to u64)
    ptr: u64,
    /// Number of elements (not bytes)
    len: usize,
    /// Element type
    dtype: DType,
    /// Device where memory is allocated
    device: R::Device,
    /// If true, we own this memory and should deallocate on drop
    owned: bool,
}

impl<R: Runtime> Storage<R> {
    /// Create new storage with allocated memory
    ///
    /// Allocates `len` elements of type `dtype` on the specified device.
    pub fn new(len: usize, dtype: DType, device: &R::Device) -> Result<Self> {
        let size_bytes = len * dtype.size_in_bytes();
        let ptr = R::allocate(size_bytes, device)?;

        Ok(Self {
            inner: Arc::new(StorageInner {
                ptr,
                len,
                dtype,
                device: device.clone(),
                owned: true,
            }),
        })
    }

    /// Create storage from existing data with inferred dtype
    ///
    /// Copies `data` to the device. The dtype is inferred from the Element type.
    pub fn from_slice<T: Element>(data: &[T], device: &R::Device) -> Result<Self> {
        let dtype = T::DTYPE;
        let len = data.len();

        // Copy data to device
        let bytes = bytemuck::cast_slice(data);
        let size_bytes = bytes.len();
        let ptr = R::allocate(size_bytes, device)?;

        R::copy_to_device(bytes, ptr, device)?;

        Ok(Self {
            inner: Arc::new(StorageInner {
                ptr,
                len,
                dtype,
                device: device.clone(),
                owned: true,
            }),
        })
    }

    /// Create storage from raw bytes with explicit dtype
    ///
    /// Use this when you have raw bytes and know the dtype.
    pub fn from_bytes(data: &[u8], dtype: DType, device: &R::Device) -> Result<Self> {
        let len = data.len() / dtype.size_in_bytes();
        let ptr = R::allocate(data.len(), device)?;

        R::copy_to_device(data, ptr, device)?;

        Ok(Self {
            inner: Arc::new(StorageInner {
                ptr,
                len,
                dtype,
                device: device.clone(),
                owned: true,
            }),
        })
    }

    /// Wrap existing device memory without taking ownership
    ///
    /// # Safety
    /// - `ptr` must point to valid device memory
    /// - The memory must remain valid for the lifetime of this Storage
    /// - Caller is responsible for eventual deallocation
    pub unsafe fn from_ptr(ptr: u64, len: usize, dtype: DType, device: &R::Device) -> Self {
        Self {
            inner: Arc::new(StorageInner {
                ptr,
                len,
                dtype,
                device: device.clone(),
                owned: false,
            }),
        }
    }

    /// Get the raw device pointer
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.inner.ptr
    }

    /// Get the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Check if storage is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Get the element type
    #[inline]
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    /// Get the device
    #[inline]
    pub fn device(&self) -> &R::Device {
        &self.inner.device
    }

    /// Get size in bytes
    #[inline]
    pub fn size_in_bytes(&self) -> usize {
        self.inner.len * self.inner.dtype.size_in_bytes()
    }

    /// Get the reference count
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Check if this is the only reference
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Get as raw buffer for passing to operations
    #[inline]
    pub fn as_raw(&self) -> RawBuffer {
        RawBuffer {
            ptr: self.inner.ptr,
            len: self.inner.len,
            dtype: self.inner.dtype,
        }
    }

    /// Copy data from device to host
    pub fn to_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        // Allocate with correct alignment for T, then cast to bytes for copy.
        // This avoids alignment violations that would occur if we allocated
        // a Vec<u8> and cast to stricter-aligned types like f64/i64.
        let mut result = vec![T::zeroed(); self.inner.len];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        R::copy_from_device(self.inner.ptr, bytes, &self.inner.device)
            .expect("copy_from_device failed in to_vec()");
        result
    }
}

impl<R: Runtime> Clone for Storage<R> {
    /// Clone increments the reference count (zero-copy)
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<R: Runtime> Drop for StorageInner<R> {
    fn drop(&mut self) {
        if self.owned && self.ptr != 0 {
            R::deallocate(
                self.ptr,
                self.len * self.dtype.size_in_bytes(),
                &self.device,
            );
        }
    }
}

impl<R: Runtime> std::fmt::Debug for Storage<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("ptr", &format!("0x{:x}", self.inner.ptr))
            .field("len", &self.inner.len)
            .field("dtype", &self.inner.dtype)
            .field("owned", &self.inner.owned)
            .field("refs", &Arc::strong_count(&self.inner))
            .finish()
    }
}

/// Raw buffer for passing to operations
///
/// This is a simple struct that can be passed across FFI boundaries
/// without lifetime complications. Contains all info needed by kernels.
#[derive(Copy, Clone, Debug)]
pub struct RawBuffer {
    /// Device pointer
    pub ptr: u64,
    /// Number of elements
    pub len: usize,
    /// Element type
    pub dtype: DType,
}

impl RawBuffer {
    /// Create a new raw buffer
    #[inline]
    pub const fn new(ptr: u64, len: usize, dtype: DType) -> Self {
        Self { ptr, len, dtype }
    }

    /// Create an empty raw buffer
    #[inline]
    pub const fn empty() -> Self {
        Self {
            ptr: 0,
            len: 0,
            dtype: DType::F32,
        }
    }

    /// Size in bytes
    #[inline]
    pub const fn size_in_bytes(&self) -> usize {
        self.len * self.dtype.size_in_bytes()
    }
}

// Storage tests are in runtime module (require concrete runtime implementation)
