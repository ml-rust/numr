//! Core Tensor type

use super::{Layout, Storage, TensorId};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use std::fmt;

/// N-dimensional array stored on a compute device
///
/// `Tensor` is the fundamental data structure in numr. It consists of:
/// - **Storage**: Reference-counted device memory
/// - **Layout**: Shape, strides, and offset defining the view into storage
/// - **DType**: Element type (determined at runtime)
///
/// # Zero-Copy Views
///
/// Operations like `transpose`, `slice`, and `reshape` create new tensors
/// that share the same underlying storage. This is achieved through:
/// - Arc-wrapped storage (reference counting)
/// - Modified layout (different strides/offset)
///
/// # Example
///
/// ```ignore
/// use numr::prelude::*;
///
/// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
/// let b = a.transpose(-1, -2); // Zero-copy, shares storage with a
/// ```
pub struct Tensor<R: Runtime> {
    /// Unique ID for autograd tracking
    id: TensorId,
    /// Device memory
    storage: Storage<R>,
    /// Shape, strides, offset
    layout: Layout,
}

impl<R: Runtime> Tensor<R> {
    /// Create a tensor from storage and layout
    pub fn from_parts(storage: Storage<R>, layout: Layout) -> Self {
        Self {
            id: TensorId::new(),
            storage,
            layout,
        }
    }

    /// Create a tensor from a slice of data
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal the product of the `shape` dimensions.
    /// For a fallible alternative, use [`Self::try_from_slice`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    /// ```
    pub fn from_slice<T: Element>(data: &[T], shape: &[usize], device: &R::Device) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let storage = Storage::from_slice(data, device);
        let layout = Layout::contiguous(shape);

        Self {
            id: TensorId::new(),
            storage,
            layout,
        }
    }

    /// Create a tensor from a slice of data (fallible version)
    ///
    /// Returns an error if `data.len()` does not equal the product of the `shape` dimensions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tensor = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// ```
    pub fn try_from_slice<T: Element>(
        data: &[T],
        shape: &[usize],
        device: &R::Device,
    ) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }

        let storage = Storage::from_slice(data, device);
        let layout = Layout::contiguous(shape);

        Ok(Self {
            id: TensorId::new(),
            storage,
            layout,
        })
    }

    /// Create an uninitialized tensor
    ///
    /// # Safety
    /// The contents are uninitialized. Reading before writing is undefined behavior.
    pub fn empty(shape: &[usize], dtype: DType, device: &R::Device) -> Self {
        let len: usize = shape.iter().product();
        let storage = Storage::new(len, dtype, device);
        let layout = Layout::contiguous(shape);

        Self {
            id: TensorId::new(),
            storage,
            layout,
        }
    }

    /// Create a tensor filled with zeros
    ///
    /// Note: For CPU runtime, memory is already zero-initialized by `alloc_zeroed`.
    /// GPU backends should implement a fill kernel for efficiency.
    pub fn zeros(shape: &[usize], dtype: DType, device: &R::Device) -> Self {
        // CPU uses alloc_zeroed, so memory is already zero.
        // GPU backends will need to dispatch a fill kernel here.
        Self::empty(shape, dtype, device)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize], dtype: DType, device: &R::Device) -> Self {
        Self::full_scalar(shape, dtype, 1.0, device)
    }

    /// Create a tensor filled with a scalar value
    ///
    /// The scalar is converted to the target dtype.
    pub fn full_scalar(shape: &[usize], dtype: DType, value: f64, device: &R::Device) -> Self {
        let len: usize = shape.iter().product();
        if len == 0 {
            return Self::empty(shape, dtype, device);
        }

        // Create a host buffer with the fill value and copy to device
        let size_bytes = len * dtype.size_in_bytes();
        let mut bytes = vec![0u8; size_bytes];

        // Fill based on dtype
        match dtype {
            DType::F64 => {
                let slice: &mut [f64] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value);
            }
            DType::F32 => {
                let slice: &mut [f32] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as f32);
            }
            DType::F16 | DType::BF16 => {
                let slice: &mut [u16] = bytemuck::cast_slice_mut(&mut bytes);
                #[cfg(feature = "f16")]
                {
                    use half::{bf16, f16};
                    let half_bits = if dtype == DType::BF16 {
                        bf16::from_f32(value as f32).to_bits()
                    } else {
                        f16::from_f32(value as f32).to_bits()
                    };
                    slice.fill(half_bits);
                }
                #[cfg(not(feature = "f16"))]
                {
                    // Fallback: manual conversion when half crate unavailable
                    let half_bits = half_from_f32(value as f32, dtype);
                    slice.fill(half_bits);
                }
            }
            DType::FP8E4M3 => {
                #[cfg(feature = "fp8")]
                {
                    let fp8_val = crate::dtype::FP8E4M3::from_f32(value as f32);
                    bytes.fill(fp8_val.to_bits());
                }
                #[cfg(not(feature = "fp8"))]
                {
                    panic!(
                        "FP8E4M3 dtype requires the 'fp8' feature. \
                         Enable it with: cargo build --features fp8"
                    );
                }
            }
            DType::FP8E5M2 => {
                #[cfg(feature = "fp8")]
                {
                    let fp8_val = crate::dtype::FP8E5M2::from_f32(value as f32);
                    bytes.fill(fp8_val.to_bits());
                }
                #[cfg(not(feature = "fp8"))]
                {
                    panic!(
                        "FP8E5M2 dtype requires the 'fp8' feature. \
                         Enable it with: cargo build --features fp8"
                    );
                }
            }
            DType::I64 => {
                let slice: &mut [i64] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as i64);
            }
            DType::I32 => {
                let slice: &mut [i32] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as i32);
            }
            DType::I16 => {
                let slice: &mut [i16] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as i16);
            }
            DType::I8 => {
                let slice: &mut [i8] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as i8);
            }
            DType::U64 => {
                let slice: &mut [u64] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as u64);
            }
            DType::U32 => {
                let slice: &mut [u32] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as u32);
            }
            DType::U16 => {
                let slice: &mut [u16] = bytemuck::cast_slice_mut(&mut bytes);
                slice.fill(value as u16);
            }
            DType::U8 => {
                bytes.fill(value as u8);
            }
            DType::Bool => {
                bytes.fill(if value != 0.0 { 1 } else { 0 });
            }
        }

        // Allocate and copy to device
        let storage = Storage::from_bytes(&bytes, dtype, device);
        let layout = Layout::contiguous(shape);

        Self {
            id: TensorId::new(),
            storage,
            layout,
        }
    }

    // ===== Accessors =====

    /// Get the tensor ID
    #[inline]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Get the storage
    #[inline]
    pub fn storage(&self) -> &Storage<R> {
        &self.storage
    }

    /// Get the layout
    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Get the shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Get the strides
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Get the number of dimensions (rank)
    #[inline]
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Get the total number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.layout.elem_count()
    }

    /// Get the element type
    #[inline]
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Get the device
    #[inline]
    pub fn device(&self) -> &R::Device {
        self.storage.device()
    }

    /// Check if the tensor is contiguous in memory
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Check if this is a scalar (0-dimensional tensor)
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.layout.is_scalar()
    }

    /// Get size along a dimension (supports negative indexing)
    pub fn size(&self, dim: isize) -> Option<usize> {
        self.layout.dim(dim)
    }

    // ===== View Operations (Zero-Copy) =====

    /// Transpose two dimensions (zero-copy)
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Result<Self> {
        let new_layout =
            self.layout
                .transpose(dim0, dim1)
                .ok_or_else(|| Error::InvalidDimension {
                    dim: dim0,
                    ndim: self.ndim(),
                })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Transpose last two dimensions (matrix transpose)
    pub fn t(&self) -> Result<Self> {
        self.transpose(-2, -1)
    }

    /// Reshape to a new shape (zero-copy if contiguous)
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let new_layout = self.layout.reshape(shape).ok_or(Error::NotContiguous)?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Flatten to 1D (zero-copy if contiguous)
    pub fn flatten(&self) -> Result<Self> {
        self.reshape(&[self.numel()])
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self, dim: Option<isize>) -> Self {
        Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.squeeze(dim),
        }
    }

    /// Add a dimension of size 1
    pub fn unsqueeze(&self, dim: isize) -> Result<Self> {
        let new_layout = self
            .layout
            .unsqueeze(dim)
            .ok_or_else(|| Error::InvalidDimension {
                dim,
                ndim: self.ndim(),
            })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// View tensor with different shape (alias for reshape)
    pub fn view(&self, shape: &[usize]) -> Result<Self> {
        self.reshape(shape)
    }

    /// Broadcast to a target shape (zero-copy)
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let new_layout = self
            .layout
            .broadcast_to(shape)
            .ok_or_else(|| Error::BroadcastError {
                lhs: self.shape().to_vec(),
                rhs: shape.to_vec(),
            })?;

        Ok(Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Make tensor contiguous (copy if needed)
    ///
    /// If the tensor is already contiguous, returns a view (zero-copy).
    /// Otherwise, allocates new storage and copies the data to a contiguous layout.
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            // Need to copy data to a new contiguous storage
            let dtype = self.dtype();
            let device = self.storage.device();
            let numel = self.numel();

            // Allocate new contiguous storage
            let new_storage = Storage::new(numel, dtype, device);
            let new_layout = Layout::contiguous(self.shape());

            // Copy data element by element (generic copy via bytes)
            // Note: Host-side copy for non-contiguous tensors. GPU kernel
            // optimization would improve performance for large tensors.
            let elem_size = dtype.size_in_bytes();
            let src_ptr = self.storage.ptr();
            let dst_ptr = new_storage.ptr();

            // For non-contiguous tensors, we need to iterate over each element
            // and copy it to the correct position in the new storage
            let shape = self.shape();
            let src_strides = self.strides();
            let src_offset = self.layout.offset();

            // Use iterative approach to copy each element
            let mut indices = vec![0usize; shape.len()];

            for dst_offset in 0..numel {
                // Calculate source offset for current indices
                let mut src_elem_offset = src_offset as isize;
                for (i, &idx) in indices.iter().enumerate() {
                    src_elem_offset += (idx as isize) * src_strides[i];
                }

                // Copy element
                R::copy_within_device(
                    src_ptr + (src_elem_offset as usize * elem_size) as u64,
                    dst_ptr + (dst_offset * elem_size) as u64,
                    elem_size,
                    device,
                );

                // Increment indices (row-major order)
                for dim in (0..shape.len()).rev() {
                    indices[dim] += 1;
                    if indices[dim] < shape[dim] {
                        break;
                    }
                    indices[dim] = 0;
                }
            }

            Self {
                id: TensorId::new(),
                storage: new_storage,
                layout: new_layout,
            }
        }
    }

    /// Detach from computation graph (for autograd)
    pub fn detach(&self) -> Self {
        Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }

    // ===== Data Access =====

    /// Copy tensor data to a Vec on the host
    pub fn to_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        assert!(
            self.is_contiguous(),
            "Tensor must be contiguous to copy to vec"
        );
        self.storage.to_vec()
    }
}

impl<R: Runtime> Clone for Tensor<R> {
    /// Clone creates a new tensor sharing the same storage (zero-copy)
    fn clone(&self) -> Self {
        Self {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<R: Runtime> fmt::Debug for Tensor<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id)
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("contiguous", &self.is_contiguous())
            .finish()
    }
}

impl<R: Runtime> fmt::Display for Tensor<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({:?}, dtype={})", self.shape(), self.dtype())
    }
}

/// Convert f32 to half-precision bit representation
///
/// This is a simple conversion that handles common cases.
/// For full IEEE 754 compliance, use the `half` crate (enabled with `f16` feature).
#[cfg(not(feature = "f16"))]
fn half_from_f32(value: f32, dtype: DType) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if dtype == DType::BF16 {
        // BF16: truncate mantissa, keep exponent
        ((bits >> 16) & 0xFFFF) as u16
    } else {
        // F16: IEEE 754 half precision
        if exp == 0 {
            // Zero or subnormal
            (sign << 15) as u16
        } else if exp == 0xFF {
            // Inf or NaN
            ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16
        } else {
            // Normal number
            let new_exp = exp - 127 + 15;
            if new_exp <= 0 {
                // Underflow to zero
                (sign << 15) as u16
            } else if new_exp >= 31 {
                // Overflow to infinity
                ((sign << 15) | 0x7C00) as u16
            } else {
                ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_from_slice() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.numel(), 6);

        let result: Vec<f32> = tensor.to_vec();
        assert_eq!(result, data);
    }

    #[test]
    fn test_transpose() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device);

        let transposed = tensor.transpose(0, 1).unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert!(!transposed.is_contiguous()); // Transpose makes it non-contiguous
        assert_eq!(transposed.numel(), 6);
    }

    #[test]
    fn test_contiguous_already_contiguous() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &device);

        assert!(tensor.is_contiguous());
        let contiguous = tensor.contiguous();
        assert!(contiguous.is_contiguous());

        let result: Vec<f32> = contiguous.to_vec();
        assert_eq!(result, data);
    }

    #[test]
    fn test_contiguous_from_transpose() {
        let device = CpuDevice::new();
        // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device);

        // Transpose to 3x2: [[1, 4], [2, 5], [3, 6]]
        let transposed = tensor.transpose(0, 1).unwrap();
        assert!(!transposed.is_contiguous());

        // Make contiguous - should copy data
        let contiguous = transposed.contiguous();
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape(), &[3, 2]);

        // Verify the data is in the correct order for row-major layout
        // Row 0: [1, 4], Row 1: [2, 5], Row 2: [3, 6]
        let result: Vec<f32> = contiguous.to_vec();
        assert_eq!(result, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_reshape() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device);

        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert!(reshaped.is_contiguous());

        let result: Vec<f32> = reshaped.to_vec();
        assert_eq!(result, data); // Data unchanged, just reinterpreted
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let device = CpuDevice::new();
        let data = [1.0f32, 2.0, 3.0];
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[1, 3, 1], &device);

        // Squeeze all dimensions of size 1
        let squeezed = tensor.squeeze(None);
        assert_eq!(squeezed.shape(), &[3]);

        // Unsqueeze at dimension 0
        let unsqueezed = squeezed.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3]);
    }

    #[test]
    fn test_zeros() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::zeros(&[2, 3], DType::F32, &device);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.is_contiguous());

        let result: Vec<f32> = tensor.to_vec();
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.is_contiguous());

        let result: Vec<f32> = tensor.to_vec();
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_full_scalar() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::full_scalar(&[2, 2], DType::I32, 42.0, &device);

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), DType::I32);

        let result: Vec<i32> = tensor.to_vec();
        assert_eq!(result, [42, 42, 42, 42]);
    }
}
