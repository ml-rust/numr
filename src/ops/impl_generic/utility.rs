//! Generic implementations of utility composite operations.
//!
//! These implementations are shared across GPU backends (CUDA, WebGPU) to ensure
//! numerical parity and eliminate code duplication.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{CompareOps, TypeConversionOps, UtilityOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// One-hot encoding via broadcasting and comparison (GPU-native)
///
/// Algorithm:
/// 1. Cast integer indices to F32 for comparison
/// 2. Create class indices `[0, 1, ..., num_classes-1]`
/// 3. Reshape indices `[...] -> [..., 1]` and class indices `[1, ..., 1, num_classes]`
/// 4. Broadcast compare: `indices == class_indices`
/// 5. Cast boolean mask to F32
pub fn one_hot_impl<R, C>(client: &C, indices: &Tensor<R>, num_classes: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: UtilityOps<R> + TypeConversionOps<R> + CompareOps<R>,
{
    if num_classes == 0 {
        return Err(Error::InvalidArgument {
            arg: "num_classes",
            reason: "one_hot requires num_classes > 0".to_string(),
        });
    }

    let idx_dtype = indices.dtype();
    if !idx_dtype.is_int() {
        return Err(Error::UnsupportedDType {
            dtype: idx_dtype,
            op: "one_hot (requires integer indices)",
        });
    }

    let numel: usize = indices.shape().iter().product();
    if numel == 0 {
        let mut out_shape = indices.shape().to_vec();
        out_shape.push(num_classes);
        return Ok(Tensor::<R>::empty(&out_shape, DType::F32, indices.device()));
    }

    // Cast indices to F32 for comparison (GPU compare ops work on same dtype)
    let indices_f32 = if idx_dtype != DType::F32 {
        client.cast(indices, DType::F32)?
    } else {
        indices.clone()
    };

    // Create class indices [0, 1, ..., num_classes-1]
    let class_indices = client.arange(0.0, num_classes as f64, 1.0, DType::F32)?;

    // Reshape for broadcasting: indices [...] -> [..., 1], class_indices -> [1, ..., 1, num_classes]
    let mut idx_shape = indices.shape().to_vec();
    idx_shape.push(1);
    let indices_expanded = indices_f32.reshape(&idx_shape)?;

    let class_shape: Vec<usize> = std::iter::repeat(1)
        .take(indices.shape().len())
        .chain(std::iter::once(num_classes))
        .collect();
    let class_expanded = class_indices.reshape(&class_shape)?;

    // Compare: indices == class_indices (broadcasts to [..., num_classes])
    let mask = client.eq(&indices_expanded, &class_expanded)?;

    // Cast bool mask to F32
    client.cast(&mask, DType::F32)
}
