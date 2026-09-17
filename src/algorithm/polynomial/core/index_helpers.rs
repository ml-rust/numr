//! Index tensor construction helpers shared across polynomial algorithm implementations.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Create a single-element index tensor with the specified dtype
///
/// # Arguments
///
/// * `index` - The index value
/// * `index_dtype` - The dtype for the index tensor (I32 or I64)
/// * `device` - The device to create the tensor on
pub(crate) fn create_index_tensor<R: Runtime<DType = DType>>(
    index: usize,
    index_dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    match index_dtype {
        DType::I32 => Tensor::<R>::from_slice(&[index as i32], &[1], device),
        _ => Tensor::<R>::from_slice(&[index as i64], &[1], device),
    }
}

/// Create an arange-like index tensor [start, start+1, ..., end-1]
///
/// # Arguments
///
/// * `start` - Start index (inclusive)
/// * `end` - End index (exclusive)
/// * `index_dtype` - The dtype for the index tensor (I32 or I64)
/// * `device` - The device to create the tensor on
pub(crate) fn create_arange_tensor<R: Runtime<DType = DType>>(
    start: usize,
    end: usize,
    index_dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    match index_dtype {
        DType::I32 => {
            let indices: Vec<i32> = (start..end).map(|i| i as i32).collect();
            Tensor::<R>::from_slice(&indices, &[indices.len()], device)
        }
        _ => {
            let indices: Vec<i64> = (start..end).map(|i| i as i64).collect();
            Tensor::<R>::from_slice(&indices, &[indices.len()], device)
        }
    }
}
