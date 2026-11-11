//! Metadata admission before `index_put` allocates or enqueues work.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::index_dtype_suffix;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

use super::super::super::tensor_inputs::check_input;

pub(super) struct IndexPutAdmission {
    pub input_count: usize,
    pub src_count: usize,
    pub outer_size: usize,
    pub dim_size: usize,
    pub inner_size: usize,
    pub index_len: usize,
}

pub(super) fn admit(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
) -> Result<IndexPutAdmission> {
    if dim >= input.ndim() {
        let reported_dim = isize::try_from(dim).map_err(|_| Error::InvalidArgument {
            arg: "dim",
            reason: format!(
                "index_put dimension {dim} exceeds isize::MAX. Select a dimension below {}.",
                input.ndim()
            ),
        })?;
        return Err(Error::InvalidDimension {
            dim: reported_dim,
            ndim: input.ndim(),
        });
    }
    if !matches!(index.dtype(), DType::I32 | DType::I64) {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }
    let index_count = check_input(client, index, "index_put")?;
    if index.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index_count],
            got: index.shape().to_vec(),
        });
    }
    if src.dtype() != input.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: input.dtype(),
            rhs: src.dtype(),
        });
    }
    let index_len = index.shape()[0];
    let mut expected = input.shape().to_vec();
    expected[dim] = index_len;
    if src.shape() != expected {
        return Err(Error::ShapeMismatch {
            expected,
            got: src.shape().to_vec(),
        });
    }
    index_dtype_suffix(input.dtype(), "index_put")?;
    let input_count = check_input(client, input, "index_put")?;
    let src_count = check_input(client, src, "index_put")?;
    index_len.checked_mul(DType::I64.size_in_bytes()).ok_or_else(|| {
        limitation(format!(
            "Normalized index count {index_len} overflows allocation bytes. Reduce the index size."
        ))
    })?;
    let outer_size = checked_product(&input.shape()[..dim])?;
    let dim_size = input.shape()[dim];
    let inner_size = checked_product(&input.shape()[dim + 1..])?;
    for (name, value) in [
        ("outer_size", outer_size),
        ("dim_size", dim_size),
        ("inner_size", inner_size),
        ("index_len", index_len),
    ] {
        u32::try_from(value).map_err(|_| {
            limitation(format!(
                "{name} {value} exceeds u32::MAX. Reduce the tensor size."
            ))
        })?;
    }
    for extent in [dim_size, index_len] {
        let count = outer_size
            .checked_mul(extent)
            .and_then(|value| value.checked_mul(inner_size))
            .ok_or_else(|| {
                limitation("Index dimensions overflow usize. Reduce the tensor size.".into())
            })?;
        u32::try_from(count).map_err(|_| {
            limitation(format!(
                "Count {count} exceeds u32::MAX. Reduce the tensor size."
            ))
        })?;
    }
    // Tensor::empty constructs signed contiguous strides even when the count is zero.
    input
        .shape()
        .iter()
        .rev()
        .try_fold(1isize, |stride, &extent| {
            isize::try_from(extent)
                .ok()
                .and_then(|extent| stride.checked_mul(extent))
                .ok_or_else(|| {
                    limitation(format!(
                        "Output shape {:?} overflows contiguous strides. Reduce the tensor size.",
                        input.shape()
                    ))
                })
        })?;
    Ok(IndexPutAdmission {
        input_count,
        src_count,
        outer_size,
        dim_size,
        inner_size,
        index_len,
    })
}

fn checked_product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count.checked_mul(dimension).ok_or_else(|| {
            limitation(format!(
                "Shape {shape:?} overflows usize. Reduce the tensor size."
            ))
        })
    })
}

fn limitation(reason: String) -> Error {
    Error::BackendLimitation {
        backend: "cuda",
        operation: "index_put",
        reason,
    }
}
