//! Generic implementations of linear algebra composite operations.
//!
//! These implementations are shared across GPU backends (CUDA, WebGPU) to ensure
//! numerical parity and eliminate code duplication. All operations stay entirely
//! on the device — NO GPU-to-CPU data transfers.

use crate::algorithm::linalg::helpers::{
    validate_linalg_dtype, validate_matrix_2d, validate_square_matrix,
};
use crate::algorithm::linalg::{LinearAlgebraAlgorithms, SlogdetResult};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, CompareOps, ReduceOps, ScalarOps, TypeConversionOps, UtilityOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Which triangle to extract.
enum Triangle {
    Upper,
    Lower,
}

/// Shared implementation for triangular matrix extraction via mask composition.
///
/// Creates row/column index tensors, broadcasts a comparison mask, and
/// multiplies element-wise to zero out the unwanted triangle.
fn triangular_mask_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    diagonal: i64,
    triangle: Triangle,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: UtilityOps<R> + ScalarOps<R> + CompareOps<R> + TypeConversionOps<R> + BinaryOps<R>,
{
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();

    let row_indices = client
        .arange(0.0, m as f64, 1.0, DType::F32)?
        .reshape(&[m, 1])?;
    let col_indices = client
        .arange(0.0, n as f64, 1.0, DType::F32)?
        .reshape(&[1, n])?;
    let row_plus_diag = client.add_scalar(&row_indices, diagonal as f64)?;

    let mask = match triangle {
        Triangle::Upper => client.ge(&col_indices, &row_plus_diag)?,
        Triangle::Lower => client.le(&col_indices, &row_plus_diag)?,
    };

    let mask_typed = if dtype != DType::F32 {
        client.cast(&mask, dtype)?
    } else {
        mask
    };

    client.mul(a, &mask_typed)
}

/// Upper triangular part of a matrix (GPU-native via mask composition).
///
/// Supports all numeric dtypes the backend can handle.
pub fn triu_impl<R, C>(client: &C, a: &Tensor<R>, diagonal: i64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: UtilityOps<R> + ScalarOps<R> + CompareOps<R> + TypeConversionOps<R> + BinaryOps<R>,
{
    triangular_mask_impl(client, a, diagonal, Triangle::Upper)
}

/// Lower triangular part of a matrix (GPU-native via mask composition).
///
/// Supports all numeric dtypes the backend can handle.
pub fn tril_impl<R, C>(client: &C, a: &Tensor<R>, diagonal: i64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: UtilityOps<R> + ScalarOps<R> + CompareOps<R> + TypeConversionOps<R> + BinaryOps<R>,
{
    triangular_mask_impl(client, a, diagonal, Triangle::Lower)
}

/// Sign and log-absolute-determinant via LU decomposition (GPU-native).
///
/// Computes `(sign, logabsdet)` where `det(A) = sign * exp(logabsdet)`.
/// Uses LU decomposition to extract diagonal, then computes sign from
/// diagonal element signs and permutation parity.
pub fn slogdet_impl<R, C>(client: &C, a: &Tensor<R>) -> Result<SlogdetResult<R>>
where
    R: Runtime,
    C: LinearAlgebraAlgorithms<R>
        + UtilityOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + crate::ops::UnaryOps<R>,
{
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();

    if n == 0 {
        let sign = client.fill(&[], 1.0, dtype)?;
        let logabsdet = client.fill(&[], 0.0, dtype)?;
        return Ok(SlogdetResult { sign, logabsdet });
    }

    let lu_result = client.lu_decompose(a)?;
    let diag = LinearAlgebraAlgorithms::diag(client, &lu_result.lu)?;

    // logabsdet = sum(log(|diag|))
    let abs_diag = client.abs(&diag)?;
    let log_abs_diag = client.log(&abs_diag)?;
    let logabsdet = client.sum(&log_abs_diag, &[], false)?;

    // sign = product(sign(diag_i)) * (-1)^num_swaps
    let zero = client.fill(&[], 0.0, dtype)?;
    let pos_mask = client.gt(&diag, &zero)?;
    let neg_mask = client.lt(&diag, &zero)?;
    let sign_per_elem = client.sub(&pos_mask, &neg_mask)?;
    let sign_product = client.prod(&sign_per_elem, &[], false)?;

    let swap_sign = if lu_result.num_swaps % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    let sign = client.mul_scalar(&sign_product, swap_sign)?;

    Ok(SlogdetResult { sign, logabsdet })
}
