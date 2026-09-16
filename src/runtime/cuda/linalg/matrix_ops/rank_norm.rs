//! Matrix rank and matrix norm for CUDA

use super::super::super::CudaRuntime;
use super::super::super::client::CudaClient;
use super::super::svd::svd_decompose_impl;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, MatrixNormOrder, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{CompareOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use crate::tensor::Tensor;

/// Matrix rank via QR decomposition - runs entirely on GPU (zero CPU transfers)
pub fn matrix_rank_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    tol: Option<f64>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let k = m.min(n);

    // Handle empty matrix
    if k == 0 {
        return Tensor::<CudaRuntime>::from_slice(&[0i64], &[], a.device());
    }

    // Compute tolerance factor (depends only on dimensions, no GPU data needed)
    let base_tol = tol.unwrap_or_else(|| {
        let eps = match dtype {
            DType::F32 => f32::EPSILON as f64,
            DType::F64 => f64::EPSILON,
            _ => f32::EPSILON as f64,
        };
        (m.max(n) as f64) * eps
    });

    // Use QR decomposition to estimate rank
    let qr = client.qr_decompose(a)?;

    // Get diagonal of R
    let r_diag = LinearAlgebraAlgorithms::diag(client, &qr.r)?;

    // Compute abs(r_diag) on GPU
    let abs_diag = client.abs(&r_diag)?;

    // Compute max(abs(r_diag)) on GPU - returns scalar tensor
    let max_val = client.max(&abs_diag, &[], false)?;

    // Compute threshold = base_tol * max on GPU
    let threshold = client.mul_scalar(&max_val, base_tol)?;

    // Compare abs_diag > threshold on GPU (broadcasts threshold)
    // CUDA comparisons return same dtype (0.0/1.0), not Bool
    let above_mask = client.gt(&abs_diag, &threshold)?;

    // Sum the mask directly (values are 0.0 or 1.0)
    let rank_float = client.sum(&above_mask, &[], false)?;

    // Cast to I64 for integer result
    let rank_tensor = client.cast(&rank_float, DType::I64)?;

    Ok(rank_tensor)
}

/// Matrix norm
pub fn matrix_norm_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    ord: MatrixNormOrder,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (_m, _n) = validate_matrix_2d(a.shape())?;

    match ord {
        MatrixNormOrder::Frobenius => {
            // Frobenius norm: ||A||_F = sqrt(sum(A²))
            // Use existing tensor ops to keep data on GPU
            let squared = client.square(a)?;
            let sum_sq = client.sum(&squared, &[], false)?;
            client.sqrt(&sum_sq)
        }
        MatrixNormOrder::Spectral => {
            // Spectral norm: ||A||_2 = max(singular_values(A))
            let svd = svd_decompose_impl(client, a)?;
            client.max(&svd.s, &[], false)
        }
        MatrixNormOrder::Nuclear => {
            // Nuclear norm: ||A||_* = sum(singular_values(A))
            let svd = svd_decompose_impl(client, a)?;
            client.sum(&svd.s, &[], false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::runtime::RuntimeClient;

    #[test]
    fn test_matrix_rank_full() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // Full rank 2x2 matrix
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let rank = LinearAlgebraAlgorithms::matrix_rank(&client, &a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_matrix_rank_deficient() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // Rank-deficient 2x2 matrix (rows are linearly dependent)
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device).unwrap();

        let rank = LinearAlgebraAlgorithms::matrix_rank(&client, &a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 1);
    }
}
