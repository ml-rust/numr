//! Matrix rank via QR-based singular value thresholding

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use super::super::decompositions::qr_decompose_impl;
use crate::algorithm::linalg::{linalg_promote, validate_linalg_dtype, validate_matrix_2d};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Matrix rank via singular value thresholding
/// Uses QR-based approach since SVD is not yet implemented
pub fn matrix_rank_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    tol: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, _original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    // matrix_rank returns I64 (integer rank) - no demotion needed
    match a.dtype() {
        DType::F32 => matrix_rank_typed::<f32>(client, &a, m, n, tol),
        DType::F64 => matrix_rank_typed::<f64>(client, &a, m, n, tol),
        _ => unreachable!(),
    }
}

fn matrix_rank_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    tol: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Use QR decomposition to estimate rank (R diagonal gives singular value bounds)
    let qr = qr_decompose_impl(client, a, true)?;
    let r_data: Vec<T> = qr.r.to_vec();

    let k = m.min(n);

    // Find max diagonal element of R (upper bound on max singular value)
    let mut max_diag = T::zero();
    for i in 0..k {
        let val = r_data[i * n + i].abs_val();
        if val.to_f64() > max_diag.to_f64() {
            max_diag = val;
        }
    }

    // Compute tolerance
    let eps = if a.dtype() == DType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    let tolerance = tol.unwrap_or_else(|| m.max(n) as f64 * eps * max_diag.to_f64());

    // Count diagonal elements above tolerance
    let mut rank = 0i64;
    for i in 0..k {
        if r_data[i * n + i].abs_val().to_f64() > tolerance {
            rank += 1;
        }
    }

    Tensor::<CpuRuntime>::from_slice(&[rank], &[], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_matrix_rank_full_rank() {
        let client = create_client();
        let device = client.device();

        // Full rank 2x2 matrix
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let rank = client.matrix_rank(&a, None).unwrap();
        let rank_val: Vec<i64> = rank.to_vec();

        assert_eq!(rank_val[0], 2, "Full rank 2x2 matrix should have rank 2");
    }

    #[test]
    fn test_matrix_rank_rank_deficient() {
        let client = create_client();
        let device = client.device();

        // Rank-deficient 2x2 matrix: second row is multiple of first
        // [[1, 2], [2, 4]] has rank 1
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device).unwrap();

        let rank = client.matrix_rank(&a, None).unwrap();
        let rank_val: Vec<i64> = rank.to_vec();

        assert_eq!(rank_val[0], 1, "Rank-deficient matrix should have rank 1");
    }
}
