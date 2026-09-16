//! Matrix norms: Frobenius, spectral, nuclear

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use super::super::svd::svd_decompose_impl;
use crate::algorithm::linalg::{
    MatrixNormOrder, linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Matrix norm implementation
pub fn matrix_norm_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    ord: MatrixNormOrder,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (_m, _n) = validate_matrix_2d(a.shape())?;

    let result = match ord {
        MatrixNormOrder::Frobenius => match a.dtype() {
            DType::F32 => frobenius_norm_typed::<f32>(client, &a),
            DType::F64 => frobenius_norm_typed::<f64>(client, &a),
            _ => unreachable!(),
        },
        MatrixNormOrder::Spectral => match a.dtype() {
            DType::F32 => spectral_norm_typed::<f32>(client, &a),
            DType::F64 => spectral_norm_typed::<f64>(client, &a),
            _ => unreachable!(),
        },
        MatrixNormOrder::Nuclear => match a.dtype() {
            DType::F32 => nuclear_norm_typed::<f32>(client, &a),
            DType::F64 => nuclear_norm_typed::<f64>(client, &a),
            _ => unreachable!(),
        },
    }?;

    linalg_demote(client, result, original_dtype)
}

/// Frobenius norm: ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2)
fn frobenius_norm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    // Sum of squares of all elements
    let mut sum_sq = T::zero();
    for &val in &a_data {
        sum_sq = sum_sq + val * val;
    }

    let norm = sum_sq.sqrt_val();
    Tensor::<CpuRuntime>::from_slice(&[norm], &[], device)
}

/// Spectral norm: ||A||_2 = max(singular_values(A))
fn spectral_norm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute SVD to get singular values
    let svd = svd_decompose_impl(client, a)?;
    let s_data: Vec<T> = svd.s.to_vec();

    // Find maximum singular value
    let mut max_sv = T::zero();
    for &val in &s_data {
        if val.to_f64() > max_sv.to_f64() {
            max_sv = val;
        }
    }

    Tensor::<CpuRuntime>::from_slice(&[max_sv], &[], device)
}

/// Nuclear norm: ||A||_* = sum(singular_values(A))
fn nuclear_norm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute SVD to get singular values
    let svd = svd_decompose_impl(client, a)?;
    let s_data: Vec<T> = svd.s.to_vec();

    // Sum of singular values
    let mut sum_sv = T::zero();
    for &val in &s_data {
        sum_sv = sum_sv + val;
    }

    Tensor::<CpuRuntime>::from_slice(&[sum_sv], &[], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_frobenius_norm_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]]
        // ||A||_F = sqrt(1² + 2² + 3² + 4²) = sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let norm = client.matrix_norm(&a, MatrixNormOrder::Frobenius).unwrap();
        let norm_val: Vec<f32> = norm.to_vec();

        let expected = (30.0f32).sqrt();
        assert!(
            (norm_val[0] - expected).abs() < 1e-5,
            "Frobenius norm = {} should be {}",
            norm_val[0],
            expected
        );
    }

    #[test]
    fn test_frobenius_norm_3x3() {
        let client = create_client();
        let device = client.device();

        // Identity matrix: ||I||_F = sqrt(3) for 3x3
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let norm = client.matrix_norm(&a, MatrixNormOrder::Frobenius).unwrap();
        let norm_val: Vec<f32> = norm.to_vec();

        let expected = (3.0f32).sqrt();
        assert!(
            (norm_val[0] - expected).abs() < 1e-5,
            "Frobenius norm of 3x3 identity = {} should be {}",
            norm_val[0],
            expected
        );
    }

    #[test]
    fn test_spectral_norm() {
        let client = create_client();
        let device = client.device();

        // Test on a 2x2 matrix [[1, 2], [3, 4]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let result = client.matrix_norm(&a, MatrixNormOrder::Spectral).unwrap();
        let norm_val: Vec<f32> = result.to_vec();

        // Spectral norm is the largest singular value
        // For [[1,2],[3,4]], singular values are approximately 5.465 and 0.366
        assert!(
            (norm_val[0] - 5.465).abs() < 0.01,
            "Spectral norm of [[1,2],[3,4]] = {} should be ~5.465",
            norm_val[0]
        );
    }

    #[test]
    fn test_nuclear_norm() {
        let client = create_client();
        let device = client.device();

        // Test on a 2x2 matrix [[1, 2], [3, 4]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let result = client.matrix_norm(&a, MatrixNormOrder::Nuclear).unwrap();
        let norm_val: Vec<f32> = result.to_vec();

        // Nuclear norm is sum of singular values
        // For [[1,2],[3,4]], singular values are approximately 5.465 and 0.366
        // Sum is approximately 5.831
        assert!(
            (norm_val[0] - 5.831).abs() < 0.01,
            "Nuclear norm of [[1,2],[3,4]] = {} should be ~5.831",
            norm_val[0]
        );
    }

    #[test]
    fn test_spectral_norm_identity() {
        let client = create_client();
        let device = client.device();

        // Identity matrix has spectral norm = 1 (all singular values = 1)
        let eye =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], device).unwrap();

        let result = client.matrix_norm(&eye, MatrixNormOrder::Spectral).unwrap();
        let norm_val: Vec<f32> = result.to_vec();

        assert!(
            (norm_val[0] - 1.0).abs() < 1e-5,
            "Spectral norm of 2x2 identity = {} should be 1.0",
            norm_val[0]
        );
    }

    #[test]
    fn test_nuclear_norm_identity() {
        let client = create_client();
        let device = client.device();

        // 3x3 Identity matrix has nuclear norm = 3 (sum of 3 singular values = 1)
        let eye = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.matrix_norm(&eye, MatrixNormOrder::Nuclear).unwrap();
        let norm_val: Vec<f32> = result.to_vec();

        assert!(
            (norm_val[0] - 3.0).abs() < 1e-5,
            "Nuclear norm of 3x3 identity = {} should be 3.0",
            norm_val[0]
        );
    }
}
