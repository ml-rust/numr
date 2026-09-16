//! Sign and log-absolute-determinant via LU decomposition

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use super::super::decompositions::lu_decompose_impl;
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Sign and log-absolute-determinant via LU decomposition
pub fn slogdet_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<crate::algorithm::linalg::SlogdetResult<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => slogdet_typed::<f32>(client, &a, n),
        DType::F64 => slogdet_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    Ok(crate::algorithm::linalg::SlogdetResult {
        sign: linalg_demote(client, result.sign, original_dtype)?,
        logabsdet: linalg_demote(client, result.logabsdet, original_dtype)?,
    })
}

fn slogdet_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<crate::algorithm::linalg::SlogdetResult<CpuRuntime>> {
    let device = client.device();

    // Handle special case n=0: det of empty matrix is 1 by convention
    if n == 0 {
        return Ok(crate::algorithm::linalg::SlogdetResult {
            sign: Tensor::<CpuRuntime>::from_slice(&[T::one()], &[], device)?,
            logabsdet: Tensor::<CpuRuntime>::from_slice(&[T::zero()], &[], device)?,
        });
    }

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl(client, a)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();

    // Start with sign from row swaps
    let mut sign_val: f64 = if lu_decomp.num_swaps % 2 == 0 {
        1.0
    } else {
        -1.0
    };

    let mut logabsdet_val: f64 = 0.0;
    let mut is_zero = false;

    for i in 0..n {
        let diag = lu_data[i * n + i].to_f64();
        if diag == 0.0 {
            is_zero = true;
            break;
        }
        if diag < 0.0 {
            sign_val = -sign_val;
        }
        logabsdet_val += diag.abs().ln();
    }

    if is_zero {
        sign_val = 0.0;
        logabsdet_val = f64::NEG_INFINITY;
    }

    Ok(crate::algorithm::linalg::SlogdetResult {
        sign: Tensor::<CpuRuntime>::from_slice(&[T::from_f64(sign_val)], &[], device)?,
        logabsdet: Tensor::<CpuRuntime>::from_slice(&[T::from_f64(logabsdet_val)], &[], device)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_slogdet_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]], det = 1*4 - 2*3 = -2
        // sign = -1, logabsdet = ln(2)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let result = client.slogdet(&a).unwrap();

        let sign_data: Vec<f32> = result.sign.to_vec();
        let logabsdet_data: Vec<f32> = result.logabsdet.to_vec();

        assert!(
            (sign_data[0] - (-1.0)).abs() < 1e-5,
            "sign should be -1, got {}",
            sign_data[0]
        );
        assert!(
            (logabsdet_data[0] - 2.0f32.ln()).abs() < 1e-5,
            "logabsdet should be ln(2) ≈ {}, got {}",
            2.0f32.ln(),
            logabsdet_data[0]
        );
    }

    #[test]
    fn test_slogdet_3x3_identity() {
        let client = create_client();
        let device = client.device();

        // Identity matrix: det = 1, sign = 1, logabsdet = 0
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.slogdet(&a).unwrap();

        let sign_data: Vec<f32> = result.sign.to_vec();
        let logabsdet_data: Vec<f32> = result.logabsdet.to_vec();

        assert!(
            (sign_data[0] - 1.0).abs() < 1e-5,
            "sign should be 1, got {}",
            sign_data[0]
        );
        assert!(
            logabsdet_data[0].abs() < 1e-5,
            "logabsdet should be 0, got {}",
            logabsdet_data[0]
        );
    }

    #[test]
    fn test_slogdet_positive_det() {
        let client = create_client();
        let device = client.device();

        // A = [[2, 0], [0, 3]], det = 6
        // sign = 1, logabsdet = ln(6)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.0, 0.0, 3.0], &[2, 2], device).unwrap();

        let result = client.slogdet(&a).unwrap();

        let sign_data: Vec<f32> = result.sign.to_vec();
        let logabsdet_data: Vec<f32> = result.logabsdet.to_vec();

        assert!(
            (sign_data[0] - 1.0).abs() < 1e-5,
            "sign should be 1, got {}",
            sign_data[0]
        );
        assert!(
            (logabsdet_data[0] - 6.0f32.ln()).abs() < 1e-4,
            "logabsdet should be ln(6) ≈ {}, got {}",
            6.0f32.ln(),
            logabsdet_data[0]
        );
    }
}
