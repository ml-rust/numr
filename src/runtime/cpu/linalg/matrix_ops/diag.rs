//! Extract diagonal / create diagonal matrix

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Extract diagonal
pub fn diag_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => diag_typed::<f32>(client, &a, m, n),
        DType::F64 => diag_typed::<f64>(client, &a, m, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn diag_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut diag: Vec<T> = vec![T::zero(); k];
    for i in 0..k {
        diag[i] = a_data[i * n + i];
    }

    Tensor::<CpuRuntime>::from_slice(&diag, &[k], device)
}

/// Create diagonal matrix from 1D tensor
pub fn diagflat_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.ndim() != 1 {
        return Err(Error::Internal(format!(
            "diagflat expects 1D tensor, got {}D",
            a.ndim()
        )));
    }
    let (a, original_dtype) = linalg_promote(client, a)?;

    let result = match a.dtype() {
        DType::F32 => diagflat_typed::<f32>(client, &a),
        DType::F64 => diagflat_typed::<f64>(client, &a),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn diagflat_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let n = a.shape()[0];
    let a_data: Vec<T> = a.to_vec();

    let mut mat: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        mat[i * n + i] = a_data[i];
    }

    Tensor::<CpuRuntime>::from_slice(&mat, &[n, n], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_diag() {
        let client = create_client();
        let device = client.device();

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                .unwrap();

        let d = client.diag(&a).unwrap();
        let d_data: Vec<f32> = d.to_vec();

        assert_eq!(d_data.len(), 2); // min(2, 3)
        assert!((d_data[0] - 1.0).abs() < 1e-5);
        assert!((d_data[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diagflat() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

        let mat = client.diagflat(&a).unwrap();
        let mat_data: Vec<f32> = mat.to_vec();

        // Should be 3x3 with [1, 2, 3] on diagonal
        assert_eq!(mat.shape(), &[3, 3]);
        assert!((mat_data[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((mat_data[4] - 2.0).abs() < 1e-5); // [1,1]
        assert!((mat_data[8] - 3.0).abs() < 1e-5); // [2,2]
        // Off-diagonal should be zero
        assert!((mat_data[1]).abs() < 1e-5);
        assert!((mat_data[2]).abs() < 1e-5);
    }
}
