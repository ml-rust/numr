//! Matrix inverse via LU decomposition

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use super::super::solvers::solve_impl;
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Matrix inverse via LU decomposition
pub fn inverse_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => inverse_typed::<f32>(client, &a, n),
        DType::F64 => inverse_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn inverse_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Create identity matrix
    let mut identity: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        identity[i * n + i] = T::one();
    }
    let identity_tensor = Tensor::<CpuRuntime>::from_slice(&identity, &[n, n], device)?;

    // Solve A @ X = I
    solve_impl(client, a, &identity_tensor)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_inverse_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[4, 7], [2, 6]]
        // det = 24 - 14 = 10
        // A^(-1) = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device).unwrap();

        let inv = client.inverse(&a).unwrap();
        let inv_data: Vec<f32> = inv.to_vec();

        assert!((inv_data[0] - 0.6).abs() < 1e-4);
        assert!((inv_data[1] - (-0.7)).abs() < 1e-4);
        assert!((inv_data[2] - (-0.2)).abs() < 1e-4);
        assert!((inv_data[3] - 0.4).abs() < 1e-4);
    }
}
