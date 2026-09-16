//! Determinant via LU decomposition

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

/// Determinant via LU decomposition
pub fn det_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => det_typed::<f32>(client, &a, n),
        DType::F64 => det_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn det_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle special case n=0
    if n == 0 {
        return Tensor::<CpuRuntime>::from_slice(&[T::one()], &[], device);
    }

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl(client, a)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();

    // det = (-1)^num_swaps * product(U[i,i])
    let mut det = if lu_decomp.num_swaps % 2 == 0 {
        T::one()
    } else {
        T::one().neg_val()
    };

    for i in 0..n {
        det = det * lu_data[i * n + i];
    }

    Tensor::<CpuRuntime>::from_slice(&[det], &[], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_det_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[4, 3], [6, 3]]
        // det = 4*3 - 3*6 = 12 - 18 = -6
        let a =
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device).unwrap();

        let det = client.det(&a).unwrap();
        let det_val: Vec<f32> = det.to_vec();

        assert!((det_val[0] - (-6.0)).abs() < 1e-5);
    }
}
