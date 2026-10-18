//! Square linear system solver (Ax=b)

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::decompositions::lu_decompose_impl;
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Solve Ax = b using LU decomposition
pub fn solve_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (b, _) = linalg_promote(client, b)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => solve_typed::<f32>(client, &a, &b, n),
        DType::F64 => solve_typed::<f64>(client, &a, &b, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn solve_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl(client, a)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();
    let pivots_data: Vec<i64> = lu_decomp.pivots.to_vec();

    // Handle 1D or 2D b
    let b_shape = b.shape();
    let (b_rows, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    if b_rows != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: vec![b_rows],
        });
    }

    let b_data: Vec<T> = b.to_vec();
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        // Apply permutation to b
        let mut pb: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            pb[i] = if num_rhs == 1 {
                b_data[i]
            } else {
                b_data[i * num_rhs + rhs]
            };
        }
        for (i, &pivot_idx) in pivots_data.iter().enumerate() {
            let pivot_row = pivot_idx as usize;
            if pivot_row != i {
                pb.swap(i, pivot_row);
            }
        }

        // Forward substitution: Ly = Pb (L has unit diagonal)
        let mut y: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                sum = sum + lu_data[i * n + j] * y[j];
            }
            y[i] = pb[i] - sum;
        }

        // Backward substitution: Ux = y
        let mut x_col: Vec<T> = vec![T::zero(); n];
        for ii in (0..n).rev() {
            let mut s = T::zero();
            for jj in (ii + 1)..n {
                s = s + lu_data[ii * n + jj] * x_col[jj];
            }
            x_col[ii] = (y[ii] - s) / lu_data[ii * n + ii];
        }
        // Copy result
        for ii in 0..n {
            if num_rhs == 1 {
                x[ii] = x_col[ii];
            } else {
                x[ii * num_rhs + rhs] = x_col[ii];
            }
        }
    }

    // Create output tensor
    if b_shape.len() == 1 {
        Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device)
    } else {
        Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_solve_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[2, 1], [1, 2]], b = [3, 3]
        // Solution: x = [1, 1]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 3.0], &[2], device).unwrap();

        let x = client.solve(&a, &b).unwrap();
        let x_data: Vec<f32> = x.to_vec();

        // Check solution is approximately [1, 1]
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 1.0).abs() < 1e-5);
    }
}
