//! Polar Decomposition: A = U @ P

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, PolarDecomposition, linalg_demote, linalg_promote,
    validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Polar decomposition: A = U @ P using SVD
pub fn polar_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<PolarDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => polar_decompose_typed::<f32>(client, &a, n),
        DType::F64 => polar_decompose_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    Ok(PolarDecomposition {
        u: linalg_demote(client, result.u, original_dtype)?,
        p: linalg_demote(client, result.p, original_dtype)?,
    })
}

fn polar_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<PolarDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle trivial case
    if n == 0 {
        return Ok(PolarDecomposition {
            u: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device)?,
            p: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device)?,
        });
    }

    // Compute SVD: A = U_svd @ S @ V^T
    let svd = client.svd_decompose(a)?;

    let u_svd: Vec<T> = svd.u.to_vec();
    let s_data: Vec<T> = svd.s.to_vec();
    let vt_data: Vec<T> = svd.vt.to_vec();

    // U = U_svd @ V^T
    let mut u_data: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += u_svd[i * n + k].to_f64() * vt_data[k * n + j].to_f64();
            }
            u_data[i * n + j] = T::from_f64(sum);
        }
    }

    // P = V @ S @ V^T
    // First compute V from V^T (transpose)
    let mut v_data: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            v_data[i * n + j] = vt_data[j * n + i];
        }
    }

    // P = V @ diag(S) @ V^T
    let mut p_data: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                // V[i,k] * S[k] * V[j,k] = V[i,k] * S[k] * V^T[k,j]
                sum += v_data[i * n + k].to_f64() * s_data[k].to_f64() * v_data[j * n + k].to_f64();
            }
            p_data[i * n + j] = T::from_f64(sum);
        }
    }

    Ok(PolarDecomposition {
        u: Tensor::<CpuRuntime>::from_slice(&u_data, &[n, n], device)?,
        p: Tensor::<CpuRuntime>::from_slice(&p_data, &[n, n], device)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn test_polar_orthogonal_input() {
        let client = create_client();
        let device = client.device();

        let angle = std::f64::consts::PI / 4.0;
        let c = angle.cos();
        let s = angle.sin();
        let a = Tensor::<CpuRuntime>::from_slice(&[c, -s, s, c], &[2, 2], device).unwrap();

        let result = client.polar_decompose(&a).unwrap();

        let u: Vec<f64> = result.u.to_vec();
        let p: Vec<f64> = result.p.to_vec();

        assert!(is_orthogonal(&u, 2, 1e-6), "U is not orthogonal");

        assert_close(p[0], 1.0, 1e-6);
        assert_close(p[1], 0.0, 1e-6);
        assert_close(p[2], 0.0, 1e-6);
        assert_close(p[3], 1.0, 1e-6);
    }

    #[test]
    fn test_polar_symmetric_input() {
        let client = create_client();
        let device = client.device();

        let a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();

        let result = client.polar_decompose(&a).unwrap();

        let u: Vec<f64> = result.u.to_vec();
        let p: Vec<f64> = result.p.to_vec();

        assert!(is_orthogonal(&u, 2, 1e-6), "U is not orthogonal");
        assert_close(p[1], p[2], 1e-6);
    }

    #[test]
    fn test_polar_reconstruction() {
        let client = create_client();
        let device = client.device();

        let a_data = [1.0f64, 2.0, 3.0, 4.0];
        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], device).unwrap();

        let result = client.polar_decompose(&a).unwrap();

        let u: Vec<f64> = result.u.to_vec();
        let p: Vec<f64> = result.p.to_vec();

        let reconstructed = matrix_multiply(&u, &p, 2);

        for i in 0..4 {
            assert_close(reconstructed[i], a_data[i], 1e-6);
        }
    }

    #[test]
    fn test_polar_p_symmetric() {
        let client = create_client();
        let device = client.device();

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let result = client.polar_decompose(&a).unwrap();
        let p: Vec<f64> = result.p.to_vec();

        assert_close(p[1], p[2], 1e-6);
    }

    #[test]
    fn test_polar_empty_matrix() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0, 0], device).unwrap();

        let result = client.polar_decompose(&a).unwrap();
        assert_eq!(result.u.shape(), &[0, 0]);
        assert_eq!(result.p.shape(), &[0, 0]);
    }
}
