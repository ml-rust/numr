//! Schur decomposition using QR iteration with Wilkinson shift

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    SchurDecomposition, linalg_demote, linalg_promote, validate_linalg_dtype,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Schur decomposition using QR iteration
///
/// Computes A = Z @ T @ Z^T where Z is orthogonal and T is upper quasi-triangular.
pub fn schur_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<SchurDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => schur_decompose_typed::<f32>(client, &a, n),
        DType::F64 => schur_decompose_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    Ok(SchurDecomposition {
        z: linalg_demote(client, result.z, original_dtype)?,
        t: linalg_demote(client, result.t, original_dtype)?,
    })
}

fn schur_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<SchurDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(SchurDecomposition {
            z: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device)?,
            t: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device)?,
        });
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        return Ok(SchurDecomposition {
            z: Tensor::<CpuRuntime>::from_slice(&[T::one()], &[1, 1], device)?,
            t: Tensor::<CpuRuntime>::from_slice(&data, &[1, 1], device)?,
        });
    }

    // Copy input to working matrix T (will become the Schur form)
    let mut t_data: Vec<T> = a.to_vec();

    // Initialize Z as identity matrix
    let mut z_data: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        z_data[i * n + i] = T::one();
    }

    // Step 1: Reduce to upper Hessenberg form using Householder reflections
    // H = Q0^T @ A @ Q0, accumulate Q0 into Z
    hessenberg_reduction::<T>(&mut t_data, &mut z_data, n);

    // Step 2: QR iteration to get upper quasi-triangular form
    let max_iter = 30 * n;
    let eps = T::epsilon_val();

    for _iter in 0..max_iter {
        // Check convergence: subdiagonal elements should be small
        let mut converged = true;
        for i in 0..(n - 1) {
            let h_ii = t_data[i * n + i].to_f64().abs();
            let h_ip1 = t_data[(i + 1) * n + (i + 1)].to_f64().abs();
            let threshold = eps * (h_ii + h_ip1).max(1.0);

            let subdiag = t_data[(i + 1) * n + i].to_f64().abs();
            if subdiag > threshold {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }

        // Perform one QR iteration with implicit shift
        qr_iteration_step::<T>(&mut t_data, &mut z_data, n);
    }

    // Clean up small subdiagonal elements to make T exactly quasi-triangular
    for i in 0..(n - 1) {
        let h_ii = t_data[i * n + i].to_f64().abs();
        let h_ip1 = t_data[(i + 1) * n + (i + 1)].to_f64().abs();
        let threshold = eps * (h_ii + h_ip1).max(1.0);

        let subdiag = t_data[(i + 1) * n + i].to_f64();
        if subdiag.abs() <= threshold {
            t_data[(i + 1) * n + i] = T::zero();
        }
    }

    // Clear lower triangular part (except 2x2 blocks for complex eigenvalues)
    for i in 2..n {
        for j in 0..(i - 1) {
            t_data[i * n + j] = T::zero();
        }
    }

    Ok(SchurDecomposition {
        z: Tensor::<CpuRuntime>::from_slice(&z_data, &[n, n], device)?,
        t: Tensor::<CpuRuntime>::from_slice(&t_data, &[n, n], device)?,
    })
}

/// Reduce matrix to upper Hessenberg form using Householder reflections.
/// Modifies H in-place and accumulates transformations into Q.
pub fn hessenberg_reduction<T: Element + LinalgElement>(h: &mut [T], q: &mut [T], n: usize) {
    for k in 0..(n - 2) {
        // Compute Householder vector for column k, rows k+1 to n-1
        let mut v = vec![T::zero(); n - k - 1];
        let mut norm_sq = 0.0;

        for i in (k + 1)..n {
            let val = h[i * n + k].to_f64();
            v[i - k - 1] = T::from_f64(val);
            norm_sq += val * val;
        }

        if norm_sq < T::epsilon_val() {
            continue;
        }

        let norm = norm_sq.sqrt();
        let x0 = v[0].to_f64();
        let alpha = if x0 >= 0.0 { -norm } else { norm };

        v[0] = T::from_f64(x0 - alpha);

        // Normalize v
        let mut v_norm_sq = 0.0;
        for vi in &v {
            v_norm_sq += vi.to_f64() * vi.to_f64();
        }
        if v_norm_sq < T::epsilon_val() {
            continue;
        }
        let v_norm = v_norm_sq.sqrt();
        for vi in &mut v {
            *vi = T::from_f64(vi.to_f64() / v_norm);
        }

        // Apply Householder: H = H - 2 * v * (v^T * H) for rows k+1..n
        // Left multiplication: H[k+1:n, :] = H[k+1:n, :] - 2 * v * (v^T @ H[k+1:n, :])
        for j in 0..n {
            let mut dot = 0.0;
            for i in 0..v.len() {
                dot += v[i].to_f64() * h[(k + 1 + i) * n + j].to_f64();
            }
            for i in 0..v.len() {
                let old_val = h[(k + 1 + i) * n + j].to_f64();
                h[(k + 1 + i) * n + j] = T::from_f64(old_val - 2.0 * v[i].to_f64() * dot);
            }
        }

        // Right multiplication: H[:, k+1:n] = H[:, k+1:n] - 2 * (H[:, k+1:n] @ v) * v^T
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..v.len() {
                dot += h[i * n + (k + 1 + j)].to_f64() * v[j].to_f64();
            }
            for j in 0..v.len() {
                let old_val = h[i * n + (k + 1 + j)].to_f64();
                h[i * n + (k + 1 + j)] = T::from_f64(old_val - 2.0 * dot * v[j].to_f64());
            }
        }

        // Accumulate into Q: Q = Q @ H_k (where H_k is the Householder reflection)
        // Q[:, k+1:n] = Q[:, k+1:n] - 2 * (Q[:, k+1:n] @ v) * v^T
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..v.len() {
                dot += q[i * n + (k + 1 + j)].to_f64() * v[j].to_f64();
            }
            for j in 0..v.len() {
                let old_val = q[i * n + (k + 1 + j)].to_f64();
                q[i * n + (k + 1 + j)] = T::from_f64(old_val - 2.0 * dot * v[j].to_f64());
            }
        }
    }
}

/// Perform one QR iteration step with implicit Wilkinson shift.
pub fn qr_iteration_step<T: Element + LinalgElement>(h: &mut [T], q: &mut [T], n: usize) {
    // Compute Wilkinson shift from bottom 2x2 block
    let a = h[(n - 2) * n + (n - 2)].to_f64();
    let b = h[(n - 2) * n + (n - 1)].to_f64();
    let c = h[(n - 1) * n + (n - 2)].to_f64();
    let d = h[(n - 1) * n + (n - 1)].to_f64();

    // Wilkinson shift: eigenvalue of 2x2 block closest to d
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    let shift = if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        if (lambda1 - d).abs() < (lambda2 - d).abs() {
            lambda1
        } else {
            lambda2
        }
    } else {
        // Complex eigenvalues, use trace/2 as shift
        trace / 2.0
    };

    // Apply shift: H = H - shift * I
    for i in 0..n {
        h[i * n + i] = T::from_f64(h[i * n + i].to_f64() - shift);
    }

    // QR factorization using Givens rotations
    // Process each subdiagonal element
    for i in 0..(n - 1) {
        let a_val = h[i * n + i].to_f64();
        let b_val = h[(i + 1) * n + i].to_f64();

        if b_val.abs() < T::epsilon_val() {
            continue;
        }

        // Compute Givens rotation to zero out b_val
        let r = (a_val * a_val + b_val * b_val).sqrt();
        let c = a_val / r;
        let s = -b_val / r;

        // Apply rotation from the left: rows i and i+1
        for j in 0..n {
            let t1 = h[i * n + j].to_f64();
            let t2 = h[(i + 1) * n + j].to_f64();
            h[i * n + j] = T::from_f64(c * t1 - s * t2);
            h[(i + 1) * n + j] = T::from_f64(s * t1 + c * t2);
        }

        // Apply rotation from the right: cols i and i+1
        for k in 0..n {
            let t1 = h[k * n + i].to_f64();
            let t2 = h[k * n + (i + 1)].to_f64();
            h[k * n + i] = T::from_f64(c * t1 - s * t2);
            h[k * n + (i + 1)] = T::from_f64(s * t1 + c * t2);
        }

        // Accumulate into Q
        for k in 0..n {
            let t1 = q[k * n + i].to_f64();
            let t2 = q[k * n + (i + 1)].to_f64();
            q[k * n + i] = T::from_f64(c * t1 - s * t2);
            q[k * n + (i + 1)] = T::from_f64(s * t1 + c * t2);
        }
    }

    // Remove shift: H = H + shift * I
    for i in 0..n {
        h[i * n + i] = T::from_f64(h[i * n + i].to_f64() + shift);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_schur_1x1() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(&[5.0f64], &[1, 1], device).unwrap();
        let schur = client.schur_decompose(&a).unwrap();

        let z_data: Vec<f64> = schur.z.to_vec();
        let t_data: Vec<f64> = schur.t.to_vec();

        // For 1x1, Z = [1] and T = A
        assert!((z_data[0] - 1.0).abs() < 1e-10);
        assert!((t_data[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_schur_2x2_reconstruction() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let schur = client.schur_decompose(&a).unwrap();

        let z_data: Vec<f64> = schur.z.to_vec();
        let t_data: Vec<f64> = schur.t.to_vec();

        // Verify Z is orthogonal: Z^T @ Z = I
        let ztza = z_data[0] * z_data[0] + z_data[2] * z_data[2];
        let ztzb = z_data[0] * z_data[1] + z_data[2] * z_data[3];
        let ztzd = z_data[1] * z_data[1] + z_data[3] * z_data[3];

        assert!((ztza - 1.0).abs() < 1e-6, "Z^T Z [0,0] should be 1");
        assert!(ztzb.abs() < 1e-6, "Z^T Z [0,1] should be 0");
        assert!((ztzd - 1.0).abs() < 1e-6, "Z^T Z [1,1] should be 1");

        // Verify reconstruction: A = Z @ T @ Z^T
        // Z @ T
        let zt00 = z_data[0] * t_data[0] + z_data[1] * t_data[2];
        let zt01 = z_data[0] * t_data[1] + z_data[1] * t_data[3];
        let zt10 = z_data[2] * t_data[0] + z_data[3] * t_data[2];
        let zt11 = z_data[2] * t_data[1] + z_data[3] * t_data[3];

        // (Z @ T) @ Z^T
        let rec00 = zt00 * z_data[0] + zt01 * z_data[1];
        let rec01 = zt00 * z_data[2] + zt01 * z_data[3];
        let rec10 = zt10 * z_data[0] + zt11 * z_data[1];
        let rec11 = zt10 * z_data[2] + zt11 * z_data[3];

        assert!(
            (rec00 - 1.0).abs() < 1e-5,
            "Reconstruction [0,0] failed: {} != 1.0",
            rec00
        );
        assert!(
            (rec01 - 2.0).abs() < 1e-5,
            "Reconstruction [0,1] failed: {} != 2.0",
            rec01
        );
        assert!(
            (rec10 - 3.0).abs() < 1e-5,
            "Reconstruction [1,0] failed: {} != 3.0",
            rec10
        );
        assert!(
            (rec11 - 4.0).abs() < 1e-5,
            "Reconstruction [1,1] failed: {} != 4.0",
            rec11
        );
    }

    #[test]
    fn test_schur_symmetric_diagonal() {
        let client = create_client();
        let device = client.device();

        // For a symmetric matrix, Schur form is diagonal (eigenvalue decomposition)
        // A = [[2, 1], [1, 3]] (symmetric)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 3.0], &[2, 2], device).unwrap();
        let schur = client.schur_decompose(&a).unwrap();

        let t_data: Vec<f64> = schur.t.to_vec();

        // T[1,0] should be small (quasi-triangular becomes diagonal for symmetric)
        assert!(
            t_data[2].abs() < 0.1, // Use looser tolerance for QR iteration
            "T should be quasi-triangular: T[1,0] = {}",
            t_data[2]
        );
    }

    #[test]
    fn test_schur_3x3() {
        let client = create_client();
        let device = client.device();

        // A 3x3 upper triangular matrix is already in Schur form
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0],
            &[3, 3],
            device,
        )
        .unwrap();
        let schur = client.schur_decompose(&a).unwrap();

        let z_data: Vec<f64> = schur.z.to_vec();
        let t_data: Vec<f64> = schur.t.to_vec();

        // Z should be close to identity
        let diag_sum = z_data[0] * z_data[0] + z_data[4] * z_data[4] + z_data[8] * z_data[8];
        assert!(
            diag_sum > 2.5,
            "For upper triangular input, Z should be close to identity"
        );

        // T should have small subdiagonal elements
        assert!(
            t_data[3].abs() < 0.1,
            "T[1,0] should be small: {}",
            t_data[3]
        );
        assert!(
            t_data[6].abs() < 0.1,
            "T[2,0] should be small: {}",
            t_data[6]
        );
        assert!(
            t_data[7].abs() < 0.1,
            "T[2,1] should be small: {}",
            t_data[7]
        );
    }
}
