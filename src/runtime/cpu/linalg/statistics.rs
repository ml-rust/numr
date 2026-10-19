//! Statistical matrix operations (pseudo-inverse, condition number, covariance, correlation)

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::svd::svd_decompose_impl;
use crate::algorithm::linalg::impl_generic::{resolve_rcond, svd_cutoff};
use crate::algorithm::linalg::{linalg_demote, linalg_promote, validate_matrix_2d};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, MatmulOps};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Moore-Penrose pseudo-inverse via SVD: A^+ = V @ diag(1/S) @ U^T
pub fn pinverse_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    rcond: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "pinverse",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let rcond = resolve_rcond(rcond, m, n, a.dtype())?;

    if m == 0 || n == 0 {
        let empty = Tensor::<CpuRuntime>::zeros(&[n, m], a.dtype(), client.device())?;
        return linalg_demote(client, empty, original_dtype);
    }

    // A = U @ diag(S) @ V^T, with U [m, k], S [k] and V^T [k, n]
    let svd = svd_decompose_impl(client, &a)?;

    // Singular values at or below rcond * max(S) count as zero. The cutoff
    // runs as tensor ops, the same path the GPU backends take.
    let (s_inv, _rank) = svd_cutoff(client, &svd.s, rcond)?;

    // A^+ = (V * S^-1) @ U^T. The [k] row scales column l of V by S^-1[l].
    let v = svd.vt.transpose(0, 1)?.contiguous()?;
    let ut = svd.u.transpose(0, 1)?.contiguous()?;
    let v_sinv = BinaryOps::mul(client, &v, &s_inv)?;
    let pinv = MatmulOps::matmul(client, &v_sinv, &ut)?;
    linalg_demote(client, pinv, original_dtype)
}

/// Condition number via SVD: cond(A) = σ_max / σ_min
pub fn cond_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "cond",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => cond_typed::<f32>(client, &a, m, n),
        DType::F64 => cond_typed::<f64>(client, &a, m, n),
        dtype => Err(Error::UnsupportedDType { dtype, op: "cond" }),
    }?;
    linalg_demote(client, result, original_dtype)
}

fn cond_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle empty matrix
    if m == 0 || n == 0 {
        // Return infinity for empty matrix (undefined condition number)
        return Tensor::<CpuRuntime>::from_slice(&[T::from_f64(f64::INFINITY)], &[], device);
    }

    // Compute SVD to get singular values
    let svd = svd_decompose_impl(client, a)?;
    let s_data: Vec<T> = svd.s.to_vec();

    if s_data.is_empty() {
        return Tensor::<CpuRuntime>::from_slice(&[T::from_f64(f64::INFINITY)], &[], device);
    }

    // Find max and min singular values (s is sorted descending)
    let s_max = s_data[0].to_f64();
    let s_min = s_data[s_data.len() - 1].to_f64();

    // Condition number = s_max / s_min
    // If s_min is essentially zero, return infinity
    let eps = T::epsilon_val();
    let cond = if s_min.abs() < eps {
        T::from_f64(f64::INFINITY)
    } else {
        T::from_f64(s_max / s_min)
    };

    Tensor::<CpuRuntime>::from_slice(&[cond], &[], device)
}

/// Covariance matrix
/// cov(X) = (X - mean(X))^T @ (X - mean(X)) / (n_samples - ddof)
pub fn cov_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    ddof: Option<usize>,
) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "cov",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (n_samples, n_features) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => cov_typed::<f32>(client, &a, n_samples, n_features, ddof),
        DType::F64 => cov_typed::<f64>(client, &a, n_samples, n_features, ddof),
        dtype => Err(Error::UnsupportedDType { dtype, op: "cov" }),
    }?;
    linalg_demote(client, result, original_dtype)
}

fn cov_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n_samples: usize,
    n_features: usize,
    ddof: Option<usize>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let ddof = ddof.unwrap_or(1); // Default: unbiased (sample) covariance

    // Check we have enough samples
    if n_samples <= ddof {
        return Err(Error::Internal(format!(
            "cov: n_samples ({}) must be greater than ddof ({})",
            n_samples, ddof
        )));
    }

    // Handle edge case: single feature
    if n_features == 0 {
        return Tensor::<CpuRuntime>::from_slice::<T>(&[], &[0, 0], device);
    }

    let a_data: Vec<T> = a.to_vec();
    let n_f64 = n_samples as f64;

    // Step 1: Compute mean for each feature (column)
    let mut means: Vec<T> = vec![T::zero(); n_features];
    for j in 0..n_features {
        let mut sum = T::zero();
        for i in 0..n_samples {
            sum = sum + a_data[i * n_features + j];
        }
        means[j] = T::from_f64(sum.to_f64() / n_f64);
    }

    // Step 2: Compute centered data and covariance matrix
    // cov[i,j] = sum_k (X[k,i] - mean[i]) * (X[k,j] - mean[j]) / (n - ddof)
    let divisor = (n_samples - ddof) as f64;
    let mut cov: Vec<T> = vec![T::zero(); n_features * n_features];

    for i in 0..n_features {
        for j in i..n_features {
            // Exploit symmetry: only compute upper triangle
            let mut sum = T::zero();
            for k in 0..n_samples {
                let xi = a_data[k * n_features + i] - means[i];
                let xj = a_data[k * n_features + j] - means[j];
                sum = sum + xi * xj;
            }
            let cov_val = T::from_f64(sum.to_f64() / divisor);
            cov[i * n_features + j] = cov_val;
            cov[j * n_features + i] = cov_val; // Symmetry
        }
    }

    Tensor::<CpuRuntime>::from_slice(&cov, &[n_features, n_features], device)
}

/// Correlation coefficient matrix
/// corr[i,j] = cov[i,j] / (std[i] * std[j])
pub fn corrcoef_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "corrcoef",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (n_samples, n_features) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => corrcoef_typed::<f32>(client, &a, n_samples, n_features),
        DType::F64 => corrcoef_typed::<f64>(client, &a, n_samples, n_features),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "corrcoef",
        }),
    }?;
    linalg_demote(client, result, original_dtype)
}

fn corrcoef_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n_samples: usize,
    n_features: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Check we have enough samples (need at least 2 for correlation)
    if n_samples <= 1 {
        return Err(Error::Internal(format!(
            "corrcoef: n_samples ({}) must be greater than 1",
            n_samples
        )));
    }

    // Handle edge case: single feature
    if n_features == 0 {
        return Tensor::<CpuRuntime>::from_slice::<T>(&[], &[0, 0], device);
    }

    // Compute covariance matrix (with ddof=1 for sample covariance)
    let cov_tensor = cov_typed::<T>(client, a, n_samples, n_features, Some(1))?;
    let cov_data: Vec<T> = cov_tensor.to_vec();

    // Extract standard deviations (sqrt of diagonal)
    let mut stds: Vec<f64> = vec![0.0; n_features];
    for i in 0..n_features {
        let var = cov_data[i * n_features + i].to_f64();
        stds[i] = if var > 0.0 { var.sqrt() } else { 0.0 };
    }

    // Compute correlation matrix
    let mut corr: Vec<T> = vec![T::zero(); n_features * n_features];

    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                // Diagonal is always 1.0 (unless std is 0)
                corr[i * n_features + j] = if stds[i] > 0.0 { T::one() } else { T::zero() };
            } else {
                // Off-diagonal: cov[i,j] / (std[i] * std[j])
                let std_prod = stds[i] * stds[j];
                let corr_val = if std_prod > 0.0 {
                    T::from_f64(cov_data[i * n_features + j].to_f64() / std_prod)
                } else {
                    T::zero() // Undefined correlation set to 0
                };
                corr[i * n_features + j] = corr_val;
            }
        }
    }

    Tensor::<CpuRuntime>::from_slice(&corr, &[n_features, n_features], device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuDevice;

    fn pinv_f64(data: &[f64], shape: &[usize], rcond: Option<f64>) -> (Vec<usize>, Vec<f64>) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let a = Tensor::<CpuRuntime>::from_slice(data, shape, &device).unwrap();
        let p = pinverse_impl(&client, &a, rcond).unwrap();
        (p.shape().to_vec(), p.to_vec::<f64>())
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "element {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn full_rank_square_matches_inverse() {
        // [[4, 7], [2, 6]]^-1 = [[0.6, -0.7], [-0.2, 0.4]]
        let (shape, p) = pinv_f64(&[4.0, 7.0, 2.0, 6.0], &[2, 2], None);
        assert_eq!(shape, vec![2, 2]);
        assert_close(&p, &[0.6, -0.7, -0.2, 0.4], 1e-12);
    }

    #[test]
    fn rank_deficient_drops_the_null_direction() {
        // A = u v^T with u = [1, 2], v = [1, 2]. A^+ = A^T / ||A||_F^2 = A^T / 25.
        let (shape, p) = pinv_f64(&[1.0, 2.0, 2.0, 4.0], &[2, 2], None);
        assert_eq!(shape, vec![2, 2]);
        assert_close(&p, &[0.04, 0.08, 0.08, 0.16], 1e-12);
    }

    #[test]
    fn rcond_sets_the_cutoff_relative_to_max_singular_value() {
        // Singular values 1 and 1e-8 of a [3, 2] matrix.
        let a = [1.0, 0.0, 0.0, 1e-8, 0.0, 0.0];

        // The default cutoff keeps 1e-8, so it is inverted.
        let (shape, p) = pinv_f64(&a, &[3, 2], None);
        assert_eq!(shape, vec![2, 3]);
        assert_close(&p, &[1.0, 0.0, 0.0, 0.0, 1e8, 0.0], 1e-4);

        // rcond = 1e-4 puts the cutoff at 1e-4 * 1, so 1e-8 counts as zero.
        let (_, p) = pinv_f64(&a, &[3, 2], Some(1e-4));
        assert_close(&p, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-12);

        // Scaling A scales the cutoff with it: the result is A^+ / 1000.
        let scaled: Vec<f64> = a.iter().map(|v| v * 1000.0).collect();
        let (_, p) = pinv_f64(&scaled, &[3, 2], Some(1e-4));
        assert_close(&p, &[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-15);
    }

    #[test]
    fn all_zero_matrix_gives_all_zero_pinverse() {
        let (shape, p) = pinv_f64(&[0.0; 6], &[2, 3], None);
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(p, vec![0.0; 6]);
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn f32_rank_deficient_and_zero() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], &device).unwrap();
        let p = pinverse_impl(&client, &a, None).unwrap();
        assert_eq!(p.dtype(), DType::F32);
        let want = [0.04f32, 0.08, 0.08, 0.16];
        for (g, w) in p.to_vec::<f32>().iter().zip(want) {
            assert!((g - w).abs() <= 1e-5, "got {g}, want {w}");
        }

        let z = Tensor::<CpuRuntime>::zeros(&[3, 2], DType::F32, &device).unwrap();
        let p = pinverse_impl(&client, &z, None).unwrap();
        assert_eq!(p.shape(), &[2, 3]);
        assert_eq!(p.to_vec::<f32>(), vec![0.0f32; 6]);
    }

    #[test]
    fn empty_matrix_gives_transposed_empty_shape() {
        let (shape, p) = pinv_f64(&[], &[0, 3], None);
        assert_eq!(shape, vec![3, 0]);
        assert!(p.is_empty());
    }

    #[test]
    fn batched_input_is_rejected() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let a = Tensor::<CpuRuntime>::zeros(&[2, 2, 2], DType::F64, &device).unwrap();
        assert!(pinverse_impl(&client, &a, None).is_err());
    }
}
