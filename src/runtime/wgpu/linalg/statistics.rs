//! Covariance and correlation coefficient computations.

use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, LinalgOps, MatmulOps, ReduceOps, UnaryOps, UtilityOps};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

pub fn cov(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    ddof: Option<usize>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (n_samples, _n_features) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();
    let ddof_val = ddof.unwrap_or(1);

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "cov" });
    }

    // Need at least ddof + 1 samples
    if n_samples <= ddof_val {
        return Err(Error::Internal(format!(
            "cov: need at least {} samples for ddof={}, got {}",
            ddof_val + 1,
            ddof_val,
            n_samples
        )));
    }

    // Compute mean along axis 0
    let sum = client.sum(a, &[0], true)?;
    let n_samples_tensor = Tensor::<WgpuRuntime>::from_slice(&[n_samples as f32], &[], device);
    let mean = client.div(&sum, &n_samples_tensor)?;

    // Center the data
    let centered = client.sub(a, &mean)?;

    // Compute covariance: C = X_centered^T @ X_centered / (n - ddof)
    let centered_t = centered.transpose(0, 1)?.contiguous();
    let centered_contig = centered.contiguous();
    let cov_unnorm = client.matmul(&centered_t, &centered_contig)?;

    // Normalize by (n - ddof)
    let divisor = (n_samples - ddof_val) as f32;
    let divisor_tensor = Tensor::<WgpuRuntime>::from_slice(&[divisor], &[], device);
    let cov_mat = client.div(&cov_unnorm, &divisor_tensor)?;

    Ok(cov_mat)
}

pub fn corrcoef(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (n_samples, n_features) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "corrcoef",
        });
    }

    // Need at least 2 samples
    if n_samples < 2 {
        return Err(Error::Internal(format!(
            "corrcoef: need at least 2 samples, got {}",
            n_samples
        )));
    }

    // Handle edge case: no features
    if n_features == 0 {
        return Ok(Tensor::<WgpuRuntime>::from_slice::<f32>(
            &[],
            &[0, 0],
            device,
        ));
    }

    // Compute covariance matrix
    let cov_mat = LinearAlgebraAlgorithms::cov(client, a, Some(1))?;

    // Extract diagonal (variances) and compute standard deviations
    let variances = LinalgOps::diag(client, &cov_mat)?;
    let std_devs = client.sqrt(&variances)?;

    // Pull data to CPU for zero-variance detection
    let std_vec: Vec<f64> = std_devs
        .to_vec::<f32>()
        .into_iter()
        .map(|x| x as f64)
        .collect();
    let cov_vec: Vec<f64> = cov_mat
        .to_vec::<f32>()
        .into_iter()
        .map(|x| x as f64)
        .collect();

    // Build correlation matrix with proper zero-variance handling
    let mut corr_data: Vec<f32> = vec![0.0; n_features * n_features];
    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                // Diagonal: 1.0 if std > 0, else 0.0
                corr_data[i * n_features + j] = if std_vec[i] > 0.0 { 1.0 } else { 0.0 };
            } else {
                // Off-diagonal: correlation if both stds > 0, else 0.0
                let std_prod = std_vec[i] * std_vec[j];
                corr_data[i * n_features + j] = if std_prod > 0.0 {
                    ((cov_vec[i * n_features + j] / std_prod).clamp(-1.0, 1.0)) as f32
                } else {
                    0.0
                };
            }
        }
    }

    Ok(Tensor::<WgpuRuntime>::from_slice(
        &corr_data,
        &[n_features, n_features],
        device,
    ))
}
