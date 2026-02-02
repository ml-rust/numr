//! Tests for covariance and correlation coefficient operations

mod statistics_common;

use numr::ops::StatisticalOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use statistics_common::approx_eq;

// ============================================================================
// Covariance Tests
// ============================================================================

#[test]
fn test_cov_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 3 samples, 2 features
    // Feature 0: [1, 2, 3], Feature 1: [4, 5, 6]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2], &device);
    let cov = client.cov(&a, None).unwrap();

    assert_eq!(cov.shape(), &[2, 2]);
    let data: Vec<f32> = cov.to_vec();

    // With default ddof=1 (sample covariance):
    // Var(X) = 1.0, Var(Y) = 1.0, Cov(X,Y) = 1.0
    // Cov matrix = [[1, 1], [1, 1]]
    assert!(
        approx_eq(data[0], 1.0, 1e-5),
        "cov[0,0]: expected 1.0, got {}",
        data[0]
    );
    assert!(
        approx_eq(data[3], 1.0, 1e-5),
        "cov[1,1]: expected 1.0, got {}",
        data[3]
    );
    assert!(
        approx_eq(data[1], 1.0, 1e-5),
        "cov[0,1]: expected 1.0, got {}",
        data[1]
    );
    assert!(
        approx_eq(data[2], 1.0, 1e-5),
        "cov[1,0]: expected 1.0, got {}",
        data[2]
    );
}

#[test]
fn test_cov_symmetry() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &device,
    );
    let cov = client.cov(&a, None).unwrap();
    let data: Vec<f32> = cov.to_vec();

    // Covariance matrix should be symmetric
    assert!(
        approx_eq(data[1], data[3], 1e-5),
        "cov[0,1] should equal cov[1,0]"
    );
    assert!(
        approx_eq(data[2], data[6], 1e-5),
        "cov[0,2] should equal cov[2,0]"
    );
    assert!(
        approx_eq(data[5], data[7], 1e-5),
        "cov[1,2] should equal cov[2,1]"
    );
}

// ============================================================================
// Correlation Coefficient Tests
// ============================================================================

#[test]
fn test_corrcoef_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Perfectly correlated features
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2], &device);
    let corr = client.corrcoef(&a).unwrap();

    assert_eq!(corr.shape(), &[2, 2]);
    let data: Vec<f32> = corr.to_vec();

    // Diagonal should be 1.0
    assert!(
        approx_eq(data[0], 1.0, 1e-4),
        "corr[0,0]: expected 1.0, got {}",
        data[0]
    );
    assert!(
        approx_eq(data[3], 1.0, 1e-4),
        "corr[1,1]: expected 1.0, got {}",
        data[3]
    );

    // Perfect positive correlation
    assert!(
        approx_eq(data[1], 1.0, 1e-4),
        "corr[0,1]: expected 1.0, got {}",
        data[1]
    );
}

#[test]
fn test_corrcoef_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Random-ish data
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 5.0, 2.0, 3.0, 4.0, 1.0, 5.0, 2.0, 3.0, 4.0, 6.0, 7.0,
        ],
        &[4, 3],
        &device,
    );
    let corr = client.corrcoef(&a).unwrap();
    let data: Vec<f32> = corr.to_vec();

    // All correlation coefficients should be in [-1, 1]
    for (i, &val) in data.iter().enumerate() {
        assert!(
            (-1.0 - 1e-5..=1.0 + 1e-5).contains(&val),
            "corr[{}] = {} out of range [-1, 1]",
            i,
            val
        );
    }
}
