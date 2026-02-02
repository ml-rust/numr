//! Tests for skewness and kurtosis (higher-order moments)

mod statistics_common;

use numr::ops::StatisticalOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Skewness Tests
// ============================================================================

#[test]
fn test_skew_symmetric() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Symmetric distribution: skewness should be ~0
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let skew = client.skew(&a, &[], false, 0).unwrap();
    let data: Vec<f32> = skew.to_vec();
    assert!(
        data[0].abs() < 0.1,
        "Symmetric data should have skew ~0, got {}",
        data[0]
    );
}

#[test]
fn test_skew_right_skewed() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Right-skewed distribution (tail toward positive)
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.5, 2.0, 2.5, 10.0], &[5], &device);
    let skew = client.skew(&a, &[], false, 0).unwrap();
    let data: Vec<f32> = skew.to_vec();
    assert!(
        data[0] > 0.0,
        "Right-skewed data should have positive skew, got {}",
        data[0]
    );
}

#[test]
fn test_skew_left_skewed() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Left-skewed distribution (tail toward negative)
    let a = Tensor::<CpuRuntime>::from_slice(&[-10.0f32, 7.5, 8.0, 8.5, 9.0], &[5], &device);
    let skew = client.skew(&a, &[], false, 0).unwrap();
    let data: Vec<f32> = skew.to_vec();
    assert!(
        data[0] < 0.0,
        "Left-skewed data should have negative skew, got {}",
        data[0]
    );
}

// ============================================================================
// Kurtosis Tests
// ============================================================================

#[test]
fn test_kurtosis_uniform() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Uniform distribution has negative excess kurtosis
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let kurt = client.kurtosis(&a, &[], false, 0).unwrap();
    let data: Vec<f32> = kurt.to_vec();
    // Uniform has kurtosis < 0 (platykurtic)
    assert!(
        data[0] < 0.0,
        "Uniform-like data should have negative kurtosis, got {}",
        data[0]
    );
}

#[test]
fn test_kurtosis_heavy_tails() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Heavy-tailed distribution (extreme values)
    let a = Tensor::<CpuRuntime>::from_slice(
        &[-100.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
        &[10],
        &device,
    );
    let kurt = client.kurtosis(&a, &[], false, 0).unwrap();
    let data: Vec<f32> = kurt.to_vec();
    // Should have positive excess kurtosis (leptokurtic)
    assert!(
        data[0] > 0.0,
        "Heavy-tailed data should have positive kurtosis, got {}",
        data[0]
    );
}

#[test]
fn test_skew_kurtosis_multidim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2x5 matrix, compute along dim=1
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        &[2, 5],
        &device,
    );

    let skew = client.skew(&a, &[1], false, 0).unwrap();
    assert_eq!(skew.shape(), &[2]);

    let kurt = client.kurtosis(&a, &[1], false, 0).unwrap();
    assert_eq!(kurt.shape(), &[2]);
}
