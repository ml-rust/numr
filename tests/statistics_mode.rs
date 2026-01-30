//! Tests for mode operations

mod statistics_common;

use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use statistics_common::{approx_eq, approx_eq_f64};

// ============================================================================
// Mode Tests
// ============================================================================

#[test]
fn test_mode_simple() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Simple case: 2 appears 3 times
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 2.0, 3.0], &[5], &device);
    let (values, counts) = client.mode(&a, Some(0), false).unwrap();

    let values_data: Vec<f32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(
        approx_eq(values_data[0], 2.0, 1e-5),
        "Expected mode 2.0, got {}",
        values_data[0]
    );
    assert_eq!(
        counts_data[0], 3,
        "Expected count 3, got {}",
        counts_data[0]
    );
}

#[test]
fn test_mode_all_unique() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // All unique: return smallest with count 1
    let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 3.0, 1.0, 4.0, 2.0], &[5], &device);
    let (values, counts) = client.mode(&a, Some(0), false).unwrap();

    let values_data: Vec<f32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(
        approx_eq(values_data[0], 1.0, 1e-5),
        "Expected mode 1.0 (smallest), got {}",
        values_data[0]
    );
    assert_eq!(
        counts_data[0], 1,
        "Expected count 1, got {}",
        counts_data[0]
    );
}

#[test]
fn test_mode_tie() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Tie: 1 and 3 both appear twice, return smallest (1)
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 1.0, 3.0, 2.0], &[5], &device);
    let (values, counts) = client.mode(&a, Some(0), false).unwrap();

    let values_data: Vec<f32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(
        approx_eq(values_data[0], 1.0, 1e-5),
        "Expected mode 1.0 (smallest tied), got {}",
        values_data[0]
    );
    assert_eq!(
        counts_data[0], 2,
        "Expected count 2, got {}",
        counts_data[0]
    );
}

#[test]
fn test_mode_all_same() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // All same value
    let a = Tensor::<CpuRuntime>::from_slice(&[7.0f32, 7.0, 7.0, 7.0], &[4], &device);
    let (values, counts) = client.mode(&a, Some(0), false).unwrap();

    let values_data: Vec<f32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(
        approx_eq(values_data[0], 7.0, 1e-5),
        "Expected mode 7.0, got {}",
        values_data[0]
    );
    assert_eq!(
        counts_data[0], 4,
        "Expected count 4, got {}",
        counts_data[0]
    );
}

#[test]
fn test_mode_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[42.0f32], &[1], &device);
    let (values, counts) = client.mode(&a, Some(0), false).unwrap();

    let values_data: Vec<f32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(approx_eq(values_data[0], 42.0, 1e-5));
    assert_eq!(counts_data[0], 1);
}

#[test]
fn test_mode_2d_along_axis() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2x4 matrix, find mode along axis 1
    // Row 0: [1, 1, 2, 3] -> mode=1, count=2
    // Row 1: [4, 4, 4, 5] -> mode=4, count=3
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0],
        &[2, 4],
        &device,
    );
    let (values, counts) = client.mode(&a, Some(1), false).unwrap();

    assert_eq!(values.shape(), &[2]);
    assert_eq!(counts.shape(), &[2]);

    let values_data: Vec<f32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(approx_eq(values_data[0], 1.0, 1e-5), "Row 0 mode");
    assert!(approx_eq(values_data[1], 4.0, 1e-5), "Row 1 mode");
    assert_eq!(counts_data[0], 2, "Row 0 count");
    assert_eq!(counts_data[1], 3, "Row 1 count");
}

#[test]
fn test_mode_keepdim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0], &[2, 2], &device);
    let (values, counts) = client.mode(&a, Some(1), true).unwrap();

    // With keepdim=true, shape should be [2, 1]
    assert_eq!(values.shape(), &[2, 1]);
    assert_eq!(counts.shape(), &[2, 1]);
}

#[test]
fn test_mode_flatten() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // With dim=None, flatten first
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0, 2.0, 1.0], &[2, 3], &device);
    let (values, counts) = client.mode(&a, None, false).unwrap();

    // Should be scalar output (empty shape)
    assert_eq!(values.shape(), &[] as &[usize]);

    let values_data: Vec<f32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(
        approx_eq(values_data[0], 2.0, 1e-5),
        "Global mode should be 2.0"
    );
    assert_eq!(counts_data[0], 3, "Global mode count should be 3");
}

#[test]
fn test_mode_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 2.0, 3.0], &[4], &device);
    let (values, counts) = client.mode(&a, Some(0), false).unwrap();

    let values_data: Vec<f64> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert!(approx_eq_f64(values_data[0], 2.0, 1e-10));
    assert_eq!(counts_data[0], 2);
}

#[test]
fn test_mode_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 2, 3, 2], &[5], &device);
    let (values, counts) = client.mode(&a, Some(0), false).unwrap();

    let values_data: Vec<i32> = values.to_vec();
    let counts_data: Vec<i64> = counts.to_vec();

    assert_eq!(values_data[0], 2);
    assert_eq!(counts_data[0], 3);
}
