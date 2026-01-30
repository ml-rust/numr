//! Tests for histogram operations

mod statistics_common;

use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use statistics_common::approx_eq;

// ============================================================================
// Histogram Tests
// ============================================================================

#[test]
fn test_histogram_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Simple data: 5 values in range [0, 5)
    let a = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.5, 2.5, 3.5, 4.5], &[5], &device);
    let (hist, edges) = client.histogram(&a, 5, Some((0.0, 5.0))).unwrap();

    assert_eq!(hist.shape(), &[5]);
    assert_eq!(edges.shape(), &[6]);

    let hist_data: Vec<i64> = hist.to_vec();
    // Each value falls into exactly one bin
    assert_eq!(
        hist_data,
        vec![1, 1, 1, 1, 1],
        "Each bin should have 1 count"
    );

    let edges_data: Vec<f32> = edges.to_vec();
    assert!(approx_eq(edges_data[0], 0.0, 1e-5));
    assert!(approx_eq(edges_data[5], 5.0, 1e-5));
}

#[test]
fn test_histogram_auto_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let (hist, _edges) = client.histogram(&a, 4, None).unwrap();

    assert_eq!(hist.shape(), &[4]);
    let hist_data: Vec<i64> = hist.to_vec();
    let total: i64 = hist_data.iter().sum();
    assert_eq!(total, 5, "Total count should equal number of elements");
}

#[test]
fn test_histogram_concentrated() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // All values in one bin
    let a = Tensor::<CpuRuntime>::from_slice(&[2.1f32, 2.2, 2.3, 2.4, 2.5], &[5], &device);
    let (hist, _edges) = client.histogram(&a, 5, Some((0.0, 5.0))).unwrap();

    let hist_data: Vec<i64> = hist.to_vec();
    // Values 2.1-2.5 all fall in bin index 2 (range [2.0, 3.0))
    assert_eq!(hist_data[2], 5, "All values should be in bin 2");
    assert_eq!(hist_data[0], 0);
    assert_eq!(hist_data[1], 0);
    assert_eq!(hist_data[3], 0);
    assert_eq!(hist_data[4], 0);
}

#[test]
fn test_histogram_zero_bins() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let result = client.histogram(&a, 0, None);
    assert!(result.is_err());
}

#[test]
fn test_histogram_invalid_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    // min >= max should fail
    let result = client.histogram(&a, 5, Some((5.0, 5.0)));
    assert!(result.is_err());

    let result = client.histogram(&a, 5, Some((10.0, 5.0)));
    assert!(result.is_err());
}

#[test]
fn test_histogram_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 1.5, 2.5], &[3], &device);
    let (hist, _edges) = client.histogram(&a, 3, Some((0.0, 3.0))).unwrap();
    let hist_data: Vec<i64> = hist.to_vec();
    assert_eq!(hist_data, vec![1, 1, 1]);
}
