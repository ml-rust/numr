//! Tests for quantile, percentile, and median operations

mod statistics_common;

use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ComplexOps, ConditionalOps, CumulativeOps, IndexingOps,
    LinalgOps, LogicalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    StatisticalOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use statistics_common::{approx_eq, approx_eq_f64};

// ============================================================================
// Quantile Tests
// ============================================================================

#[test]
fn test_quantile_1d_median() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Odd number of elements - exact median
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 2.0, 5.0, 4.0], &[5], &device);
    let result = client.quantile(&a, 0.5, Some(0), false, "linear").unwrap();
    let data: Vec<f32> = result.to_vec();
    assert!(
        approx_eq(data[0], 3.0, 1e-5),
        "Expected 3.0, got {}",
        data[0]
    );
}

#[test]
fn test_quantile_1d_interpolation() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Even number of elements - requires interpolation
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let result = client.quantile(&a, 0.5, Some(0), false, "linear").unwrap();
    let data: Vec<f32> = result.to_vec();
    // q=0.5 with n=4: idx = 0.5 * 3 = 1.5, interpolate between 2.0 and 3.0
    assert!(
        approx_eq(data[0], 2.5, 1e-5),
        "Expected 2.5, got {}",
        data[0]
    );
}

#[test]
fn test_quantile_interpolation_methods() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

    // q=0.25: idx = 0.25 * 3 = 0.75
    // linear: 1 * 0.25 + 2 * 0.75 = 1.75
    let linear = client.quantile(&a, 0.25, Some(0), false, "linear").unwrap();
    let data: Vec<f32> = linear.to_vec();
    assert!(
        approx_eq(data[0], 1.75, 1e-5),
        "Linear: expected 1.75, got {}",
        data[0]
    );

    // lower: floor(0.75) = 0 -> value at index 0 = 1.0
    let lower = client.quantile(&a, 0.25, Some(0), false, "lower").unwrap();
    let data: Vec<f32> = lower.to_vec();
    assert!(
        approx_eq(data[0], 1.0, 1e-5),
        "Lower: expected 1.0, got {}",
        data[0]
    );

    // higher: ceil(0.75) = 1 -> value at index 1 = 2.0
    let higher = client.quantile(&a, 0.25, Some(0), false, "higher").unwrap();
    let data: Vec<f32> = higher.to_vec();
    assert!(
        approx_eq(data[0], 2.0, 1e-5),
        "Higher: expected 2.0, got {}",
        data[0]
    );

    // nearest: round(0.75) = 1 -> value at index 1 = 2.0
    let nearest = client
        .quantile(&a, 0.25, Some(0), false, "nearest")
        .unwrap();
    let data: Vec<f32> = nearest.to_vec();
    assert!(
        approx_eq(data[0], 2.0, 1e-5),
        "Nearest: expected 2.0, got {}",
        data[0]
    );

    // midpoint: (1.0 + 2.0) / 2 = 1.5
    let midpoint = client
        .quantile(&a, 0.25, Some(0), false, "midpoint")
        .unwrap();
    let data: Vec<f32> = midpoint.to_vec();
    assert!(
        approx_eq(data[0], 1.5, 1e-5),
        "Midpoint: expected 1.5, got {}",
        data[0]
    );
}

#[test]
fn test_quantile_2d_along_axis() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2x5 matrix
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 3.0, 2.0, 5.0, 4.0, 10.0, 30.0, 20.0, 50.0, 40.0],
        &[2, 5],
        &device,
    );

    // Median along dim=1 (reduce each row)
    let result = client.quantile(&a, 0.5, Some(1), false, "linear").unwrap();
    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    assert!(
        approx_eq(data[0], 3.0, 1e-5),
        "Row 0 median: expected 3.0, got {}",
        data[0]
    );
    assert!(
        approx_eq(data[1], 30.0, 1e-5),
        "Row 1 median: expected 30.0, got {}",
        data[1]
    );
}

#[test]
fn test_quantile_keepdim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let result = client.quantile(&a, 0.5, Some(0), true, "linear").unwrap();
    assert_eq!(
        result.shape(),
        &[1],
        "keepdim should preserve dimension as 1"
    );
}

#[test]
fn test_quantile_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[42.0f32], &[1], &device);
    let q = client.quantile(&a, 0.5, Some(0), false, "linear").unwrap();
    let data: Vec<f32> = q.to_vec();
    assert!(approx_eq(data[0], 42.0, 1e-5));
}

#[test]
fn test_quantile_invalid_q() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    // q < 0 should fail
    let result = client.quantile(&a, -0.1, Some(0), false, "linear");
    assert!(result.is_err());

    // q > 1 should fail
    let result = client.quantile(&a, 1.1, Some(0), false, "linear");
    assert!(result.is_err());
}

#[test]
fn test_quantile_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let q = client.quantile(&a, 0.5, Some(0), false, "linear").unwrap();
    let data: Vec<f64> = q.to_vec();
    assert!(approx_eq_f64(data[0], 3.0, 1e-10));
}

// ============================================================================
// Percentile Tests
// ============================================================================

#[test]
fn test_percentile_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    // 50th percentile = median
    let p50 = client.percentile(&a, 50.0, Some(0), false).unwrap();
    let data: Vec<f32> = p50.to_vec();
    assert!(
        approx_eq(data[0], 3.0, 1e-5),
        "P50: expected 3.0, got {}",
        data[0]
    );

    // 0th percentile = min
    let p0 = client.percentile(&a, 0.0, Some(0), false).unwrap();
    let data: Vec<f32> = p0.to_vec();
    assert!(
        approx_eq(data[0], 1.0, 1e-5),
        "P0: expected 1.0, got {}",
        data[0]
    );

    // 100th percentile = max
    let p100 = client.percentile(&a, 100.0, Some(0), false).unwrap();
    let data: Vec<f32> = p100.to_vec();
    assert!(
        approx_eq(data[0], 5.0, 1e-5),
        "P100: expected 5.0, got {}",
        data[0]
    );
}

#[test]
fn test_percentile_invalid_p() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    // p < 0 should fail
    let result = client.percentile(&a, -1.0, Some(0), false);
    assert!(result.is_err());

    // p > 100 should fail
    let result = client.percentile(&a, 101.0, Some(0), false);
    assert!(result.is_err());
}

// ============================================================================
// Median Tests
// ============================================================================

#[test]
fn test_median_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Odd number
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 2.0], &[3], &device);
    let med = client.median(&a, Some(0), false).unwrap();
    let data: Vec<f32> = med.to_vec();
    assert!(
        approx_eq(data[0], 2.0, 1e-5),
        "Odd median: expected 2.0, got {}",
        data[0]
    );

    // Even number
    let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 1.0, 3.0, 2.0], &[4], &device);
    let med = client.median(&b, Some(0), false).unwrap();
    let data: Vec<f32> = med.to_vec();
    assert!(
        approx_eq(data[0], 2.5, 1e-5),
        "Even median: expected 2.5, got {}",
        data[0]
    );
}

#[test]
fn test_median_flatten() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2x3 matrix, find median of all elements
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let med = client.median(&a, None, false).unwrap();
    assert_eq!(med.shape(), &[], "Scalar output when flattening");
    let data: Vec<f32> = med.to_vec();
    // Sorted: [1,2,3,4,5,6], median = (3+4)/2 = 3.5
    assert!(
        approx_eq(data[0], 3.5, 1e-5),
        "Flat median: expected 3.5, got {}",
        data[0]
    );
}
