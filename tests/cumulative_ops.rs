//! Integration tests for cumulative operations (cumsum, cumprod, logsumexp)
//!
//! Tests verify correctness across:
//! - Different dimensions
//! - Negative dimension indexing
//! - Multiple dtypes
//! - Edge cases

use numr::dtype::DType;
use numr::ops::CumulativeOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Cumsum Tests
// ============================================================================

#[test]
fn test_cumsum_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let result = client.cumsum(&a, 0).unwrap();

    assert_eq!(result.shape(), &[5]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 3.0, 6.0, 10.0, 15.0]);
}

#[test]
fn test_cumsum_2d_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [3, 2]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    let result = client.cumsum(&a, 0).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let data: Vec<f32> = result.to_vec();
    // Column-wise cumsum:
    // [1, 2] -> [1, 2]
    // [3, 4] -> [4, 6]
    // [5, 6] -> [9, 12]
    assert_eq!(data, [1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_cumsum_2d_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 4]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    );

    let result = client.cumsum(&a, 1).unwrap();

    assert_eq!(result.shape(), &[2, 4]);
    let data: Vec<f32> = result.to_vec();
    // Row-wise cumsum:
    // [1, 2, 3, 4] -> [1, 3, 6, 10]
    // [5, 6, 7, 8] -> [5, 11, 18, 26]
    assert_eq!(data, [1.0, 3.0, 6.0, 10.0, 5.0, 11.0, 18.0, 26.0]);
}

#[test]
fn test_cumsum_negative_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    // dim=-1 should be equivalent to dim=1
    let result = client.cumsum(&a, -1).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 3.0, 3.0, 7.0]);
}

#[test]
fn test_cumsum_3d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
        &device,
    );

    // Cumsum along last dimension
    let result = client.cumsum(&a, 2).unwrap();

    assert_eq!(result.shape(), &[2, 2, 3]);
    let data: Vec<f32> = result.to_vec();
    // Each row of 3 elements gets cumsum'd
    assert_eq!(
        data,
        [
            1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 7.0, 15.0, 24.0, 10.0, 21.0, 33.0
        ]
    );
}

#[test]
fn test_cumsum_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4, 5], &[5], &device);

    let result = client.cumsum(&a, 0).unwrap();

    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, [1, 3, 6, 10, 15]);
}

#[test]
fn test_cumsum_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[4], &device);

    let result = client.cumsum(&a, 0).unwrap();

    assert_eq!(result.dtype(), DType::F64);
    let data: Vec<f64> = result.to_vec();
    assert_eq!(data, [1.0, 3.0, 6.0, 10.0]);
}

// ============================================================================
// Cumprod Tests
// ============================================================================

#[test]
fn test_cumprod_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let result = client.cumprod(&a, 0).unwrap();

    assert_eq!(result.shape(), &[5]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 6.0, 24.0, 120.0]);
}

#[test]
fn test_cumprod_2d_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [3, 2]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    let result = client.cumprod(&a, 0).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let data: Vec<f32> = result.to_vec();
    // Column-wise cumprod:
    // [1, 2] -> [1, 2]
    // [3, 4] -> [3, 8]
    // [5, 6] -> [15, 48]
    assert_eq!(data, [1.0, 2.0, 3.0, 8.0, 15.0, 48.0]);
}

#[test]
fn test_cumprod_2d_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 4]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
        &[2, 4],
        &device,
    );

    let result = client.cumprod(&a, 1).unwrap();

    assert_eq!(result.shape(), &[2, 4]);
    let data: Vec<f32> = result.to_vec();
    // Row-wise cumprod:
    // [1, 2, 3, 4] -> [1, 2, 6, 24]
    // [2, 3, 4, 5] -> [2, 6, 24, 120]
    assert_eq!(data, [1.0, 2.0, 6.0, 24.0, 2.0, 6.0, 24.0, 120.0]);
}

#[test]
fn test_cumprod_negative_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    // dim=-1 should be equivalent to dim=1
    let result = client.cumprod(&a, -1).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    // [1, 2] -> [1, 2]
    // [3, 4] -> [3, 12]
    assert_eq!(data, [1.0, 2.0, 3.0, 12.0]);
}

#[test]
fn test_cumprod_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4, 5], &[5], &device);

    let result = client.cumprod(&a, 0).unwrap();

    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, [1, 2, 6, 24, 120]);
}

// ============================================================================
// Logsumexp Tests
// ============================================================================

#[test]
fn test_logsumexp_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    // With keepdim=true to get shape [1]
    let result = client.logsumexp(&a, &[0], true).unwrap();

    assert_eq!(result.shape(), &[1]);
    let data: Vec<f32> = result.to_vec();
    // logsumexp([1, 2, 3]) = log(e^1 + e^2 + e^3) ≈ 3.407606
    let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    assert!(
        (data[0] - expected).abs() < 1e-5,
        "Got {}, expected {}",
        data[0],
        expected
    );
}

#[test]
fn test_logsumexp_1d_no_keepdim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    // With keepdim=false, output is scalar (0-dim)
    let result = client.logsumexp(&a, &[0], false).unwrap();

    // For 1D reduction, shape becomes [] (scalar)
    assert_eq!(result.shape(), &[] as &[usize]);
    let data: Vec<f32> = result.to_vec();
    let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    assert!(
        (data[0] - expected).abs() < 1e-5,
        "Got {}, expected {}",
        data[0],
        expected
    );
}

#[test]
fn test_logsumexp_2d_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [3, 2]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    let result = client.logsumexp(&a, &[0], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();

    // Column 0: logsumexp([1, 3, 5])
    let exp0 = (1.0f32.exp() + 3.0f32.exp() + 5.0f32.exp()).ln();
    // Column 1: logsumexp([2, 4, 6])
    let exp1 = (2.0f32.exp() + 4.0f32.exp() + 6.0f32.exp()).ln();

    assert!(
        (data[0] - exp0).abs() < 1e-5,
        "Got {}, expected {}",
        data[0],
        exp0
    );
    assert!(
        (data[1] - exp1).abs() < 1e-5,
        "Got {}, expected {}",
        data[1],
        exp1
    );
}

#[test]
fn test_logsumexp_2d_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    let result = client.logsumexp(&a, &[1], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();

    // Row 0: logsumexp([1, 2, 3])
    let exp0 = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    // Row 1: logsumexp([4, 5, 6])
    let exp1 = (4.0f32.exp() + 5.0f32.exp() + 6.0f32.exp()).ln();

    assert!(
        (data[0] - exp0).abs() < 1e-5,
        "Got {}, expected {}",
        data[0],
        exp0
    );
    assert!(
        (data[1] - exp1).abs() < 1e-5,
        "Got {}, expected {}",
        data[1],
        exp1
    );
}

#[test]
fn test_logsumexp_keepdim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    let result = client.logsumexp(&a, &[1], true).unwrap();

    // With keepdim=true, shape should be [2, 1]
    assert_eq!(result.shape(), &[2, 1]);
}

#[test]
fn test_logsumexp_full_reduction() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    // Full reduction: pass all dims explicitly
    let result = client.logsumexp(&a, &[0, 1], false).unwrap();

    // Shape becomes [] (scalar) after full reduction with keepdim=false
    assert_eq!(result.shape(), &[] as &[usize]);
    let data: Vec<f32> = result.to_vec();

    let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp() + 4.0f32.exp()).ln();
    assert!(
        (data[0] - expected).abs() < 1e-5,
        "Got {}, expected {}",
        data[0],
        expected
    );
}

#[test]
fn test_logsumexp_empty_dims_returns_copy() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    // Empty dims means no reduction - returns copy
    let result = client.logsumexp(&a, &[], false).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_logsumexp_numerical_stability() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Test with large values that would overflow if done naively
    let a = Tensor::<CpuRuntime>::from_slice(&[1000.0f32, 1001.0, 1002.0], &[3], &device);

    let result = client.logsumexp(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();

    // Using the stable formula: max(x) + log(sum(exp(x - max(x))))
    // max = 1002
    // logsumexp = 1002 + log(exp(-2) + exp(-1) + exp(0))
    //           = 1002 + log(e^-2 + e^-1 + 1)
    let expected = 1002.0 + ((-2.0f32).exp() + (-1.0f32).exp() + 1.0).ln();

    assert!(
        (data[0] - expected).abs() < 1e-3,
        "Got {}, expected {}",
        data[0],
        expected
    );
    assert!(data[0].is_finite(), "Result should be finite");
}

#[test]
fn test_logsumexp_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

    let result = client.logsumexp(&a, &[0], false).unwrap();

    assert_eq!(result.dtype(), DType::F64);
    let data: Vec<f64> = result.to_vec();
    let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
    assert!((data[0] - expected).abs() < 1e-10);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_cumsum_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[42.0f32], &[1], &device);

    let result = client.cumsum(&a, 0).unwrap();

    assert_eq!(result.shape(), &[1]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [42.0]);
}

#[test]
fn test_cumprod_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[42.0f32], &[1], &device);

    let result = client.cumprod(&a, 0).unwrap();

    assert_eq!(result.shape(), &[1]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [42.0]);
}

#[test]
fn test_logsumexp_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device);

    let result = client.logsumexp(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // logsumexp of single element is the element itself
    assert!((data[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_cumprod_with_zeros() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 3.0, 4.0], &[4], &device);

    let result = client.cumprod(&a, 0).unwrap();

    let data: Vec<f32> = result.to_vec();
    // After zero, all subsequent products should be zero
    assert_eq!(data, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_cumsum_with_negatives() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -2.0, 3.0, -4.0], &[4], &device);

    let result = client.cumsum(&a, 0).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, -1.0, 2.0, -2.0]);
}
