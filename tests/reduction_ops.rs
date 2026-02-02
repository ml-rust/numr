//! Integration tests for reduction operations (prod, any, all)
//!
//! Tests verify correctness across:
//! - Different dimensions
//! - Keepdim variations
//! - Multiple dtypes
//! - Edge cases

use numr::dtype::DType;
use numr::ops::ReduceOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Prod Tests
// ============================================================================

#[test]
fn test_prod_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let result = client.prod(&a, &[0], false).unwrap();

    // prod([1, 2, 3, 4, 5]) = 120
    assert_eq!(result.shape(), &[] as &[usize]);
    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 120.0).abs() < 1e-5);
}

#[test]
fn test_prod_1d_keepdim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

    let result = client.prod(&a, &[0], true).unwrap();

    assert_eq!(result.shape(), &[1]);
    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 24.0).abs() < 1e-5);
}

#[test]
fn test_prod_2d_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [3, 2]: [[1, 2], [3, 4], [5, 6]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    let result = client.prod(&a, &[0], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // Column 0: 1 * 3 * 5 = 15
    // Column 1: 2 * 4 * 6 = 48
    assert!((data[0] - 15.0).abs() < 1e-5);
    assert!((data[1] - 48.0).abs() < 1e-5);
}

#[test]
fn test_prod_2d_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 4]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
        &[2, 4],
        &device,
    );

    let result = client.prod(&a, &[1], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // Row 0: 1 * 2 * 3 * 4 = 24
    // Row 1: 2 * 3 * 4 * 5 = 120
    assert!((data[0] - 24.0).abs() < 1e-5);
    assert!((data[1] - 120.0).abs() < 1e-5);
}

#[test]
fn test_prod_full_reduction() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    let result = client.prod(&a, &[0, 1], false).unwrap();

    // Full product: 1 * 2 * 3 * 4 = 24
    assert_eq!(result.shape(), &[] as &[usize]);
    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 24.0).abs() < 1e-5);
}

#[test]
fn test_prod_with_zeros() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 3.0, 4.0], &[4], &device);

    let result = client.prod(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // Product with zero should be zero
    assert!((data[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_prod_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[42.0f32], &[1], &device);

    let result = client.prod(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 42.0).abs() < 1e-5);
}

#[test]
fn test_prod_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);

    let result = client.prod(&a, &[0], false).unwrap();

    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data[0], 24);
}

#[test]
fn test_prod_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[4], &device);

    let result = client.prod(&a, &[0], false).unwrap();

    assert_eq!(result.dtype(), DType::F64);
    let data: Vec<f64> = result.to_vec();
    assert!((data[0] - 24.0).abs() < 1e-10);
}

// ============================================================================
// Any Tests (returns 1.0 if any element is non-zero, 0.0 otherwise)
// ============================================================================

#[test]
fn test_any_all_nonzero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

    let result = client.any(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // All elements are non-zero, so result should be 1.0
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_any_with_zeros() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 3.0, 0.0], &[4], &device);

    let result = client.any(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // At least one element is non-zero (3.0), so result should be 1.0
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_any_all_zeros() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

    let result = client.any(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // All elements are zero, so result should be 0.0
    assert!((data[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_any_2d_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [3, 2]: [[0, 1], [0, 0], [3, 0]]
    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 0.0, 0.0, 3.0, 0.0], &[3, 2], &device);

    let result = client.any(&a, &[0], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // Column 0: any(0, 0, 3) = 1.0 (has non-zero)
    // Column 1: any(1, 0, 0) = 1.0 (has non-zero)
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_any_2d_dim1_with_zero_row() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 3]: [[1, 0, 2], [0, 0, 0]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 2.0, 0.0, 0.0, 0.0], &[2, 3], &device);

    let result = client.any(&a, &[1], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // Row 0: any(1, 0, 2) = 1.0 (has non-zero)
    // Row 1: any(0, 0, 0) = 0.0 (all zero)
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 0.0).abs() < 1e-5);
}

#[test]
fn test_any_keepdim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 2.0], &[2, 2], &device);

    let result = client.any(&a, &[1], true).unwrap();

    assert_eq!(result.shape(), &[2, 1]);
    let data: Vec<f32> = result.to_vec();
    // Row 0: any(1, 0) = 1.0
    // Row 1: any(0, 2) = 1.0
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_any_negative_values() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Negative values are also non-zero
    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, -1.0, 0.0], &[3], &device);

    let result = client.any(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-5);
}

// ============================================================================
// All Tests (returns 1.0 if all elements are non-zero, 0.0 otherwise)
// ============================================================================

#[test]
fn test_all_all_nonzero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

    let result = client.all(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // All elements are non-zero, so result should be 1.0
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_all_with_zero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 3.0, 4.0], &[4], &device);

    let result = client.all(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // Not all elements are non-zero (has 0.0), so result should be 0.0
    assert!((data[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_all_all_zeros() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

    let result = client.all(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    // None are non-zero, so result should be 0.0
    assert!((data[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_all_2d_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [3, 2]: [[1, 1], [2, 0], [3, 1]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 2.0, 0.0, 3.0, 1.0], &[3, 2], &device);

    let result = client.all(&a, &[0], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // Column 0: all(1, 2, 3) = 1.0 (all non-zero)
    // Column 1: all(1, 0, 1) = 0.0 (has zero)
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 0.0).abs() < 1e-5);
}

#[test]
fn test_all_2d_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 3]: [[1, 2, 3], [1, 0, 2]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 1.0, 0.0, 2.0], &[2, 3], &device);

    let result = client.all(&a, &[1], false).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // Row 0: all(1, 2, 3) = 1.0 (all non-zero)
    // Row 1: all(1, 0, 2) = 0.0 (has zero)
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 0.0).abs() < 1e-5);
}

#[test]
fn test_all_keepdim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    let result = client.all(&a, &[1], true).unwrap();

    assert_eq!(result.shape(), &[2, 1]);
    let data: Vec<f32> = result.to_vec();
    // Row 0: all(1, 2) = 1.0
    // Row 1: all(3, 4) = 1.0
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_all_single_element_nonzero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device);

    let result = client.all(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_all_single_element_zero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);

    let result = client.all(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_all_negative_values() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Negative values are non-zero
    let a = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, -2.0, -3.0], &[3], &device);

    let result = client.all(&a, &[0], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_all_full_reduction() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    let result = client.all(&a, &[0, 1], false).unwrap();

    assert_eq!(result.shape(), &[] as &[usize]);
    let data: Vec<f32> = result.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-5);
}

// ============================================================================
// 3D Tests
// ============================================================================

#[test]
fn test_prod_3d_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 3, 2] - 12 elements
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // batch 1
        ],
        &[2, 3, 2],
        &device,
    );

    let result = client.prod(&a, &[1], false).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    // batch 0, col 0: 1 * 3 * 5 = 15
    // batch 0, col 1: 2 * 4 * 6 = 48
    // batch 1, col 0: 2 * 4 * 6 = 48
    // batch 1, col 1: 3 * 5 * 7 = 105
    assert!((data[0] - 15.0).abs() < 1e-5);
    assert!((data[1] - 48.0).abs() < 1e-5);
    assert!((data[2] - 48.0).abs() < 1e-5);
    assert!((data[3] - 105.0).abs() < 1e-5);
}

#[test]
fn test_any_3d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            0.0f32, 0.0, 1.0, // batch 0, row 0: has nonzero
            0.0, 0.0, 0.0, // batch 0, row 1: all zero
            1.0, 0.0, 0.0, // batch 1, row 0: has nonzero
            0.0, 0.0, 0.0, // batch 1, row 1: all zero
        ],
        &[2, 2, 3],
        &device,
    );

    let result = client.any(&a, &[2], false).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    // batch 0, row 0: any(0, 0, 1) = 1.0
    // batch 0, row 1: any(0, 0, 0) = 0.0
    // batch 1, row 0: any(1, 0, 0) = 1.0
    // batch 1, row 1: any(0, 0, 0) = 0.0
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 0.0).abs() < 1e-5);
    assert!((data[2] - 1.0).abs() < 1e-5);
    assert!((data[3] - 0.0).abs() < 1e-5);
}

#[test]
fn test_all_3d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 2.0, 3.0, // batch 0, row 0: all nonzero
            1.0, 0.0, 3.0, // batch 0, row 1: has zero
            1.0, 2.0, 3.0, // batch 1, row 0: all nonzero
            4.0, 5.0, 6.0, // batch 1, row 1: all nonzero
        ],
        &[2, 2, 3],
        &device,
    );

    let result = client.all(&a, &[2], false).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    // batch 0, row 0: all(1, 2, 3) = 1.0
    // batch 0, row 1: all(1, 0, 3) = 0.0
    // batch 1, row 0: all(1, 2, 3) = 1.0
    // batch 1, row 1: all(4, 5, 6) = 1.0
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 0.0).abs() < 1e-5);
    assert!((data[2] - 1.0).abs() < 1e-5);
    assert!((data[3] - 1.0).abs() < 1e-5);
}
