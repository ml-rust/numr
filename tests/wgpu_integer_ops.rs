//! Integration tests for WebGPU Integer DType Support (I32, U32)
//!
//! Tests that validate integer operations work correctly on WebGPU backend.
//! This is critical for P1.2 completion.

#![cfg(feature = "wgpu")]

use numr::dtype::DType;
use numr::ops::{CompareOps, TensorOps};
use numr::runtime::Runtime;
use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Binary Operations (I32)
// ============================================================================

#[test]
fn test_i32_add() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);
    let b = Tensor::<WgpuRuntime>::from_slice(&[10i32, 20, 30, 40], &[4], &device);

    let result = client.add(&a, &b).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![11, 22, 33, 44]);
}

#[test]
fn test_i32_sub() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[10i32, 20, 30, 40], &[4], &device);
    let b = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);

    let result = client.sub(&a, &b).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![9, 18, 27, 36]);
}

#[test]
fn test_i32_mul() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[2i32, 3, 4, 5], &[4], &device);
    let b = Tensor::<WgpuRuntime>::from_slice(&[10i32, 10, 10, 10], &[4], &device);

    let result = client.mul(&a, &b).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![20, 30, 40, 50]);
}

// ============================================================================
// Binary Operations (U32)
// ============================================================================

#[test]
fn test_u32_add() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1u32, 2, 3, 4], &[4], &device);
    let b = Tensor::<WgpuRuntime>::from_slice(&[10u32, 20, 30, 40], &[4], &device);

    let result = client.add(&a, &b).unwrap();

    let data: Vec<u32> = result.to_vec();
    assert_eq!(data, vec![11, 22, 33, 44]);
}

#[test]
fn test_u32_mul() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[2u32, 3, 4, 5], &[4], &device);
    let b = Tensor::<WgpuRuntime>::from_slice(&[10u32, 10, 10, 10], &[4], &device);

    let result = client.mul(&a, &b).unwrap();

    let data: Vec<u32> = result.to_vec();
    assert_eq!(data, vec![20, 30, 40, 50]);
}

// ============================================================================
// Unary Operations (I32)
// ============================================================================

#[test]
fn test_i32_neg() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, -2, 3, -4], &[4], &device);

    let result = client.neg(&a).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![-1, 2, -3, 4]);
}

#[test]
fn test_i32_abs() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, -2, 3, -4], &[4], &device);

    let result = client.abs(&a).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![1, 2, 3, 4]);
}

// ============================================================================
// Unary Operations (U32)
// ============================================================================

#[test]
fn test_u32_abs() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1u32, 2, 3, 4], &[4], &device);

    let result = client.abs(&a).unwrap();

    let data: Vec<u32> = result.to_vec();
    assert_eq!(data, vec![1, 2, 3, 4]); // abs of unsigned is identity
}

// ============================================================================
// Float-Only Operations Should Reject Integers
// ============================================================================

#[test]
fn test_i32_sqrt_should_fail() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 4, 9, 16], &[4], &device);

    // sqrt is float-only - should return UnsupportedDType error
    let result = client.sqrt(&a);
    assert!(result.is_err(), "Expected sqrt on I32 to fail");

    // Verify it's the correct error type
    match result {
        Err(numr::error::Error::UnsupportedDType { dtype, op }) => {
            assert_eq!(dtype, DType::I32);
            assert_eq!(op, "sqrt");
        }
        _ => panic!("Expected UnsupportedDType error, got: {:?}", result),
    }
}

#[test]
fn test_i32_exp_should_fail() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);

    // exp is float-only - should return UnsupportedDType error
    let result = client.exp(&a);
    assert!(result.is_err(), "Expected exp on I32 to fail");

    // Verify it's the correct error type
    match result {
        Err(numr::error::Error::UnsupportedDType { dtype, op }) => {
            assert_eq!(dtype, DType::I32);
            assert_eq!(op, "exp");
        }
        _ => panic!("Expected UnsupportedDType error, got: {:?}", result),
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

#[test]
fn test_i32_eq() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);
    let b = Tensor::<WgpuRuntime>::from_slice(&[1i32, 0, 3, 0], &[4], &device);

    let result = client.eq(&a, &b).unwrap();

    // Note: WebGPU compare ops currently output F32 (0.0 or 1.0)
    assert_eq!(result.dtype(), DType::F32);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![1.0, 0.0, 1.0, 0.0]);
}

// ============================================================================
// Reduction Operations (I32)
// ============================================================================

#[test]
fn test_i32_sum() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);

    let result = client.sum(&a, &[], false).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![10]);
}

#[test]
fn test_i32_max() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 20, 3, 40, 5], &[5], &device);

    let result = client.max(&a, &[], false).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![40]);
}

#[test]
fn test_i32_min() {
    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[10i32, 2, 30, 4, 50], &[5], &device);

    let result = client.min(&a, &[], false).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![2]);
}
