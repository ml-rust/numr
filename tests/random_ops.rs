//! Integration tests for random operations (rand, randn, randint)
//!
//! Tests verify:
//! - Correct output shapes
//! - Values in expected ranges
//! - Multiple dtypes
//! - Statistical properties (mean, variance for randn)
//! - Edge cases

use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

// ============================================================================
// Rand Tests (Uniform [0, 1))
// ============================================================================

#[test]
fn test_rand_1d_f32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.rand(&[100], DType::F32).unwrap();

    assert_eq!(result.shape(), &[100]);
    assert_eq!(result.dtype(), DType::F32);

    // All values should be in [0, 1)
    let data: Vec<f32> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0, "rand value {} out of range", v);
    }
}

#[test]
fn test_rand_2d_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.rand(&[10, 10], DType::F64).unwrap();

    assert_eq!(result.shape(), &[10, 10]);
    assert_eq!(result.dtype(), DType::F64);

    // All values should be in [0, 1)
    let data: Vec<f64> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0, "rand value {} out of range", v);
    }
}

#[test]
fn test_rand_3d_shape() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.rand(&[2, 3, 4], DType::F32).unwrap();

    assert_eq!(result.shape(), &[2, 3, 4]);
    assert_eq!(result.numel(), 24);
}

#[test]
fn test_rand_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.rand(&[0], DType::F32).unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.numel(), 0);
}

#[test]
fn test_rand_mean_near_half() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Generate many samples
    let result = client.rand(&[10000], DType::F32).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Mean of uniform [0, 1) should be approximately 0.5
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(
        (mean - 0.5).abs() < 0.05,
        "rand mean {} should be close to 0.5",
        mean
    );
}

// ============================================================================
// Randn Tests (Standard Normal N(0, 1))
// ============================================================================

#[test]
fn test_randn_1d_f32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randn(&[100], DType::F32).unwrap();

    assert_eq!(result.shape(), &[100]);
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_randn_2d_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randn(&[10, 10], DType::F64).unwrap();

    assert_eq!(result.shape(), &[10, 10]);
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn test_randn_3d_shape() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randn(&[2, 3, 4], DType::F32).unwrap();

    assert_eq!(result.shape(), &[2, 3, 4]);
    assert_eq!(result.numel(), 24);
}

#[test]
fn test_randn_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randn(&[0], DType::F32).unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.numel(), 0);
}

#[test]
fn test_randn_mean_near_zero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Generate many samples
    let result = client.randn(&[10000], DType::F32).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Mean of N(0, 1) should be approximately 0
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.1, "randn mean {} should be close to 0", mean);
}

#[test]
fn test_randn_std_near_one() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Generate many samples
    let result = client.randn(&[10000], DType::F32).unwrap();
    let data: Vec<f32> = result.to_vec();

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 =
        data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
    let std = variance.sqrt();

    // Std of N(0, 1) should be approximately 1
    assert!(
        (std - 1.0).abs() < 0.1,
        "randn std {} should be close to 1",
        std
    );
}

// ============================================================================
// Randint Tests (Uniform integers in [low, high))
// ============================================================================

#[test]
fn test_randint_1d_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randint(0, 10, &[100], DType::I32).unwrap();

    assert_eq!(result.shape(), &[100]);
    assert_eq!(result.dtype(), DType::I32);

    // All values should be in [0, 10)
    let data: Vec<i32> = result.to_vec();
    for v in &data {
        assert!(
            *v >= 0 && *v < 10,
            "randint value {} out of range [0, 10)",
            v
        );
    }
}

#[test]
fn test_randint_2d_i64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randint(-5, 5, &[10, 10], DType::I64).unwrap();

    assert_eq!(result.shape(), &[10, 10]);
    assert_eq!(result.dtype(), DType::I64);

    // All values should be in [-5, 5)
    let data: Vec<i64> = result.to_vec();
    for v in &data {
        assert!(
            *v >= -5 && *v < 5,
            "randint value {} out of range [-5, 5)",
            v
        );
    }
}

#[test]
fn test_randint_unsigned_u32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randint(0, 100, &[100], DType::U32).unwrap();

    assert_eq!(result.shape(), &[100]);
    assert_eq!(result.dtype(), DType::U32);

    // All values should be in [0, 100)
    let data: Vec<u32> = result.to_vec();
    for v in &data {
        assert!(*v < 100, "randint value {} out of range [0, 100)", v);
    }
}

#[test]
fn test_randint_unsigned_u64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randint(10, 20, &[50], DType::U64).unwrap();

    assert_eq!(result.shape(), &[50]);
    assert_eq!(result.dtype(), DType::U64);

    // All values should be in [10, 20)
    let data: Vec<u64> = result.to_vec();
    for v in &data {
        assert!(
            *v >= 10 && *v < 20,
            "randint value {} out of range [10, 20)",
            v
        );
    }
}

#[test]
fn test_randint_i8_small_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Small range to easily verify distribution
    let result = client.randint(0, 2, &[100], DType::I8).unwrap();

    assert_eq!(result.shape(), &[100]);
    assert_eq!(result.dtype(), DType::I8);

    // All values should be 0 or 1
    let data: Vec<i8> = result.to_vec();
    for v in &data {
        assert!(*v == 0 || *v == 1, "randint value {} should be 0 or 1", v);
    }

    // Both 0 and 1 should appear (with high probability)
    let zeros = data.iter().filter(|&&x| x == 0).count();
    let ones = data.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "both 0 and 1 should appear");
}

#[test]
fn test_randint_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randint(0, 10, &[0], DType::I32).unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.numel(), 0);
}

#[test]
fn test_randint_negative_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Range entirely in negative numbers
    let result = client.randint(-100, -50, &[100], DType::I32).unwrap();

    let data: Vec<i32> = result.to_vec();
    for v in &data {
        assert!(
            *v >= -100 && *v < -50,
            "randint value {} out of range [-100, -50)",
            v
        );
    }
}

#[test]
fn test_randint_large_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Large range
    let result = client.randint(0, 1_000_000, &[1000], DType::I64).unwrap();

    let data: Vec<i64> = result.to_vec();
    for v in &data {
        assert!(
            *v >= 0 && *v < 1_000_000,
            "randint value {} out of range [0, 1000000)",
            v
        );
    }
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_randint_invalid_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // high <= low should fail
    let result = client.randint(10, 5, &[10], DType::I32);
    assert!(result.is_err());

    let result = client.randint(5, 5, &[10], DType::I32);
    assert!(result.is_err());
}

#[test]
fn test_randint_unsigned_negative_low() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Negative low with unsigned dtype should fail
    let result = client.randint(-1, 10, &[10], DType::U32);
    assert!(result.is_err());
}

#[test]
fn test_randint_float_dtype_fails() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Float dtype should fail for randint
    let result = client.randint(0, 10, &[10], DType::F32);
    assert!(result.is_err());
}

// ============================================================================
// Multi-dimensional Tests
// ============================================================================

#[test]
fn test_rand_high_dimensional() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.rand(&[2, 3, 4, 5], DType::F32).unwrap();

    assert_eq!(result.shape(), &[2, 3, 4, 5]);
    assert_eq!(result.numel(), 120);

    let data: Vec<f32> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0);
    }
}

#[test]
fn test_randn_high_dimensional() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randn(&[2, 3, 4, 5], DType::F32).unwrap();

    assert_eq!(result.shape(), &[2, 3, 4, 5]);
    assert_eq!(result.numel(), 120);
}

#[test]
fn test_randint_high_dimensional() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.randint(0, 10, &[2, 3, 4, 5], DType::I32).unwrap();

    assert_eq!(result.shape(), &[2, 3, 4, 5]);
    assert_eq!(result.numel(), 120);

    let data: Vec<i32> = result.to_vec();
    for v in &data {
        assert!(*v >= 0 && *v < 10);
    }
}
