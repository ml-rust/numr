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

// ============================================================================
// Multinomial Tests (Categorical sampling)
// ============================================================================

use numr::tensor::Tensor;

#[test]
fn test_multinomial_1d_with_replacement() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Create probability distribution: [0.1, 0.3, 0.6]
    let probs = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.3, 0.6], &[3], &device);

    let result = client.multinomial(&probs, 10, true).unwrap();

    assert_eq!(result.shape(), &[10]);
    assert_eq!(result.dtype(), DType::I64);

    // All indices should be in [0, 3)
    let data: Vec<i64> = result.to_vec();
    for v in &data {
        assert!(
            *v >= 0 && *v < 3,
            "multinomial index {} out of range [0, 3)",
            v
        );
    }
}

#[test]
fn test_multinomial_2d_with_replacement() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2 distributions, 4 categories each
    let probs = Tensor::<CpuRuntime>::from_slice(
        &[0.1f32, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25],
        &[2, 4],
        &device,
    );

    let result = client.multinomial(&probs, 5, true).unwrap();

    assert_eq!(result.shape(), &[2, 5]);
    assert_eq!(result.dtype(), DType::I64);

    // All indices should be in [0, 4)
    let data: Vec<i64> = result.to_vec();
    for v in &data {
        assert!(
            *v >= 0 && *v < 4,
            "multinomial index {} out of range [0, 4)",
            v
        );
    }
}

#[test]
fn test_multinomial_without_replacement() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Create uniform distribution
    let probs = Tensor::<CpuRuntime>::from_slice(&[0.25f32, 0.25, 0.25, 0.25], &[4], &device);

    let result = client.multinomial(&probs, 4, false).unwrap();

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.dtype(), DType::I64);

    // All indices should be unique for without replacement
    let data: Vec<i64> = result.to_vec();
    let mut sorted = data.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        4,
        "all indices should be unique without replacement"
    );

    // Should be exactly [0, 1, 2, 3] in some order
    assert_eq!(sorted, vec![0, 1, 2, 3]);
}

#[test]
fn test_multinomial_2d_without_replacement() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 3 distributions, 5 categories each
    let probs = Tensor::<CpuRuntime>::from_slice(
        &[0.2f32; 15], // 3 x 5 uniform
        &[3, 5],
        &device,
    );

    let result = client.multinomial(&probs, 3, false).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    assert_eq!(result.dtype(), DType::I64);

    // For each distribution, indices should be unique
    let data: Vec<i64> = result.to_vec();
    for i in 0..3 {
        let mut row: Vec<i64> = vec![data[i * 3], data[i * 3 + 1], data[i * 3 + 2]];
        row.sort();
        row.dedup();
        assert_eq!(row.len(), 3, "row {} should have unique indices", i);
    }
}

#[test]
fn test_multinomial_respects_probabilities() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Heavily biased distribution: category 2 has 90% probability
    let probs = Tensor::<CpuRuntime>::from_slice(&[0.05f32, 0.05, 0.90], &[3], &device);

    // Sample many times
    let result = client.multinomial(&probs, 1000, true).unwrap();
    let data: Vec<i64> = result.to_vec();

    // Count occurrences
    let count_0 = data.iter().filter(|&&x| x == 0).count();
    let count_1 = data.iter().filter(|&&x| x == 1).count();
    let count_2 = data.iter().filter(|&&x| x == 2).count();

    // Category 2 should dominate (should be around 900)
    assert!(
        count_2 > 700,
        "category 2 count {} should be dominant (expected ~900)",
        count_2
    );
    assert!(
        count_0 < 150 && count_1 < 150,
        "categories 0 ({}) and 1 ({}) should be rare",
        count_0,
        count_1
    );
}

#[test]
fn test_multinomial_single_category() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Only one category - should always sample 0
    let probs = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);

    let result = client.multinomial(&probs, 10, true).unwrap();
    let data: Vec<i64> = result.to_vec();

    for v in &data {
        assert_eq!(*v, 0, "single category should always produce index 0");
    }
}

#[test]
fn test_multinomial_unnormalized_probs() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Unnormalized probabilities (sum = 10, not 1)
    let probs = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

    let result = client.multinomial(&probs, 100, true).unwrap();

    // Should still work - multinomial normalizes internally
    assert_eq!(result.shape(), &[100]);

    let data: Vec<i64> = result.to_vec();
    for v in &data {
        assert!(*v >= 0 && *v < 4, "index {} out of range", v);
    }
}

#[test]
fn test_multinomial_f64_input() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let probs = Tensor::<CpuRuntime>::from_slice(&[0.3f64, 0.3, 0.4], &[3], &device);

    let result = client.multinomial(&probs, 10, true).unwrap();

    assert_eq!(result.shape(), &[10]);
    assert_eq!(result.dtype(), DType::I64);
}

// ============================================================================
// Multinomial Error Cases
// ============================================================================

#[test]
fn test_multinomial_zero_samples_fails() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let probs = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 0.5], &[2], &device);

    let result = client.multinomial(&probs, 0, true);
    assert!(result.is_err(), "num_samples=0 should fail");
}

#[test]
fn test_multinomial_integer_input_fails() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let probs = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3], &[3], &device);

    let result = client.multinomial(&probs, 5, true);
    assert!(result.is_err(), "integer input should fail");
}

#[test]
fn test_multinomial_without_replacement_too_many_samples() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let probs = Tensor::<CpuRuntime>::from_slice(&[0.25f32, 0.25, 0.25, 0.25], &[4], &device);

    // Requesting 5 samples without replacement from 4 categories should fail
    let result = client.multinomial(&probs, 5, false);
    assert!(
        result.is_err(),
        "without replacement with num_samples > num_categories should fail"
    );
}

#[test]
fn test_multinomial_3d_input_fails() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let probs = Tensor::<CpuRuntime>::from_slice(&[0.5f32; 8], &[2, 2, 2], &device);

    let result = client.multinomial(&probs, 5, true);
    assert!(
        result.is_err(),
        "3D input should fail (only 1D or 2D supported)"
    );
}

#[test]
fn test_multinomial_all_zero_probs_fails() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // All zero probabilities - should fail
    let probs = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[3], &device);

    let result = client.multinomial(&probs, 1, true);
    assert!(result.is_err(), "all zero probabilities should fail");
}

#[test]
fn test_multinomial_negative_probs_treated_as_zero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Some negative, some positive - should work (negatives treated as zero)
    let probs = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.5, 0.5], &[3], &device);

    let result = client.multinomial(&probs, 100, true).unwrap();
    let data: Vec<i64> = result.to_vec();

    // Should never sample category 0 (negative prob)
    for v in &data {
        assert!(
            *v == 1 || *v == 2,
            "should not sample category with negative prob, got {}",
            v
        );
    }
}
