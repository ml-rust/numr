//! Integration tests for advanced PRNG operations
//!
//! Tests verify:
//! - Correct output shapes
//! - Values in expected ranges ([0,1) for uniform, ~N(0,1) for randn)
//! - Reproducibility (same key+counter → same output)
//! - Different seeds/counters produce different outputs
//! - Statistical properties (mean, variance)
//! - Multiple dtypes (F32, F64)
//! - Edge cases (empty tensors, unsupported dtypes)

use numr::dtype::DType;
use numr::ops::AdvancedRandomOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

// ============================================================================
// Philox Tests
// ============================================================================

#[test]
fn test_philox_uniform_range_f32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.philox_uniform(&[1000], 42, 0, DType::F32).unwrap();

    assert_eq!(result.shape(), &[1000]);
    assert_eq!(result.dtype(), DType::F32);

    // All values in [0, 1)
    let data: Vec<f32> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0, "value {} out of range", v);
    }
}

#[test]
fn test_philox_uniform_range_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.philox_uniform(&[1000], 42, 0, DType::F64).unwrap();

    assert_eq!(result.shape(), &[1000]);
    assert_eq!(result.dtype(), DType::F64);

    let data: Vec<f64> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0, "value {} out of range", v);
    }
}

#[test]
fn test_philox_reproducibility() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.philox_uniform(&[100], 42, 0, DType::F64).unwrap();
    let r2 = client.philox_uniform(&[100], 42, 0, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_eq!(data1, data2, "Same key+counter should produce same output");
}

#[test]
fn test_philox_different_keys() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.philox_uniform(&[100], 42, 0, DType::F64).unwrap();
    let r2 = client.philox_uniform(&[100], 43, 0, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_ne!(
        data1, data2,
        "Different keys should produce different output"
    );
}

#[test]
fn test_philox_different_counters() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.philox_uniform(&[100], 42, 0, DType::F64).unwrap();
    let r2 = client.philox_uniform(&[100], 42, 1, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_ne!(
        data1, data2,
        "Different counters should produce different output"
    );
}

#[test]
fn test_philox_randn_statistics() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.philox_randn(&[10000], 42, 0, DType::F64).unwrap();
    let data: Vec<f64> = result.to_vec();

    let mean: f64 = data.iter().sum::<f64>() / 10000.0;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 10000.0;

    assert!((mean - 0.0).abs() < 0.05, "mean should be ~0, got {}", mean);
    assert!(
        (variance - 1.0).abs() < 0.1,
        "variance should be ~1, got {}",
        variance
    );
}

#[test]
fn test_philox_uniform_mean() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.philox_uniform(&[10000], 42, 0, DType::F64).unwrap();
    let data: Vec<f64> = result.to_vec();

    let mean: f64 = data.iter().sum::<f64>() / 10000.0;

    assert!(
        (mean - 0.5).abs() < 0.05,
        "uniform mean should be ~0.5, got {}",
        mean
    );
}

#[test]
fn test_philox_2d_shape() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.philox_uniform(&[10, 20], 42, 0, DType::F32).unwrap();

    assert_eq!(result.shape(), &[10, 20]);
    assert_eq!(result.numel(), 200);
}

#[test]
fn test_philox_empty_tensor() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.philox_uniform(&[0], 42, 0, DType::F32).unwrap();

    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.numel(), 0);
}

#[test]
fn test_philox_unsupported_dtype() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.philox_uniform(&[10], 42, 0, DType::I32);
    assert!(result.is_err(), "I32 should not be supported");
}

// ============================================================================
// ThreeFry Tests
// ============================================================================

#[test]
fn test_threefry_uniform_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.threefry_uniform(&[1000], 42, 0, DType::F32).unwrap();

    let data: Vec<f32> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0);
    }
}

#[test]
fn test_threefry_reproducibility() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.threefry_uniform(&[100], 42, 0, DType::F64).unwrap();
    let r2 = client.threefry_uniform(&[100], 42, 0, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_eq!(data1, data2);
}

#[test]
fn test_threefry_randn_statistics() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.threefry_randn(&[10000], 42, 0, DType::F64).unwrap();
    let data: Vec<f64> = result.to_vec();

    let mean: f64 = data.iter().sum::<f64>() / 10000.0;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 10000.0;

    assert!((mean - 0.0).abs() < 0.05, "mean = {}", mean);
    assert!((variance - 1.0).abs() < 0.1, "variance = {}", variance);
}

// ============================================================================
// PCG64 Tests
// ============================================================================

#[test]
fn test_pcg64_uniform_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.pcg64_uniform(&[1000], 42, 0, DType::F32).unwrap();

    let data: Vec<f32> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0);
    }
}

#[test]
fn test_pcg64_reproducibility() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.pcg64_uniform(&[100], 42, 0, DType::F64).unwrap();
    let r2 = client.pcg64_uniform(&[100], 42, 0, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_eq!(data1, data2);
}

#[test]
fn test_pcg64_different_streams() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.pcg64_uniform(&[100], 42, 0, DType::F64).unwrap();
    let r2 = client.pcg64_uniform(&[100], 42, 1, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_ne!(
        data1, data2,
        "Different streams should produce different output"
    );
}

#[test]
fn test_pcg64_randn_statistics() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.pcg64_randn(&[10000], 42, 0, DType::F64).unwrap();
    let data: Vec<f64> = result.to_vec();

    let mean: f64 = data.iter().sum::<f64>() / 10000.0;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 10000.0;

    assert!((mean - 0.0).abs() < 0.05, "mean = {}", mean);
    assert!((variance - 1.0).abs() < 0.1, "variance = {}", variance);
}

// ============================================================================
// Xoshiro256++ Tests
// ============================================================================

#[test]
fn test_xoshiro256_uniform_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.xoshiro256_uniform(&[1000], 42, DType::F32).unwrap();

    let data: Vec<f32> = result.to_vec();
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0);
    }
}

#[test]
fn test_xoshiro256_reproducibility() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.xoshiro256_uniform(&[100], 42, DType::F64).unwrap();
    let r2 = client.xoshiro256_uniform(&[100], 42, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_eq!(data1, data2);
}

#[test]
fn test_xoshiro256_different_seeds() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let r1 = client.xoshiro256_uniform(&[100], 42, DType::F64).unwrap();
    let r2 = client.xoshiro256_uniform(&[100], 43, DType::F64).unwrap();

    let data1: Vec<f64> = r1.to_vec();
    let data2: Vec<f64> = r2.to_vec();

    assert_ne!(data1, data2);
}

#[test]
fn test_xoshiro256_randn_statistics() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let result = client.xoshiro256_randn(&[10000], 42, DType::F64).unwrap();
    let data: Vec<f64> = result.to_vec();

    let mean: f64 = data.iter().sum::<f64>() / 10000.0;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 10000.0;

    assert!((mean - 0.0).abs() < 0.05, "mean = {}", mean);
    assert!((variance - 1.0).abs() < 0.1, "variance = {}", variance);
}

// ============================================================================
// Cross-PRNG Comparison Tests
// ============================================================================

#[test]
fn test_all_prngs_produce_different_outputs() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let philox = client.philox_uniform(&[100], 42, 0, DType::F64).unwrap();
    let threefry = client.threefry_uniform(&[100], 42, 0, DType::F64).unwrap();
    let pcg64 = client.pcg64_uniform(&[100], 42, 0, DType::F64).unwrap();
    let xoshiro = client.xoshiro256_uniform(&[100], 42, DType::F64).unwrap();

    let philox_data: Vec<f64> = philox.to_vec();
    let threefry_data: Vec<f64> = threefry.to_vec();
    let pcg64_data: Vec<f64> = pcg64.to_vec();
    let xoshiro_data: Vec<f64> = xoshiro.to_vec();

    // All should be different (independent PRNGs)
    assert_ne!(philox_data, threefry_data);
    assert_ne!(philox_data, pcg64_data);
    assert_ne!(philox_data, xoshiro_data);
    assert_ne!(threefry_data, pcg64_data);
    assert_ne!(threefry_data, xoshiro_data);
    assert_ne!(pcg64_data, xoshiro_data);
}

#[test]
fn test_all_prngs_uniform_mean() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let philox = client.philox_uniform(&[10000], 42, 0, DType::F64).unwrap();
    let threefry = client
        .threefry_uniform(&[10000], 42, 0, DType::F64)
        .unwrap();
    let pcg64 = client.pcg64_uniform(&[10000], 42, 0, DType::F64).unwrap();
    let xoshiro = client.xoshiro256_uniform(&[10000], 42, DType::F64).unwrap();

    let philox_mean: f64 = philox.to_vec().iter().sum::<f64>() / 10000.0;
    let threefry_mean: f64 = threefry.to_vec().iter().sum::<f64>() / 10000.0;
    let pcg64_mean: f64 = pcg64.to_vec().iter().sum::<f64>() / 10000.0;
    let xoshiro_mean: f64 = xoshiro.to_vec().iter().sum::<f64>() / 10000.0;

    // All should have mean ~0.5
    assert!(
        (philox_mean - 0.5).abs() < 0.05,
        "philox mean = {}",
        philox_mean
    );
    assert!(
        (threefry_mean - 0.5).abs() < 0.05,
        "threefry mean = {}",
        threefry_mean
    );
    assert!(
        (pcg64_mean - 0.5).abs() < 0.05,
        "pcg64 mean = {}",
        pcg64_mean
    );
    assert!(
        (xoshiro_mean - 0.5).abs() < 0.05,
        "xoshiro mean = {}",
        xoshiro_mean
    );
}

#[test]
fn test_all_prngs_randn_mean() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let philox = client.philox_randn(&[10000], 42, 0, DType::F64).unwrap();
    let threefry = client.threefry_randn(&[10000], 42, 0, DType::F64).unwrap();
    let pcg64 = client.pcg64_randn(&[10000], 42, 0, DType::F64).unwrap();
    let xoshiro = client.xoshiro256_randn(&[10000], 42, DType::F64).unwrap();

    let philox_mean: f64 = philox.to_vec().iter().sum::<f64>() / 10000.0;
    let threefry_mean: f64 = threefry.to_vec().iter().sum::<f64>() / 10000.0;
    let pcg64_mean: f64 = pcg64.to_vec().iter().sum::<f64>() / 10000.0;
    let xoshiro_mean: f64 = xoshiro.to_vec().iter().sum::<f64>() / 10000.0;

    // All should have mean ~0.0
    assert!(
        (philox_mean - 0.0).abs() < 0.05,
        "philox mean = {}",
        philox_mean
    );
    assert!(
        (threefry_mean - 0.0).abs() < 0.05,
        "threefry mean = {}",
        threefry_mean
    );
    assert!(
        (pcg64_mean - 0.0).abs() < 0.05,
        "pcg64 mean = {}",
        pcg64_mean
    );
    assert!(
        (xoshiro_mean - 0.0).abs() < 0.05,
        "xoshiro mean = {}",
        xoshiro_mean
    );
}
