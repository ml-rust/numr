//! Tests for advanced statistics operations: quantile, percentile, median,
//! histogram, cov, corrcoef, skew, kurtosis

use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Helper Functions
// ============================================================================

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ============================================================================
// Quantile / Percentile / Median Tests
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

// ============================================================================
// Covariance and Correlation Tests
// ============================================================================

#[test]
fn test_cov_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 3 samples, 2 features
    // Feature 0: [1, 2, 3], Feature 1: [4, 5, 6]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2], &device);
    let cov = client.cov(&a, None).unwrap();

    assert_eq!(cov.shape(), &[2, 2]);
    let data: Vec<f32> = cov.to_vec();

    // With default ddof=1 (sample covariance):
    // Var(X) = 1.0, Var(Y) = 1.0, Cov(X,Y) = 1.0
    // Cov matrix = [[1, 1], [1, 1]]
    assert!(
        approx_eq(data[0], 1.0, 1e-5),
        "cov[0,0]: expected 1.0, got {}",
        data[0]
    );
    assert!(
        approx_eq(data[3], 1.0, 1e-5),
        "cov[1,1]: expected 1.0, got {}",
        data[3]
    );
    assert!(
        approx_eq(data[1], 1.0, 1e-5),
        "cov[0,1]: expected 1.0, got {}",
        data[1]
    );
    assert!(
        approx_eq(data[2], 1.0, 1e-5),
        "cov[1,0]: expected 1.0, got {}",
        data[2]
    );
}

#[test]
fn test_cov_symmetry() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &device,
    );
    let cov = client.cov(&a, None).unwrap();
    let data: Vec<f32> = cov.to_vec();

    // Covariance matrix should be symmetric
    assert!(
        approx_eq(data[1], data[3], 1e-5),
        "cov[0,1] should equal cov[1,0]"
    );
    assert!(
        approx_eq(data[2], data[6], 1e-5),
        "cov[0,2] should equal cov[2,0]"
    );
    assert!(
        approx_eq(data[5], data[7], 1e-5),
        "cov[1,2] should equal cov[2,1]"
    );
}

#[test]
fn test_corrcoef_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Perfectly correlated features
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2], &device);
    let corr = client.corrcoef(&a).unwrap();

    assert_eq!(corr.shape(), &[2, 2]);
    let data: Vec<f32> = corr.to_vec();

    // Diagonal should be 1.0
    assert!(
        approx_eq(data[0], 1.0, 1e-4),
        "corr[0,0]: expected 1.0, got {}",
        data[0]
    );
    assert!(
        approx_eq(data[3], 1.0, 1e-4),
        "corr[1,1]: expected 1.0, got {}",
        data[3]
    );

    // Perfect positive correlation
    assert!(
        approx_eq(data[1], 1.0, 1e-4),
        "corr[0,1]: expected 1.0, got {}",
        data[1]
    );
}

#[test]
fn test_corrcoef_range() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Random-ish data
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 5.0, 2.0, 3.0, 4.0, 1.0, 5.0, 2.0, 3.0, 4.0, 6.0, 7.0,
        ],
        &[4, 3],
        &device,
    );
    let corr = client.corrcoef(&a).unwrap();
    let data: Vec<f32> = corr.to_vec();

    // All correlation coefficients should be in [-1, 1]
    for (i, &val) in data.iter().enumerate() {
        assert!(
            val >= -1.0 - 1e-5 && val <= 1.0 + 1e-5,
            "corr[{}] = {} out of range [-1, 1]",
            i,
            val
        );
    }
}

// ============================================================================
// Skewness and Kurtosis Tests
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

// ============================================================================
// Edge Cases
// ============================================================================

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

// ============================================================================
// F64 DType Tests
// ============================================================================

#[test]
fn test_quantile_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let q = client.quantile(&a, 0.5, Some(0), false, "linear").unwrap();
    let data: Vec<f64> = q.to_vec();
    assert!(approx_eq_f64(data[0], 3.0, 1e-10));
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
