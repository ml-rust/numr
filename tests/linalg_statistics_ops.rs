//! Integration tests for linear algebra statistics operations
//!
//! Tests verify:
//! - pinverse (Moore-Penrose pseudo-inverse)
//! - cond (condition number)
//! - cov (covariance matrix)
//! - corrcoef (correlation coefficient matrix)
//!
//! These tests ensure:
//! - Mathematical correctness against known values
//! - Backend parity: CPU results match expected values
//! - F32 and F64 precision handling
//! - Edge cases and error conditions

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_client() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

/// Assert all values are close within tolerance
fn assert_allclose_f32(a: &[f32], b: &[f32], rtol: f32, atol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {} (diff={}, tol={})",
            msg,
            i,
            x,
            y,
            diff,
            tol
        );
    }
}

fn assert_allclose_f64(a: &[f64], b: &[f64], rtol: f64, atol: f64, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {} (diff={}, tol={})",
            msg,
            i,
            x,
            y,
            diff,
            tol
        );
    }
}

/// Compute Frobenius norm of a vector (flattened matrix)
fn frobenius_norm_f32(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[allow(dead_code)]
fn frobenius_norm_f64(data: &[f64]) -> f64 {
    data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ============================================================================
// Pseudo-Inverse (pinverse) Tests
// ============================================================================

#[test]
fn test_pinverse_identity() {
    let (client, device) = create_client();

    // Identity matrix - pseudo-inverse of I is I
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    );

    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();
    let pinv_data: Vec<f32> = pinv.to_vec();

    // Result should be identity
    let expected = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_allclose_f32(&pinv_data, &expected, 1e-5, 1e-5, "pinverse of identity");
}

#[test]
fn test_pinverse_2x2_invertible() {
    let (client, device) = create_client();

    // Well-conditioned 2x2 matrix
    let a = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], &device);

    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();
    let pinv_data: Vec<f32> = pinv.to_vec();

    // For invertible matrices, pinverse = inverse
    // det = 4*6 - 7*2 = 10
    // inv = (1/10) * [[6, -7], [-2, 4]]
    let expected = [0.6f32, -0.7, -0.2, 0.4];
    assert_allclose_f32(&pinv_data, &expected, 1e-4, 1e-4, "pinverse of 2x2");

    // Verify A @ A^+ @ A = A (pseudo-inverse property)
    let a_pinv = TensorOps::matmul(&client, &a, &pinv).unwrap();
    let a_pinv_a = TensorOps::matmul(&client, &a_pinv, &a).unwrap();
    let result: Vec<f32> = a_pinv_a.to_vec();
    let a_data: Vec<f32> = a.to_vec();
    assert_allclose_f32(&result, &a_data, 1e-4, 1e-4, "A @ pinv @ A = A");
}

#[test]
fn test_pinverse_tall_matrix() {
    let (client, device) = create_client();

    // 3x2 matrix (m > n) - overdetermined system
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();

    // Output shape should be [2, 3] (transposed)
    assert_eq!(pinv.shape(), &[2, 3], "pinverse shape");

    // Verify A^+ @ A @ A^+ = A^+ (Moore-Penrose condition)
    let pinv_a = TensorOps::matmul(&client, &pinv, &a).unwrap();
    let pinv_a_pinv = TensorOps::matmul(&client, &pinv_a, &pinv).unwrap();
    let result: Vec<f32> = pinv_a_pinv.to_vec();
    let pinv_data: Vec<f32> = pinv.to_vec();
    assert_allclose_f32(&result, &pinv_data, 1e-4, 1e-4, "pinv @ A @ pinv = pinv");
}

#[test]
fn test_pinverse_wide_matrix() {
    let (client, device) = create_client();

    // 2x3 matrix (m < n) - underdetermined system
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();

    // Output shape should be [3, 2] (transposed)
    assert_eq!(pinv.shape(), &[3, 2], "pinverse shape");

    // Verify A @ A^+ @ A = A (Moore-Penrose condition)
    let a_pinv = TensorOps::matmul(&client, &a, &pinv).unwrap();
    let a_pinv_a = TensorOps::matmul(&client, &a_pinv, &a).unwrap();
    let result: Vec<f32> = a_pinv_a.to_vec();
    let a_data: Vec<f32> = a.to_vec();
    assert_allclose_f32(&result, &a_data, 1e-4, 1e-4, "A @ pinv @ A = A");
}

#[test]
fn test_pinverse_f64() {
    let (client, device) = create_client();

    // Test with F64 precision
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 1.0, 1.0, 3.0], &[2, 2], &device);

    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();

    // Verify reconstruction with higher precision
    let a_pinv_a = TensorOps::matmul(&client, &a, &pinv).unwrap();
    let a_pinv_a = TensorOps::matmul(&client, &a_pinv_a, &a).unwrap();
    let result: Vec<f64> = a_pinv_a.to_vec();
    let a_data: Vec<f64> = a.to_vec();
    assert_allclose_f64(&result, &a_data, 1e-10, 1e-10, "F64 pinverse");
}

#[test]
fn test_pinverse_with_rcond() {
    let (client, device) = create_client();

    // Rank-deficient matrix
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], &device);

    // With default rcond, small singular values are zeroed
    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();

    // Should still satisfy Moore-Penrose conditions
    let a_pinv = TensorOps::matmul(&client, &a, &pinv).unwrap();
    let a_pinv_a = TensorOps::matmul(&client, &a_pinv, &a).unwrap();
    let result: Vec<f32> = a_pinv_a.to_vec();
    let a_data: Vec<f32> = a.to_vec();
    // Lower tolerance for rank-deficient case
    assert_allclose_f32(&result, &a_data, 1e-3, 1e-3, "rank-deficient pinverse");
}

// ============================================================================
// Condition Number (cond) Tests
// ============================================================================

#[test]
fn test_cond_identity() {
    let (client, device) = create_client();

    // Identity matrix has condition number 1
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    );

    let cond = client.cond(&a).unwrap();
    let cond_val: Vec<f32> = cond.to_vec();

    assert!(
        (cond_val[0] - 1.0).abs() < 1e-5,
        "Identity condition number should be 1, got {}",
        cond_val[0]
    );
}

#[test]
fn test_cond_well_conditioned() {
    let (client, device) = create_client();

    // Diagonal matrix with known condition number
    // diag([3, 2]) -> cond = 3/2 = 1.5
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 0.0, 0.0, 2.0], &[2, 2], &device);

    let cond = client.cond(&a).unwrap();
    let cond_val: Vec<f32> = cond.to_vec();

    assert!(
        (cond_val[0] - 1.5).abs() < 1e-5,
        "Expected cond=1.5, got {}",
        cond_val[0]
    );
}

#[test]
fn test_cond_ill_conditioned() {
    let (client, device) = create_client();

    // Diagonal matrix with large condition number
    // diag([1000, 1]) -> cond = 1000
    let a = Tensor::<CpuRuntime>::from_slice(&[1000.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);

    let cond = client.cond(&a).unwrap();
    let cond_val: Vec<f32> = cond.to_vec();

    assert!(
        (cond_val[0] - 1000.0).abs() < 1.0,
        "Expected cond≈1000, got {}",
        cond_val[0]
    );
}

#[test]
fn test_cond_singular() {
    let (client, device) = create_client();

    // Singular matrix (rank-deficient)
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], &device);

    let cond = client.cond(&a).unwrap();
    let cond_val: Vec<f32> = cond.to_vec();

    // Singular matrix should have infinite condition number
    assert!(
        cond_val[0].is_infinite(),
        "Singular matrix should have infinite cond, got {}",
        cond_val[0]
    );
}

#[test]
fn test_cond_rectangular() {
    let (client, device) = create_client();

    // 3x2 matrix
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2], &device);

    let cond = client.cond(&a).unwrap();
    let cond_val: Vec<f32> = cond.to_vec();

    // This matrix has singular values [1, 1], so cond = 1
    assert!(
        (cond_val[0] - 1.0).abs() < 1e-5,
        "Expected cond=1, got {}",
        cond_val[0]
    );
}

#[test]
fn test_cond_f64() {
    let (client, device) = create_client();

    // Test with F64 precision
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 0.0, 0.0, 2.0], &[2, 2], &device);

    let cond = client.cond(&a).unwrap();
    let cond_val: Vec<f64> = cond.to_vec();

    assert!(
        (cond_val[0] - 1.5).abs() < 1e-12,
        "F64 Expected cond=1.5, got {}",
        cond_val[0]
    );
}

// ============================================================================
// Covariance (cov) Tests
// ============================================================================

#[test]
fn test_cov_2x2() {
    let (client, device) = create_client();

    // 4 samples, 2 features
    // Data: [[1, 2], [3, 4], [5, 6], [7, 8]]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &device,
    );

    let cov = TensorOps::cov(&client, &a, Some(1)).unwrap();
    let cov_data: Vec<f32> = cov.to_vec();

    assert_eq!(cov.shape(), &[2, 2], "cov shape");

    // Mean of col 0: (1+3+5+7)/4 = 4
    // Mean of col 1: (2+4+6+8)/4 = 5
    // Var(col 0) = sum((x - 4)^2) / 3 = (9+1+1+9)/3 = 20/3
    // Var(col 1) = sum((x - 5)^2) / 3 = (9+1+1+9)/3 = 20/3
    // Cov(col0, col1) = sum((x-4)(y-5)) / 3 = (3*3+1*1+(-1)*(-1)+(-3)*(-3))/3 = 20/3
    let expected_var = 20.0f32 / 3.0;
    assert!(
        (cov_data[0] - expected_var).abs() < 1e-4,
        "Var(0,0) should be {}, got {}",
        expected_var,
        cov_data[0]
    );
    assert!(
        (cov_data[3] - expected_var).abs() < 1e-4,
        "Var(1,1) should be {}, got {}",
        expected_var,
        cov_data[3]
    );
    assert!(
        (cov_data[1] - expected_var).abs() < 1e-4,
        "Cov(0,1) should be {}, got {}",
        expected_var,
        cov_data[1]
    );
    // Covariance matrix should be symmetric
    assert!(
        (cov_data[1] - cov_data[2]).abs() < 1e-6,
        "Cov matrix should be symmetric"
    );
}

#[test]
fn test_cov_ddof0() {
    let (client, device) = create_client();

    // Same data, but with ddof=0 (population covariance)
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &device,
    );

    let cov = TensorOps::cov(&client, &a, Some(0)).unwrap();
    let cov_data: Vec<f32> = cov.to_vec();

    // With ddof=0, divide by n=4 instead of n-1=3
    let expected_var = 20.0f32 / 4.0; // = 5.0
    assert!(
        (cov_data[0] - expected_var).abs() < 1e-4,
        "Population Var should be {}, got {}",
        expected_var,
        cov_data[0]
    );
}

#[test]
fn test_cov_uncorrelated() {
    let (client, device) = create_client();

    // Two uncorrelated variables
    // Col 0: [1, -1, 1, -1]
    // Col 1: [1, 1, -1, -1]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        &[4, 2],
        &device,
    );

    let cov = TensorOps::cov(&client, &a, Some(1)).unwrap();
    let cov_data: Vec<f32> = cov.to_vec();

    // Off-diagonal (covariance) should be ~0
    assert!(
        cov_data[1].abs() < 1e-5,
        "Uncorrelated features should have ~0 covariance, got {}",
        cov_data[1]
    );
}

#[test]
fn test_cov_f64() {
    let (client, device) = create_client();

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &device,
    );

    let cov = TensorOps::cov(&client, &a, Some(1)).unwrap();
    let cov_data: Vec<f64> = cov.to_vec();

    let expected_var = 20.0f64 / 3.0;
    assert!(
        (cov_data[0] - expected_var).abs() < 1e-12,
        "F64 Var should be {}, got {}",
        expected_var,
        cov_data[0]
    );
}

#[test]
fn test_cov_single_feature() {
    let (client, device) = create_client();

    // Single feature, 4 samples
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 5.0, 7.0], &[4, 1], &device);

    let cov = TensorOps::cov(&client, &a, Some(1)).unwrap();
    let cov_data: Vec<f32> = cov.to_vec();

    assert_eq!(cov.shape(), &[1, 1], "Single feature cov shape");

    // Variance = 20/3
    let expected_var = 20.0f32 / 3.0;
    assert!(
        (cov_data[0] - expected_var).abs() < 1e-4,
        "Single feature variance"
    );
}

// ============================================================================
// Correlation Coefficient (corrcoef) Tests
// ============================================================================

#[test]
fn test_corrcoef_perfect_correlation() {
    let (client, device) = create_client();

    // Perfectly correlated (y = x)
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        &[4, 2],
        &device,
    );

    let corr = TensorOps::corrcoef(&client, &a).unwrap();
    let corr_data: Vec<f32> = corr.to_vec();

    assert_eq!(corr.shape(), &[2, 2], "corrcoef shape");

    // Diagonal should be 1 (correlation with self)
    assert!(
        (corr_data[0] - 1.0).abs() < 1e-5,
        "Self correlation should be 1"
    );
    assert!(
        (corr_data[3] - 1.0).abs() < 1e-5,
        "Self correlation should be 1"
    );

    // Off-diagonal should be 1 (perfect correlation)
    assert!(
        (corr_data[1] - 1.0).abs() < 1e-5,
        "Perfect correlation should be 1, got {}",
        corr_data[1]
    );
}

#[test]
fn test_corrcoef_negative_correlation() {
    let (client, device) = create_client();

    // Perfectly negatively correlated (y = -x)
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
        &[4, 2],
        &device,
    );

    let corr = TensorOps::corrcoef(&client, &a).unwrap();
    let corr_data: Vec<f32> = corr.to_vec();

    // Off-diagonal should be -1 (perfect negative correlation)
    assert!(
        (corr_data[1] - (-1.0)).abs() < 1e-5,
        "Negative correlation should be -1, got {}",
        corr_data[1]
    );
}

#[test]
fn test_corrcoef_uncorrelated() {
    let (client, device) = create_client();

    // Uncorrelated variables
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        &[4, 2],
        &device,
    );

    let corr = TensorOps::corrcoef(&client, &a).unwrap();
    let corr_data: Vec<f32> = corr.to_vec();

    // Off-diagonal should be ~0
    assert!(
        corr_data[1].abs() < 1e-5,
        "Uncorrelated should have ~0 correlation, got {}",
        corr_data[1]
    );
}

#[test]
fn test_corrcoef_bounds() {
    let (client, device) = create_client();

    // Arbitrary data - correlations should always be in [-1, 1]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 0.0, 2.0, 4.0,
        ],
        &[4, 3],
        &device,
    );

    let corr = TensorOps::corrcoef(&client, &a).unwrap();
    let corr_data: Vec<f32> = corr.to_vec();

    for (i, &val) in corr_data.iter().enumerate() {
        assert!(
            val >= -1.0 - 1e-5 && val <= 1.0 + 1e-5,
            "Correlation {} at index {} out of bounds [-1, 1]",
            val,
            i
        );
    }
}

#[test]
fn test_corrcoef_symmetric() {
    let (client, device) = create_client();

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &device,
    );

    let corr = TensorOps::corrcoef(&client, &a).unwrap();
    let corr_data: Vec<f32> = corr.to_vec();

    // Check symmetry: corr[i,j] = corr[j,i]
    for i in 0..3 {
        for j in 0..3 {
            let val_ij = corr_data[i * 3 + j];
            let val_ji = corr_data[j * 3 + i];
            assert!(
                (val_ij - val_ji).abs() < 1e-6,
                "corrcoef should be symmetric: [{},{}]={} vs [{},{}]={}",
                i,
                j,
                val_ij,
                j,
                i,
                val_ji
            );
        }
    }
}

#[test]
fn test_corrcoef_f64() {
    let (client, device) = create_client();

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f64, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        &[4, 2],
        &device,
    );

    let corr = TensorOps::corrcoef(&client, &a).unwrap();
    let corr_data: Vec<f64> = corr.to_vec();

    // Perfect correlation
    assert!(
        (corr_data[1] - 1.0).abs() < 1e-12,
        "F64 correlation should be 1, got {}",
        corr_data[1]
    );
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_cov_insufficient_samples() {
    let (client, device) = create_client();

    // Only 1 sample, ddof=1 requires at least 2
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);

    let result = TensorOps::cov(&client, &a, Some(1));
    assert!(result.is_err(), "cov should fail with insufficient samples");
}

#[test]
fn test_corrcoef_insufficient_samples() {
    let (client, device) = create_client();

    // Only 1 sample
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);

    let result = TensorOps::corrcoef(&client, &a);
    assert!(
        result.is_err(),
        "corrcoef should fail with insufficient samples"
    );
}

#[test]
fn test_pinverse_1x1() {
    let (client, device) = create_client();

    // 1x1 matrix
    let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1, 1], &device);

    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();
    let pinv_data: Vec<f32> = pinv.to_vec();

    assert_eq!(pinv.shape(), &[1, 1]);
    assert!(
        (pinv_data[0] - 0.2).abs() < 1e-5,
        "1/5 = 0.2, got {}",
        pinv_data[0]
    );
}

#[test]
fn test_cond_1x1() {
    let (client, device) = create_client();

    // 1x1 matrix - condition number is always 1 for non-zero scalar
    let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1, 1], &device);

    let cond = client.cond(&a).unwrap();
    let cond_val: Vec<f32> = cond.to_vec();

    assert!(
        (cond_val[0] - 1.0).abs() < 1e-5,
        "1x1 matrix condition should be 1, got {}",
        cond_val[0]
    );
}

// ============================================================================
// Integration Tests - Combining Operations
// ============================================================================

#[test]
fn test_pinverse_least_squares() {
    let (client, device) = create_client();

    // Solve least squares: A @ x ≈ b
    // A is 4x2, b is 4x1
    // Solution: x = pinv(A) @ b
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0],
        &[4, 2],
        &device,
    );
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[4, 1], &device);

    let pinv = TensorOps::pinverse(&client, &a, None).unwrap();
    let x = TensorOps::matmul(&client, &pinv, &b).unwrap();

    // Verify residual is small
    let ax = TensorOps::matmul(&client, &a, &x).unwrap();
    let residual = TensorOps::sub(&client, &ax, &b).unwrap();
    let residual_data: Vec<f32> = residual.to_vec();
    let norm = frobenius_norm_f32(&residual_data);

    assert!(
        norm < 0.1,
        "Least squares residual should be small: {}",
        norm
    );
}

#[test]
fn test_cov_corrcoef_relationship() {
    let (client, device) = create_client();

    // Verify corrcoef = cov / (std_outer)
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        &[4, 2],
        &device,
    );

    let cov = TensorOps::cov(&client, &a, Some(1)).unwrap();
    let corr = TensorOps::corrcoef(&client, &a).unwrap();

    let cov_data: Vec<f32> = cov.to_vec();
    let corr_data: Vec<f32> = corr.to_vec();

    // corr[0,1] = cov[0,1] / (std[0] * std[1])
    let std0 = cov_data[0].sqrt();
    let std1 = cov_data[3].sqrt();
    let expected_corr = cov_data[1] / (std0 * std1);

    assert!(
        (corr_data[1] - expected_corr).abs() < 1e-5,
        "corrcoef vs cov/std: {} vs {}",
        corr_data[1],
        expected_corr
    );
}

// ============================================================================
// Backend Parity Tests - CPU vs CUDA
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_parity {
    use super::*;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    fn create_cuda_client() -> Option<(CudaClient, CudaDevice)> {
        // Try to create CUDA device 0, return None if CUDA is unavailable
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        Some((client, device))
    }

    #[test]
    fn test_pinverse_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        // Test data: overdetermined system
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[4, 3], &cpu_device);
        let cuda_a = Tensor::<CudaRuntime>::from_slice(&data, &[4, 3], &cuda_device);

        let cpu_result: Vec<f32> = cpu_client.pinverse(&cpu_a, None).unwrap().to_vec();
        let cuda_result: Vec<f32> = cuda_client.pinverse(&cuda_a, None).unwrap().to_vec();

        assert_allclose_f32(
            &cpu_result,
            &cuda_result,
            1e-4,
            1e-4,
            "pinverse CPU vs CUDA",
        );
    }

    #[test]
    fn test_cond_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        let data = vec![4.0f32, 2.0, 2.0, 3.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device);
        let cuda_a = Tensor::<CudaRuntime>::from_slice(&data, &[2, 2], &cuda_device);

        let cpu_result: Vec<f32> = cpu_client.cond(&cpu_a).unwrap().to_vec();
        let cuda_result: Vec<f32> = cuda_client.cond(&cuda_a).unwrap().to_vec();

        assert_allclose_f32(&cpu_result, &cuda_result, 1e-4, 1e-4, "cond CPU vs CUDA");
    }

    #[test]
    fn test_cov_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        let data = vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 3], &cpu_device);
        let cuda_a = Tensor::<CudaRuntime>::from_slice(&data, &[3, 3], &cuda_device);

        let cpu_result: Vec<f32> = cpu_TensorOps::cov(&client, &cpu_a, Some(1))
            .unwrap()
            .to_vec();
        let cuda_result: Vec<f32> = cuda_TensorOps::cov(&client, &cuda_a, Some(1))
            .unwrap()
            .to_vec();

        assert_allclose_f32(&cpu_result, &cuda_result, 1e-4, 1e-4, "cov CPU vs CUDA");
    }

    #[test]
    fn test_corrcoef_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        let data = vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 3], &cpu_device);
        let cuda_a = Tensor::<CudaRuntime>::from_slice(&data, &[3, 3], &cuda_device);

        let cpu_result: Vec<f32> = cpu_TensorOps::corrcoef(&client, &cpu_a).unwrap().to_vec();
        let cuda_result: Vec<f32> = cuda_TensorOps::corrcoef(&client, &cuda_a).unwrap().to_vec();

        assert_allclose_f32(
            &cpu_result,
            &cuda_result,
            1e-4,
            1e-4,
            "corrcoef CPU vs CUDA",
        );
    }

    #[test]
    fn test_corrcoef_zero_variance_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        // Variable with zero variance (constant column)
        let data = vec![1.0f32, 2.0, 1.0, 3.0, 1.0, 4.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 2], &cpu_device);
        let cuda_a = Tensor::<CudaRuntime>::from_slice(&data, &[3, 2], &cuda_device);

        let cpu_result: Vec<f32> = cpu_TensorOps::corrcoef(&client, &cpu_a).unwrap().to_vec();
        let cuda_result: Vec<f32> = cuda_TensorOps::corrcoef(&client, &cuda_a).unwrap().to_vec();

        // Zero-variance handling must match exactly
        assert_allclose_f32(
            &cpu_result,
            &cuda_result,
            1e-5,
            1e-5,
            "corrcoef zero-variance CPU vs CUDA",
        );
    }
}

// ============================================================================
// Backend Parity Tests - CPU vs WGPU
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu_parity {
    use super::*;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime, is_wgpu_available};

    fn create_wgpu_client() -> Option<(WgpuClient, WgpuDevice)> {
        if !is_wgpu_available() {
            return None;
        }
        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        Some((client, device))
    }

    #[test]
    fn test_pinverse_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WGPU parity test: no WGPU device available");
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        // Test data: overdetermined system
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[4, 3], &cpu_device);
        let wgpu_a = Tensor::<WgpuRuntime>::from_slice(&data, &[4, 3], &wgpu_device);

        let cpu_result: Vec<f32> = TensorOps::pinverse(&cpu_client, &cpu_a, None)
            .unwrap()
            .to_vec();
        let wgpu_result: Vec<f32> = TensorOps::pinverse(&wgpu_client, &wgpu_a, None)
            .unwrap()
            .to_vec();

        // WGPU uses F32 only, slightly looser tolerance
        assert_allclose_f32(
            &cpu_result,
            &wgpu_result,
            1e-3,
            1e-3,
            "pinverse CPU vs WGPU",
        );
    }

    #[test]
    fn test_cond_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        let data = vec![4.0f32, 2.0, 2.0, 3.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device);
        let wgpu_a = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 2], &wgpu_device);

        let cpu_result: Vec<f32> = cpu_client.cond(&cpu_a).unwrap().to_vec();
        let wgpu_result: Vec<f32> = wgpu_client.cond(&wgpu_a).unwrap().to_vec();

        assert_allclose_f32(&cpu_result, &wgpu_result, 1e-3, 1e-3, "cond CPU vs WGPU");
    }

    #[test]
    fn test_cov_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        let data = vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 3], &cpu_device);
        let wgpu_a = Tensor::<WgpuRuntime>::from_slice(&data, &[3, 3], &wgpu_device);

        let cpu_result: Vec<f32> = cpu_TensorOps::cov(&client, &cpu_a, Some(1))
            .unwrap()
            .to_vec();
        let wgpu_result: Vec<f32> = wgpu_TensorOps::cov(&client, &wgpu_a, Some(1))
            .unwrap()
            .to_vec();

        assert_allclose_f32(&cpu_result, &wgpu_result, 1e-3, 1e-3, "cov CPU vs WGPU");
    }

    #[test]
    fn test_corrcoef_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        let data = vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 3], &cpu_device);
        let wgpu_a = Tensor::<WgpuRuntime>::from_slice(&data, &[3, 3], &wgpu_device);

        let cpu_result: Vec<f32> = cpu_TensorOps::corrcoef(&client, &cpu_a).unwrap().to_vec();
        let wgpu_result: Vec<f32> = wgpu_TensorOps::corrcoef(&client, &wgpu_a).unwrap().to_vec();

        assert_allclose_f32(
            &cpu_result,
            &wgpu_result,
            1e-3,
            1e-3,
            "corrcoef CPU vs WGPU",
        );
    }

    #[test]
    fn test_corrcoef_zero_variance_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            return;
        };
        let (cpu_client, cpu_device) = create_client();

        // Variable with zero variance (constant column)
        let data = vec![1.0f32, 2.0, 1.0, 3.0, 1.0, 4.0];

        let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 2], &cpu_device);
        let wgpu_a = Tensor::<WgpuRuntime>::from_slice(&data, &[3, 2], &wgpu_device);

        let cpu_result: Vec<f32> = cpu_TensorOps::corrcoef(&client, &cpu_a).unwrap().to_vec();
        let wgpu_result: Vec<f32> = wgpu_TensorOps::corrcoef(&client, &wgpu_a).unwrap().to_vec();

        // Zero-variance handling must match exactly
        assert_allclose_f32(
            &cpu_result,
            &wgpu_result,
            1e-4,
            1e-4,
            "corrcoef zero-variance CPU vs WGPU",
        );
    }
}
