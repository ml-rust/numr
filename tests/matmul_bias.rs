//! Comprehensive tests for fused matrix multiplication with bias operation.
//!
//! Tests the `matmul_bias` operation which computes C = A @ B + bias
//! where the bias addition is fused into the GEMM epilogue for efficiency.

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ComplexOps, ConditionalOps, CumulativeOps, IndexingOps,
    LinalgOps, LogicalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    StatisticalOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Basic 2D matmul_bias tests
// ============================================================================

#[test]
fn test_matmul_bias_2x2() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // A @ B = [[19, 22], [43, 50]]
    // bias = [1, 2]
    // C = A @ B + bias = [[20, 24], [44, 52]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [20.0, 24.0, 44.0, 52.0]);
}

#[test]
fn test_matmul_bias_3x2_2x4() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A = [[1, 2], [3, 4], [5, 6]] (3x2)
    // B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
    // A @ B (3x4) = [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
    // bias = [0.1, 0.2, 0.3, 0.4]
    // C = A @ B + bias
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    );
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[3, 4]);
    let result: Vec<f32> = c.to_vec();
    let expected = [
        11.1, 14.2, 17.3, 20.4, // Row 0
        23.1, 30.2, 37.3, 44.4, // Row 1
        35.1, 46.2, 57.3, 68.4, // Row 2
    ];
    for (got, want) in result.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5, "got {} want {}", got, want);
    }
}

#[test]
fn test_matmul_bias_1x1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Scalar case: 1x1 @ 1x1 + bias[1]
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1, 1], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32], &[1, 1], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[1, 1]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [14.0]); // 3*4 + 2 = 14
}

#[test]
fn test_matmul_bias_large_matrices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Larger matrices to test blocking/tiling
    let m = 64;
    let k = 32;
    let n = 48;

    // Create matrices with simple patterns for easy verification
    let a_data: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32).collect();
    let bias_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[m, n]);

    // Verify against reference implementation
    let matmul_result = client.matmul(&a, &b).unwrap();
    let expected = client
        .add(&matmul_result, &bias.broadcast_to(&[m, n]).unwrap())
        .unwrap();

    let c_data: Vec<f32> = c.to_vec();
    let expected_data: Vec<f32> = expected.to_vec();

    for (got, want) in c_data.iter().zip(expected_data.iter()) {
        assert!(
            (got - want).abs() < 1e-4,
            "Large matrix mismatch: got {} want {}",
            got,
            want
        );
    }
}

// ============================================================================
// Batched matmul_bias tests
// ============================================================================

#[test]
fn test_matmul_bias_batched_2x2() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Batch of 2 matrices
    // A[0] = [[1, 2], [3, 4]], A[1] = [[5, 6], [7, 8]]
    // B[0] = [[1, 0], [0, 1]], B[1] = [[2, 0], [0, 2]]
    // bias = [0.5, 1.0]
    //
    // C[0] = A[0] @ B[0] + bias = [[1, 2], [3, 4]] + [0.5, 1.0] = [[1.5, 3], [3.5, 5]]
    // C[1] = A[1] @ B[1] + bias = [[10, 12], [14, 16]] + [0.5, 1.0] = [[10.5, 13], [14.5, 17]]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
        &device,
    );
    let b = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        &[2, 2, 2],
        &device,
    );
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.0], &[2], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[2, 2, 2]);
    let result: Vec<f32> = c.to_vec();
    let expected = [1.5, 3.0, 3.5, 5.0, 10.5, 13.0, 14.5, 17.0];
    for (got, want) in result.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "Batched mismatch: got {} want {}",
            got,
            want
        );
    }
}

#[test]
fn test_matmul_bias_batched_larger() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let batch_size = 4;
    let m = 8;
    let k = 6;
    let n = 10;

    // Create test data
    let a_data: Vec<f32> = (0..batch_size * m * k)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..batch_size * k * n)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();
    let bias_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[batch_size, m, k], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[batch_size, k, n], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[batch_size, m, n]);

    // Verify against reference
    let matmul_result = client.matmul(&a, &b).unwrap();
    let bias_broadcast = bias.broadcast_to(&[batch_size, m, n]).unwrap();
    let expected = client.add(&matmul_result, &bias_broadcast).unwrap();

    let c_data: Vec<f32> = c.to_vec();
    let expected_data: Vec<f32> = expected.to_vec();

    for (i, (got, want)) in c_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "Batched larger mismatch at {}: got {} want {}",
            i,
            got,
            want
        );
    }
}

// ============================================================================
// Different dtypes
// ============================================================================

#[test]
fn test_matmul_bias_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.1f64, 0.2], &[2], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.dtype(), DType::F64);

    let result: Vec<f64> = c.to_vec();
    // A @ B = [[19, 22], [43, 50]]
    // + bias = [[19.1, 22.2], [43.1, 50.2]]
    let expected = [19.1, 22.2, 43.1, 50.2];
    for (got, want) in result.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-10,
            "F64 mismatch: got {} want {}",
            got,
            want
        );
    }
}

// ============================================================================
// Zero bias (should match plain matmul)
// ============================================================================

#[test]
fn test_matmul_bias_zero_bias() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device);

    let c_bias = client.matmul_bias(&a, &b, &bias).unwrap();
    let c_matmul = client.matmul(&a, &b).unwrap();

    let result_bias: Vec<f32> = c_bias.to_vec();
    let result_matmul: Vec<f32> = c_matmul.to_vec();

    assert_eq!(result_bias, result_matmul);
}

// ============================================================================
// Validation error tests
// ============================================================================

#[test]
fn test_matmul_bias_shape_mismatch_inner_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A is 2x3, B is 2x2 - inner dimensions don't match
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 2], &[2], &device);

    let result = client.matmul_bias(&a, &b, &bias);
    assert!(result.is_err());
}

#[test]
fn test_matmul_bias_wrong_bias_size() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A is 2x3, B is 3x4, output is 2x4, but bias has 3 elements
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, 4], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 3], &[3], &device); // Wrong: should be [4]

    let result = client.matmul_bias(&a, &b, &bias);
    assert!(result.is_err());
}

#[test]
fn test_matmul_bias_bias_not_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // bias is 2D, should be 1D
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device); // Wrong: should be [2]

    let result = client.matmul_bias(&a, &b, &bias);
    assert!(result.is_err());
}

#[test]
fn test_matmul_bias_dtype_mismatch() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // a is F32, b is F64
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 4], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 2], &[2], &device);

    let result = client.matmul_bias(&a, &b, &bias);
    assert!(result.is_err());
}

#[test]
fn test_matmul_bias_bias_dtype_mismatch() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // a and b are F32, bias is F64
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 2], &[2], &device);

    let result = client.matmul_bias(&a, &b, &bias);
    assert!(result.is_err());
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_matmul_bias_single_row() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A is 1x3, B is 3x2, C is 1x2
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.0], &[2], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[1, 2]);
    let result: Vec<f32> = c.to_vec();
    // A @ B = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    // + bias = [22.5, 29]
    assert!((result[0] - 22.5).abs() < 1e-5);
    assert!((result[1] - 29.0).abs() < 1e-5);
}

#[test]
fn test_matmul_bias_single_col() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A is 3x2, B is 2x1, C is 3x1
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2, 1], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[3, 1]);
    let result: Vec<f32> = c.to_vec();
    // A @ B = [[5], [11], [17]]
    // + bias = [[15], [21], [27]]
    assert_eq!(result, [15.0, 21.0, 27.0]);
}

#[test]
fn test_matmul_bias_negative_bias() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[-19.0f32, -22.0], &[2], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<f32> = c.to_vec();
    // A @ B = [[19, 22], [43, 50]]
    // + bias = [[0, 0], [24, 28]]
    assert_eq!(result, [0.0, 0.0, 24.0, 28.0]);
}

// ============================================================================
// Numerical accuracy test
// ============================================================================

#[test]
fn test_matmul_bias_numerical_accuracy() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Test with values that might cause floating-point issues
    let a = Tensor::<CpuRuntime>::from_slice(&[1e-6f32, 1e6, 1e-6, 1e6], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1e6f32, 1e-6, 1e6, 1e-6], &[2, 2], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);

    let c = client.matmul_bias(&a, &b, &bias).unwrap();

    // Compare with separate matmul + add
    let matmul_result = client.matmul(&a, &b).unwrap();
    let expected = client
        .add(&matmul_result, &bias.broadcast_to(&[2, 2]).unwrap())
        .unwrap();

    let c_data: Vec<f32> = c.to_vec();
    let expected_data: Vec<f32> = expected.to_vec();

    for (got, want) in c_data.iter().zip(expected_data.iter()) {
        let rel_err = if want.abs() > 1e-10 {
            (got - want).abs() / want.abs()
        } else {
            (got - want).abs()
        };
        assert!(
            rel_err < 1e-5,
            "Numerical accuracy issue: got {} want {}, rel_err {}",
            got,
            want,
            rel_err
        );
    }
}
