//! Integration tests for Eigendecomposition (Symmetric Matrices)
//!
//! Tests verify:
//! - Reconstruction: A @ V ≈ V @ diag(λ) (eigenvalue equation)
//! - Orthogonality: V^T @ V ≈ I
//! - Eigenvalues sorted by magnitude descending
//! - Edge cases: identity, diagonal, single element
//! - Backend parity: CPU, CUDA, WGPU results match expected values

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{LinalgOps, MatmulOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

mod common;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_client() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

/// Assert all values are close within tolerance
fn assert_allclose(a: &[f32], b: &[f32], rtol: f32, atol: f32, msg: &str) {
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

/// Check if matrix is close to identity
fn assert_near_identity(data: &[f32], n: usize, tol: f32, msg: &str) {
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = data[i * n + j];
            let diff = (actual - expected).abs();
            assert!(
                diff <= tol,
                "{}: element [{},{}] differs: {} vs {} (diff={})",
                msg,
                i,
                j,
                actual,
                expected,
                diff
            );
        }
    }
}

/// Check if eigenvalues are sorted by magnitude descending
fn assert_magnitude_descending(eigenvalues: &[f32], msg: &str) {
    for i in 1..eigenvalues.len() {
        let mag_prev = eigenvalues[i - 1].abs();
        let mag_curr = eigenvalues[i].abs();
        assert!(
            mag_prev >= mag_curr - 1e-6,
            "{}: |λ[{}]|={} should be >= |λ[{}]|={}",
            msg,
            i - 1,
            mag_prev,
            i,
            mag_curr
        );
    }
}

/// Compute Frobenius norm of a vector (flattened matrix)
fn frobenius_norm(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============================================================================
// Basic Eigendecomposition Tests
// ============================================================================

#[test]
fn test_eig_2x2_symmetric() {
    let (client, device) = create_client();

    // Symmetric 2x2 matrix: [[2, 1], [1, 2]]
    // Known eigenvalues: 3 and 1
    // Eigenvectors: [1, 1]/sqrt(2) for λ=3, [1, -1]/sqrt(2) for λ=1
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], &device);

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    // Check shapes
    assert_eq!(eig.eigenvalues.shape(), &[2], "eigenvalues shape");
    assert_eq!(eig.eigenvectors.shape(), &[2, 2], "eigenvectors shape");

    // Get data
    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();

    // Check eigenvalues are sorted by magnitude descending
    assert_magnitude_descending(&eigenvalues, "eigenvalues");

    // Known eigenvalues: 3 and 1 (sorted by magnitude: 3, 1)
    assert!(
        (eigenvalues[0] - 3.0).abs() < 1e-5,
        "Expected λ[0]=3, got {}",
        eigenvalues[0]
    );
    assert!(
        (eigenvalues[1] - 1.0).abs() < 1e-5,
        "Expected λ[1]=1, got {}",
        eigenvalues[1]
    );

    // Verify orthogonality: V^T @ V ≈ I
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&vt, &eig.eigenvectors).unwrap();
    let vt_v_data: Vec<f32> = vt_v.to_vec();
    assert_near_identity(&vt_v_data, 2, 1e-5, "V^T @ V");

    // Verify eigenvalue equation: A @ V ≈ V @ diag(λ)
    let av = client.matmul(&a, &eig.eigenvectors).unwrap();
    let lambda_diag = LinalgOps::diagflat(&client, &eig.eigenvalues).unwrap();
    let v_lambda = client.matmul(&eig.eigenvectors, &lambda_diag).unwrap();

    let av_data: Vec<f32> = av.to_vec();
    let v_lambda_data: Vec<f32> = v_lambda.to_vec();
    assert_allclose(&av_data, &v_lambda_data, 1e-5, 1e-5, "eigenvalue equation");
}

#[test]
fn test_eig_3x3_symmetric() {
    let (client, device) = create_client();

    // Symmetric 3x3 matrix: [[4, 1, 1], [1, 4, 1], [1, 1, 4]]
    // Known eigenvalues: 6, 3, 3 (one eigenvalue is 6, two are 3)
    let a = Tensor::<CpuRuntime>::from_slice(
        &[4.0f32, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0],
        &[3, 3],
        &device,
    );

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    // Check shapes
    assert_eq!(eig.eigenvalues.shape(), &[3]);
    assert_eq!(eig.eigenvectors.shape(), &[3, 3]);

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();
    assert_magnitude_descending(&eigenvalues, "eigenvalues");

    // Eigenvalues should be 6 and 3, 3
    assert!(
        (eigenvalues[0] - 6.0).abs() < 1e-4,
        "Expected λ[0]=6, got {}",
        eigenvalues[0]
    );
    assert!(
        (eigenvalues[1] - 3.0).abs() < 1e-4,
        "Expected λ[1]=3, got {}",
        eigenvalues[1]
    );
    assert!(
        (eigenvalues[2] - 3.0).abs() < 1e-4,
        "Expected λ[2]=3, got {}",
        eigenvalues[2]
    );

    // Verify orthogonality: V^T @ V ≈ I
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&vt, &eig.eigenvectors).unwrap();
    let vt_v_data: Vec<f32> = vt_v.to_vec();
    assert_near_identity(&vt_v_data, 3, 1e-4, "V^T @ V");

    // Verify eigenvalue equation
    let av = client.matmul(&a, &eig.eigenvectors).unwrap();
    let lambda_diag = LinalgOps::diagflat(&client, &eig.eigenvalues).unwrap();
    let v_lambda = client.matmul(&eig.eigenvectors, &lambda_diag).unwrap();

    let av_data: Vec<f32> = av.to_vec();
    let v_lambda_data: Vec<f32> = v_lambda.to_vec();
    assert_allclose(&av_data, &v_lambda_data, 1e-4, 1e-4, "eigenvalue equation");
}

#[test]
fn test_eig_identity() {
    let (client, device) = create_client();

    // 3x3 identity matrix - all eigenvalues are 1
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    );

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();

    // All eigenvalues should be 1
    for (i, &val) in eigenvalues.iter().enumerate() {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "λ[{}] should be 1, got {}",
            i,
            val
        );
    }

    // Verify eigenvalue equation
    let av = client.matmul(&a, &eig.eigenvectors).unwrap();
    let lambda_diag = LinalgOps::diagflat(&client, &eig.eigenvalues).unwrap();
    let v_lambda = client.matmul(&eig.eigenvectors, &lambda_diag).unwrap();

    let av_data: Vec<f32> = av.to_vec();
    let v_lambda_data: Vec<f32> = v_lambda.to_vec();
    assert_allclose(
        &av_data,
        &v_lambda_data,
        1e-5,
        1e-5,
        "identity eigenvalue equation",
    );
}

#[test]
fn test_eig_diagonal() {
    let (client, device) = create_client();

    // Diagonal matrix - eigenvalues are the diagonal elements
    let a = Tensor::<CpuRuntime>::from_slice(
        &[5.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    );

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();
    assert_magnitude_descending(&eigenvalues, "eigenvalues");

    // Eigenvalues should be 5, 3, 1 (sorted by magnitude)
    assert!(
        (eigenvalues[0] - 5.0).abs() < 1e-5,
        "Expected λ[0]=5, got {}",
        eigenvalues[0]
    );
    assert!(
        (eigenvalues[1] - 3.0).abs() < 1e-5,
        "Expected λ[1]=3, got {}",
        eigenvalues[1]
    );
    assert!(
        (eigenvalues[2] - 1.0).abs() < 1e-5,
        "Expected λ[2]=1, got {}",
        eigenvalues[2]
    );
}

#[test]
fn test_eig_negative_eigenvalues() {
    let (client, device) = create_client();

    // Matrix with negative eigenvalues: [[0, 1], [1, 0]]
    // Eigenvalues: 1 and -1
    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[2, 2], &device);

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();
    assert_magnitude_descending(&eigenvalues, "eigenvalues");

    // Sorted by magnitude: |1| = |-1|, but 1 should come first
    // Both have magnitude 1, so order may vary
    let sorted_eig: Vec<f32> = {
        let mut v = eigenvalues.clone();
        v.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        v
    };
    assert!(
        (sorted_eig[0].abs() - 1.0).abs() < 1e-5,
        "Expected |λ[0]|=1"
    );
    assert!(
        (sorted_eig[1].abs() - 1.0).abs() < 1e-5,
        "Expected |λ[1]|=1"
    );

    // Verify orthogonality
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&vt, &eig.eigenvectors).unwrap();
    let vt_v_data: Vec<f32> = vt_v.to_vec();
    assert_near_identity(&vt_v_data, 2, 1e-5, "V^T @ V");

    // Verify eigenvalue equation
    let av = client.matmul(&a, &eig.eigenvectors).unwrap();
    let lambda_diag = LinalgOps::diagflat(&client, &eig.eigenvalues).unwrap();
    let v_lambda = client.matmul(&eig.eigenvectors, &lambda_diag).unwrap();

    let av_data: Vec<f32> = av.to_vec();
    let v_lambda_data: Vec<f32> = v_lambda.to_vec();
    assert_allclose(
        &av_data,
        &v_lambda_data,
        1e-5,
        1e-5,
        "eigenvalue equation with negative eigenvalues",
    );
}

// ============================================================================
// F64 Tests
// ============================================================================

#[test]
fn test_eig_f64() {
    let (client, device) = create_client();

    // 2x2 symmetric matrix with F64
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device);

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    assert_eq!(eig.eigenvalues.shape(), &[2]);
    assert_eq!(eig.eigenvectors.shape(), &[2, 2]);

    let eigenvalues: Vec<f64> = eig.eigenvalues.to_vec();

    // Known eigenvalues: 3 and 1
    assert!(
        (eigenvalues[0] - 3.0).abs() < 1e-10,
        "Expected λ[0]=3, got {}",
        eigenvalues[0]
    );
    assert!(
        (eigenvalues[1] - 1.0).abs() < 1e-10,
        "Expected λ[1]=1, got {}",
        eigenvalues[1]
    );

    // Verify orthogonality with F64 precision
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&vt, &eig.eigenvectors).unwrap();
    let vt_v_data: Vec<f64> = vt_v.to_vec();

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = vt_v_data[i * 2 + j];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-10,
                "F64 orthogonality [{},{}] differs: {} vs {} (diff={})",
                i,
                j,
                actual,
                expected,
                diff
            );
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_eig_single_element() {
    let (client, device) = create_client();

    // 1x1 matrix
    let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1, 1], &device);

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    assert_eq!(eig.eigenvalues.shape(), &[1]);
    assert_eq!(eig.eigenvectors.shape(), &[1, 1]);

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();
    let eigenvectors: Vec<f32> = eig.eigenvectors.to_vec();

    assert!(
        (eigenvalues[0] - 5.0).abs() < 1e-5,
        "Expected λ[0]=5, got {}",
        eigenvalues[0]
    );
    assert!(
        (eigenvectors[0] - 1.0).abs() < 1e-5,
        "Expected v[0]=1, got {}",
        eigenvectors[0]
    );
}

#[test]
fn test_eig_larger_matrix() {
    let (client, device) = create_client();

    // Create a symmetric 4x4 matrix
    // Use A = B^T @ B + I for positive definite
    let data = vec![
        1.0f32, 0.5, 0.2, 0.1, 0.5, 2.0, 0.4, 0.2, 0.2, 0.4, 3.0, 0.3, 0.1, 0.2, 0.3, 4.0,
    ];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[4, 4], &device);

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    assert_eq!(eig.eigenvalues.shape(), &[4]);
    assert_eq!(eig.eigenvectors.shape(), &[4, 4]);

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();
    assert_magnitude_descending(&eigenvalues, "eigenvalues");

    // All eigenvalues should be positive (matrix is positive definite)
    for (i, &val) in eigenvalues.iter().enumerate() {
        assert!(val > 0.0, "λ[{}] should be positive, got {}", i, val);
    }

    // Verify orthogonality
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&vt, &eig.eigenvectors).unwrap();
    let vt_v_data: Vec<f32> = vt_v.to_vec();
    assert_near_identity(&vt_v_data, 4, 1e-4, "V^T @ V");

    // Verify eigenvalue equation
    let av = client.matmul(&a, &eig.eigenvectors).unwrap();
    let lambda_diag = LinalgOps::diagflat(&client, &eig.eigenvalues).unwrap();
    let v_lambda = client.matmul(&eig.eigenvectors, &lambda_diag).unwrap();

    let av_data: Vec<f32> = av.to_vec();
    let v_lambda_data: Vec<f32> = v_lambda.to_vec();

    let diff: Vec<f32> = av_data
        .iter()
        .zip(v_lambda_data.iter())
        .map(|(a, b)| a - b)
        .collect();
    let error = frobenius_norm(&diff);
    assert!(
        error < 1e-3,
        "Eigenvalue equation error too large: {}",
        error
    );
}

#[test]
fn test_eig_reconstruction() {
    let (client, device) = create_client();

    // Test reconstruction: A = V @ diag(λ) @ V^T
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 1.0, 3.0], &[2, 2], &device);

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    // Reconstruct: A_reconstructed = V @ diag(λ) @ V^T
    let lambda_diag = LinalgOps::diagflat(&client, &eig.eigenvalues).unwrap();
    let v_lambda = client.matmul(&eig.eigenvectors, &lambda_diag).unwrap();
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let reconstructed = client.matmul(&v_lambda, &vt).unwrap();

    let a_data: Vec<f32> = a.to_vec();
    let reconstructed_data: Vec<f32> = reconstructed.to_vec();

    assert_allclose(&reconstructed_data, &a_data, 1e-5, 1e-5, "reconstruction");
}

// ============================================================================
// Degenerate Cases
// ============================================================================

#[test]
fn test_eig_repeated_eigenvalues() {
    let (client, device) = create_client();

    // Matrix with repeated eigenvalues: all eigenvalues are 2
    let a = Tensor::<CpuRuntime>::from_slice(
        &[2.0f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
        &[3, 3],
        &device,
    );

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();

    // All eigenvalues should be 2
    for (i, &val) in eigenvalues.iter().enumerate() {
        assert!(
            (val - 2.0).abs() < 1e-5,
            "λ[{}] should be 2, got {}",
            i,
            val
        );
    }

    // Eigenvectors should still be orthogonal
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&vt, &eig.eigenvectors).unwrap();
    let vt_v_data: Vec<f32> = vt_v.to_vec();
    assert_near_identity(&vt_v_data, 3, 1e-5, "V^T @ V with repeated eigenvalues");
}

#[test]
fn test_eig_zero_matrix() {
    let (client, device) = create_client();

    // Zero matrix - all eigenvalues are 0
    let a = Tensor::<CpuRuntime>::from_slice(
        &[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[3, 3],
        &device,
    );

    let eig = client.eig_decompose_symmetric(&a).unwrap();

    let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();

    // All eigenvalues should be 0
    for (i, &val) in eigenvalues.iter().enumerate() {
        assert!(val.abs() < 1e-5, "λ[{}] should be 0, got {}", i, val);
    }

    // Eigenvectors should still be orthonormal
    let vt = eig.eigenvectors.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&vt, &eig.eigenvectors).unwrap();
    let vt_v_data: Vec<f32> = vt_v.to_vec();
    assert_near_identity(&vt_v_data, 3, 1e-5, "V^T @ V for zero matrix");
}

// ============================================================================
// CUDA Backend Tests (conditional)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use numr::runtime::cuda::CudaRuntime;

    #[test]
    fn test_cuda_eig_2x2() {
        let Some((client, device)) = common::create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        let a = Tensor::<CudaRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], &device);

        let eig = client.eig_decompose_symmetric(&a).unwrap();

        let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();

        // Known eigenvalues: 3 and 1
        assert!(
            (eigenvalues[0] - 3.0).abs() < 1e-5,
            "Expected λ[0]=3, got {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 1.0).abs() < 1e-5,
            "Expected λ[1]=1, got {}",
            eigenvalues[1]
        );
    }

    #[test]
    fn test_cuda_eig_f64() {
        let Some((client, device)) = common::create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        let a = Tensor::<CudaRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device);

        let eig = client.eig_decompose_symmetric(&a).unwrap();

        let eigenvalues: Vec<f64> = eig.eigenvalues.to_vec();

        // Known eigenvalues: 3 and 1
        assert!(
            (eigenvalues[0] - 3.0).abs() < 1e-10,
            "Expected λ[0]=3, got {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 1.0).abs() < 1e-10,
            "Expected λ[1]=1, got {}",
            eigenvalues[1]
        );
    }

    #[test]
    fn test_cuda_cpu_parity() {
        let Some((cuda_client, cuda_device)) = common::create_cuda_client() else {
            println!("CUDA not available, skipping test");
            return;
        };

        let (cpu_client, cpu_device) = super::create_client();

        let data = vec![3.0f32, 1.0, 1.0, 3.0];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device);
        let a_cuda = Tensor::<CudaRuntime>::from_slice(&data, &[2, 2], &cuda_device);

        let eig_cpu = cpu_client.eig_decompose_symmetric(&a_cpu).unwrap();
        let eig_cuda = cuda_client.eig_decompose_symmetric(&a_cuda).unwrap();

        let cpu_eigenvalues: Vec<f32> = eig_cpu.eigenvalues.to_vec();
        let cuda_eigenvalues: Vec<f32> = eig_cuda.eigenvalues.to_vec();

        // Eigenvalues should match
        for (i, (c, g)) in cpu_eigenvalues
            .iter()
            .zip(cuda_eigenvalues.iter())
            .enumerate()
        {
            let diff = (c - g).abs();
            assert!(
                diff < 1e-5,
                "CPU/CUDA eigenvalue {} mismatch: {} vs {} (diff={})",
                i,
                c,
                g,
                diff
            );
        }
    }
}

// ============================================================================
// WGPU Backend Tests (conditional)
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu_tests {
    use super::*;
    use numr::runtime::wgpu::WgpuRuntime;

    #[test]
    fn test_wgpu_eig_2x2() {
        let Some((client, device)) = common::create_wgpu_client() else {
            println!("WGPU not available, skipping test");
            return;
        };

        let a = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], &device);

        let eig = client.eig_decompose_symmetric(&a).unwrap();

        let eigenvalues: Vec<f32> = eig.eigenvalues.to_vec();

        // Known eigenvalues: 3 and 1
        assert!(
            (eigenvalues[0] - 3.0).abs() < 1e-5,
            "Expected λ[0]=3, got {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 1.0).abs() < 1e-5,
            "Expected λ[1]=1, got {}",
            eigenvalues[1]
        );
    }

    #[test]
    fn test_wgpu_cpu_parity() {
        let Some((wgpu_client, wgpu_device)) = common::create_wgpu_client() else {
            println!("WGPU not available, skipping test");
            return;
        };

        let (cpu_client, cpu_device) = super::create_client();

        let data = vec![3.0f32, 1.0, 1.0, 3.0];

        let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device);
        let a_wgpu = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 2], &wgpu_device);

        let eig_cpu = cpu_client.eig_decompose_symmetric(&a_cpu).unwrap();
        let eig_wgpu = wgpu_client.eig_decompose_symmetric(&a_wgpu).unwrap();

        let cpu_eigenvalues: Vec<f32> = eig_cpu.eigenvalues.to_vec();
        let wgpu_eigenvalues: Vec<f32> = eig_wgpu.eigenvalues.to_vec();

        // Eigenvalues should match
        for (i, (c, w)) in cpu_eigenvalues
            .iter()
            .zip(wgpu_eigenvalues.iter())
            .enumerate()
        {
            let diff = (c - w).abs();
            assert!(
                diff < 1e-5,
                "CPU/WGPU eigenvalue {} mismatch: {} vs {} (diff={})",
                i,
                c,
                w,
                diff
            );
        }
    }
}
