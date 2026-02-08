//! Integration tests for Polynomial Operations
//!
//! Tests verify:
//! - polyroots: Find roots of polynomials (real and complex)
//! - polyval: Evaluate polynomials using Horner's method
//! - polyfromroots: Construct polynomial from roots
//! - polymul: Multiply polynomials via convolution
//! - Roundtrip: polyroots → polyfromroots ≈ original (for monic polynomials)
//! - Backend parity: CPU, CUDA, WGPU results match

use numr::algorithm::polynomial::PolynomialAlgorithms;
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

// ============================================================================
// polyroots Tests
// ============================================================================

#[test]
fn test_polyroots_linear() {
    let (client, device) = create_client();

    // p(x) = -2 + x = 0 → x = 2
    // coeffs: [-2, 1]
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[-2.0f32, 1.0], &[2], &device);

    let roots = client.polyroots(&coeffs).unwrap();

    let real: Vec<f32> = roots.roots_real.to_vec();
    let imag: Vec<f32> = roots.roots_imag.to_vec();

    assert_eq!(real.len(), 1);
    assert!(
        (real[0] - 2.0).abs() < 1e-5,
        "Root should be 2, got {}",
        real[0]
    );
    assert!(imag[0].abs() < 1e-5, "Root should be real");
}

#[test]
fn test_polyroots_quadratic_two_real() {
    let (client, device) = create_client();

    // x² - 5x + 6 = (x-2)(x-3), roots: 2, 3
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[6.0f32, -5.0, 1.0], &[3], &device);

    let roots = client.polyroots(&coeffs).unwrap();

    let real: Vec<f32> = roots.roots_real.to_vec();
    let imag: Vec<f32> = roots.roots_imag.to_vec();

    assert_eq!(real.len(), 2);

    // Sort roots for comparison
    let mut sorted_real: Vec<f32> = real.clone();
    sorted_real.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((sorted_real[0] - 2.0).abs() < 1e-4);
    assert!((sorted_real[1] - 3.0).abs() < 1e-4);

    for &im in &imag {
        assert!(im.abs() < 1e-4, "Roots should be real");
    }
}

#[test]
fn test_polyroots_quadratic_complex() {
    let (client, device) = create_client();

    // x² + 2x + 5 = 0 → roots: -1 ± 2i
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 2.0, 1.0], &[3], &device);

    let roots = client.polyroots(&coeffs).unwrap();

    let real: Vec<f32> = roots.roots_real.to_vec();
    let imag: Vec<f32> = roots.roots_imag.to_vec();

    assert_eq!(real.len(), 2);

    // Both real parts should be -1
    for &r in &real {
        assert!(
            (r - (-1.0)).abs() < 1e-4,
            "Real part should be -1, got {}",
            r
        );
    }

    // Imaginary parts should be ±2
    let mut sorted_imag: Vec<f32> = imag.clone();
    sorted_imag.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((sorted_imag[0] - (-2.0)).abs() < 1e-4);
    assert!((sorted_imag[1] - 2.0).abs() < 1e-4);
}

#[test]
fn test_polyroots_cubic() {
    let (client, device) = create_client();

    // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
    // coeffs: [-6, 11, -6, 1]
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[-6.0f32, 11.0, -6.0, 1.0], &[4], &device);

    let roots = client.polyroots(&coeffs).unwrap();

    let real: Vec<f32> = roots.roots_real.to_vec();
    let imag: Vec<f32> = roots.roots_imag.to_vec();

    assert_eq!(real.len(), 3);

    let mut sorted_real: Vec<f32> = real.clone();
    sorted_real.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((sorted_real[0] - 1.0).abs() < 1e-3);
    assert!((sorted_real[1] - 2.0).abs() < 1e-3);
    assert!((sorted_real[2] - 3.0).abs() < 1e-3);

    for &im in &imag {
        assert!(im.abs() < 1e-3, "All roots should be real");
    }
}

#[test]
fn test_polyroots_constant() {
    let (client, device) = create_client();

    // p(x) = 5 (degree 0, no roots)
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device);

    let roots = client.polyroots(&coeffs).unwrap();

    let real: Vec<f32> = roots.roots_real.to_vec();
    assert!(real.is_empty(), "Constant polynomial has no roots");
}

// ============================================================================
// polyval Tests
// ============================================================================

#[test]
fn test_polyval_constant() {
    let (client, device) = create_client();

    // p(x) = 7
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[7.0f32], &[1], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 2.0, 100.0], &[4], &device);

    let result = client.polyval(&coeffs, &x).unwrap();
    let data: Vec<f32> = result.to_vec();

    for &v in &data {
        assert!((v - 7.0).abs() < 1e-6);
    }
}

#[test]
fn test_polyval_linear() {
    let (client, device) = create_client();

    // p(x) = 3 + 2x
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 2.0], &[2], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 5.0], &[3], &device);

    let result = client.polyval(&coeffs, &x).unwrap();
    let data: Vec<f32> = result.to_vec();

    assert!((data[0] - 3.0).abs() < 1e-6); // 3 + 2*0 = 3
    assert!((data[1] - 5.0).abs() < 1e-6); // 3 + 2*1 = 5
    assert!((data[2] - 13.0).abs() < 1e-6); // 3 + 2*5 = 13
}

#[test]
fn test_polyval_quadratic() {
    let (client, device) = create_client();

    // p(x) = 1 - x + x²
    // p(0) = 1, p(1) = 1, p(2) = 3, p(3) = 7
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -1.0, 1.0], &[3], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4], &device);

    let result = client.polyval(&coeffs, &x).unwrap();
    let data: Vec<f32> = result.to_vec();

    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
    assert!((data[2] - 3.0).abs() < 1e-5);
    assert!((data[3] - 7.0).abs() < 1e-5);
}

#[test]
fn test_polyval_at_roots() {
    let (client, device) = create_client();

    // p(x) = x² - 5x + 6 = (x-2)(x-3)
    // p(2) = 0, p(3) = 0
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[6.0f32, -5.0, 1.0], &[3], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device);

    let result = client.polyval(&coeffs, &x).unwrap();
    let data: Vec<f32> = result.to_vec();

    assert!(data[0].abs() < 1e-5, "p(2) should be 0");
    assert!(data[1].abs() < 1e-5, "p(3) should be 0");
}

// ============================================================================
// polyfromroots Tests
// ============================================================================

#[test]
fn test_polyfromroots_single() {
    let (client, device) = create_client();

    // Root: 3 → (x-3) = [-3, 1]
    let roots_real = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device);
    let roots_imag = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);

    let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
    let data: Vec<f32> = coeffs.to_vec();

    assert_eq!(data.len(), 2);
    assert!((data[0] - (-3.0)).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_polyfromroots_two_real() {
    let (client, device) = create_client();

    // Roots: 1, -2 → (x-1)(x+2) = x² + x - 2 = [-2, 1, 1]
    let roots_real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -2.0], &[2], &device);
    let roots_imag = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device);

    let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
    let data: Vec<f32> = coeffs.to_vec();

    assert_eq!(data.len(), 3);
    assert!((data[0] - (-2.0)).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
    assert!((data[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_polyfromroots_complex_pair() {
    let (client, device) = create_client();

    // Roots: 1+2i, 1-2i → (x - (1+2i))(x - (1-2i)) = x² - 2x + 5 = [5, -2, 1]
    let roots_real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
    let roots_imag = Tensor::<CpuRuntime>::from_slice(&[2.0f32, -2.0], &[2], &device);

    let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
    let data: Vec<f32> = coeffs.to_vec();

    assert_eq!(data.len(), 3);
    assert!(
        (data[0] - 5.0).abs() < 1e-5,
        "c0: expected 5, got {}",
        data[0]
    );
    assert!(
        (data[1] - (-2.0)).abs() < 1e-5,
        "c1: expected -2, got {}",
        data[1]
    );
    assert!(
        (data[2] - 1.0).abs() < 1e-5,
        "c2: expected 1, got {}",
        data[2]
    );
}

#[test]
fn test_polyfromroots_empty() {
    let (client, device) = create_client();

    // No roots → constant 1
    let roots_real = Tensor::<CpuRuntime>::from_slice(&[] as &[f32], &[0], &device);
    let roots_imag = Tensor::<CpuRuntime>::from_slice(&[] as &[f32], &[0], &device);

    let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();
    let data: Vec<f32> = coeffs.to_vec();

    assert_eq!(data.len(), 1);
    assert!((data[0] - 1.0).abs() < 1e-6);
}

// ============================================================================
// polymul Tests
// ============================================================================

#[test]
fn test_polymul_constants() {
    let (client, device) = create_client();

    // 3 * 4 = 12
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32], &[1], &device);

    let c = client.polymul(&a, &b).unwrap();
    let data: Vec<f32> = c.to_vec();

    assert_eq!(data.len(), 1);
    assert!((data[0] - 12.0).abs() < 1e-6);
}

#[test]
fn test_polymul_linear_constant() {
    let (client, device) = create_client();

    // (1 + x) * 2 = 2 + 2x = [2, 2]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

    let c = client.polymul(&a, &b).unwrap();
    let data: Vec<f32> = c.to_vec();

    assert_eq!(data.len(), 2);
    assert!((data[0] - 2.0).abs() < 1e-6);
    assert!((data[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_polymul_two_linear() {
    let (client, device) = create_client();

    // (1 + x)(2 + 3x) = 2 + 5x + 3x² = [2, 5, 3]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device);

    let c = client.polymul(&a, &b).unwrap();
    let data: Vec<f32> = c.to_vec();

    assert_eq!(data.len(), 3);
    assert!((data[0] - 2.0).abs() < 1e-6);
    assert!((data[1] - 5.0).abs() < 1e-6);
    assert!((data[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_polymul_quadratics() {
    let (client, device) = create_client();

    // (1 + x + x²)(1 - x + x²)
    // = 1 - x + x² + x - x² + x³ + x² - x³ + x⁴
    // = 1 + x² + x⁴ = [1, 0, 1, 0, 1]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -1.0, 1.0], &[3], &device);

    let c = client.polymul(&a, &b).unwrap();
    let data: Vec<f32> = c.to_vec();

    assert_eq!(data.len(), 5);
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!(data[1].abs() < 1e-5);
    assert!((data[2] - 1.0).abs() < 1e-5);
    assert!(data[3].abs() < 1e-5);
    assert!((data[4] - 1.0).abs() < 1e-5);
}

// ============================================================================
// Roundtrip Tests
// ============================================================================

#[test]
fn test_roundtrip_polyroots_polyfromroots_real() {
    let (client, device) = create_client();

    // Original monic polynomial: x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    let original = Tensor::<CpuRuntime>::from_slice(&[-6.0f32, 11.0, -6.0, 1.0], &[4], &device);

    // Find roots
    let roots = client.polyroots(&original).unwrap();

    // Reconstruct
    let reconstructed = client
        .polyfromroots(&roots.roots_real, &roots.roots_imag)
        .unwrap();
    let recon_data: Vec<f32> = reconstructed.to_vec();
    let orig_data: Vec<f32> = original.to_vec();

    assert_allclose(&recon_data, &orig_data, 1e-3, 1e-3, "roundtrip");
}

#[test]
fn test_roundtrip_polyroots_polyfromroots_complex() {
    let (client, device) = create_client();

    // x² + 4 = 0 has roots ±2i
    let original = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 0.0, 1.0], &[3], &device);

    let roots = client.polyroots(&original).unwrap();
    let reconstructed = client
        .polyfromroots(&roots.roots_real, &roots.roots_imag)
        .unwrap();

    let recon_data: Vec<f32> = reconstructed.to_vec();
    let orig_data: Vec<f32> = original.to_vec();

    assert_allclose(&recon_data, &orig_data, 1e-3, 1e-3, "complex roundtrip");
}

#[test]
fn test_polyval_at_computed_roots() {
    let (client, device) = create_client();

    // p(x) = x³ - 2x² - 5x + 6 = (x-1)(x+2)(x-3)
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[6.0f32, -5.0, -2.0, 1.0], &[4], &device);

    // Find roots
    let roots = client.polyroots(&coeffs).unwrap();

    // Evaluate polynomial at roots - should all be ~0
    let values = client.polyval(&coeffs, &roots.roots_real).unwrap();
    let data: Vec<f32> = values.to_vec();

    for (i, &v) in data.iter().enumerate() {
        assert!(v.abs() < 1e-3, "p(root_{}) should be ~0, got {}", i, v);
    }
}

// ============================================================================
// F64 Tests
// ============================================================================

#[test]
fn test_polyroots_f64() {
    let (client, device) = create_client();

    // x² - 3x + 2, roots: 1, 2
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[2.0f64, -3.0, 1.0], &[3], &device);

    let roots = client.polyroots(&coeffs).unwrap();

    let real: Vec<f64> = roots.roots_real.to_vec();
    let imag: Vec<f64> = roots.roots_imag.to_vec();

    let mut sorted_real: Vec<f64> = real.clone();
    sorted_real.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((sorted_real[0] - 1.0).abs() < 1e-12);
    assert!((sorted_real[1] - 2.0).abs() < 1e-12);

    for im in &imag {
        assert!(im.abs() < 1e-12);
    }
}

#[test]
fn test_polyval_f64() {
    let (client, device) = create_client();

    // p(x) = 1 + 2x + 3x², p(2) = 1 + 4 + 12 = 17
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device);

    let result = client.polyval(&coeffs, &x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!((data[0] - 17.0).abs() < 1e-14);
}

// ============================================================================
// CUDA Backend Tests
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use numr::runtime::cuda::CudaRuntime;

    #[test]
    fn test_cuda_polyroots() {
        let Some((client, device)) = common::create_cuda_client() else {
            println!("CUDA not available, skipping");
            return;
        };

        let coeffs = Tensor::<CudaRuntime>::from_slice(&[6.0f32, -5.0, 1.0], &[3], &device);
        let roots = client.polyroots(&coeffs).unwrap();

        let real: Vec<f32> = roots.roots_real.to_vec();
        let mut sorted: Vec<f32> = real.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 2.0).abs() < 1e-4);
        assert!((sorted[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_cuda_cpu_parity() {
        let Some((cuda_client, cuda_device)) = common::create_cuda_client() else {
            println!("CUDA not available, skipping");
            return;
        };

        let (cpu_client, cpu_device) = create_client();

        // Test polymul
        let a_cpu = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &cpu_device);
        let b_cpu = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &cpu_device);
        let cpu_result: Vec<f32> = cpu_client.polymul(&a_cpu, &b_cpu).unwrap().to_vec();

        let a_cuda = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0], &[2], &cuda_device);
        let b_cuda = Tensor::<CudaRuntime>::from_slice(&[3.0f32, 4.0], &[2], &cuda_device);
        let cuda_result: Vec<f32> = cuda_client.polymul(&a_cuda, &b_cuda).unwrap().to_vec();

        assert_allclose(&cpu_result, &cuda_result, 1e-5, 1e-5, "CPU/CUDA polymul");
    }
}

// ============================================================================
// WGPU Backend Tests
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu_tests {
    use super::*;
    use numr::runtime::wgpu::WgpuRuntime;

    #[test]
    fn test_wgpu_polyroots() {
        let Some((client, device)) = common::create_wgpu_client() else {
            println!("WGPU not available, skipping");
            return;
        };

        let coeffs = Tensor::<WgpuRuntime>::from_slice(&[6.0f32, -5.0, 1.0], &[3], &device);
        let roots = client.polyroots(&coeffs).unwrap();

        let real: Vec<f32> = roots.roots_real.to_vec();
        let mut sorted: Vec<f32> = real.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 2.0).abs() < 1e-4);
        assert!((sorted[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_wgpu_cpu_parity() {
        let Some((wgpu_client, wgpu_device)) = common::create_wgpu_client() else {
            println!("WGPU not available, skipping");
            return;
        };

        let (cpu_client, cpu_device) = create_client();

        // Test polyval
        let coeffs_cpu = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &cpu_device);
        let x_cpu = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.5, 2.5], &[3], &cpu_device);
        let cpu_result: Vec<f32> = cpu_client.polyval(&coeffs_cpu, &x_cpu).unwrap().to_vec();

        let coeffs_wgpu =
            Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &wgpu_device);
        let x_wgpu = Tensor::<WgpuRuntime>::from_slice(&[0.5f32, 1.5, 2.5], &[3], &wgpu_device);
        let wgpu_result: Vec<f32> = wgpu_client.polyval(&coeffs_wgpu, &x_wgpu).unwrap().to_vec();

        assert_allclose(&cpu_result, &wgpu_result, 1e-5, 1e-5, "CPU/WGPU polyval");
    }

    #[test]
    fn test_wgpu_f64_unsupported() {
        let Some((client, device)) = common::create_wgpu_client() else {
            println!("WGPU not available, skipping");
            return;
        };

        let coeffs = Tensor::<WgpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);
        assert!(client.polyroots(&coeffs).is_err());
    }
}

// ============================================================================
// Error Case Tests
//
// These tests verify proper error handling for invalid inputs.
// All polynomial operations should return errors (not panic) on invalid input.
// ============================================================================

#[test]
fn test_polyroots_empty_coeffs() {
    let (client, device) = create_client();

    // Empty coefficient tensor should error
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[] as &[f32], &[0], &device);
    assert!(
        client.polyroots(&coeffs).is_err(),
        "polyroots should error on empty coefficients"
    );
}

#[test]
fn test_polyval_dtype_mismatch() {
    let (client, device) = create_client();

    // Coefficients F32, x F64 - should error on dtype mismatch
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);

    assert!(
        client.polyval(&coeffs, &x).is_err(),
        "polyval should error on dtype mismatch"
    );
}

#[test]
fn test_polyfromroots_shape_mismatch() {
    let (client, device) = create_client();

    // Real and imag have different lengths - should error
    let roots_real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    let roots_imag = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[3], &device);

    assert!(
        client.polyfromroots(&roots_real, &roots_imag).is_err(),
        "polyfromroots should error on shape mismatch"
    );
}

#[test]
fn test_polyfromroots_dtype_mismatch() {
    let (client, device) = create_client();

    // Real F32, imag F64 - should error
    let roots_real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    let roots_imag = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.0], &[2], &device);

    assert!(
        client.polyfromroots(&roots_real, &roots_imag).is_err(),
        "polyfromroots should error on dtype mismatch"
    );
}

#[test]
fn test_polymul_dtype_mismatch() {
    let (client, device) = create_client();

    // a F32, b F64 - should error
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device);

    assert!(
        client.polymul(&a, &b).is_err(),
        "polymul should error on dtype mismatch"
    );
}

#[test]
fn test_polyval_2d_coeffs_error() {
    let (client, device) = create_client();

    // 2D coefficient tensor should error (must be 1D)
    let coeffs = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);

    assert!(
        client.polyval(&coeffs, &x).is_err(),
        "polyval should error on 2D coefficients"
    );
}

// ============================================================================
// Convolution Performance Tests
//
// These tests verify the optimized convolution implementation that
// auto-selects between direct O(n*m) and FFT O(n log n) algorithms.
// ============================================================================

#[test]
fn test_polymul_small_uses_direct() {
    let (client, device) = create_client();

    // 3 * 2 = 6 < 64 (threshold), should use direct convolution
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device);

    // (1 + 2x + 3x²) * (4 + 5x) = 4 + 13x + 22x² + 15x³
    let c = client.polymul(&a, &b).unwrap();
    let data: Vec<f32> = c.to_vec();

    assert_eq!(data.len(), 4);
    assert!((data[0] - 4.0).abs() < 1e-5);
    assert!((data[1] - 13.0).abs() < 1e-5);
    assert!((data[2] - 22.0).abs() < 1e-5);
    assert!((data[3] - 15.0).abs() < 1e-5);
}

#[test]
fn test_polymul_large_uses_fft() {
    let (client, device) = create_client();

    // 10 * 10 = 100 >= 64 (threshold), should use FFT convolution
    // Test with polynomial of all 1s: (1 + x + x² + ... + x⁹)
    let a_data: Vec<f32> = vec![1.0; 10];
    let b_data: Vec<f32> = vec![1.0; 10];

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[10], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[10], &device);

    let c = client.polymul(&a, &b).unwrap();
    let data: Vec<f32> = c.to_vec();

    // Result length: 10 + 10 - 1 = 19
    assert_eq!(data.len(), 19);

    // Coefficients of (1 + x + ... + x⁹)² are:
    // c[k] = min(k+1, 10, 19-k) for k in 0..19
    // This gives: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    let expected: Vec<f32> = (0..19)
        .map(|k| {
            let v = (k + 1).min(10).min(19 - k);
            v as f32
        })
        .collect();

    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "Coefficient {} differs: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_polymul_boundary_threshold() {
    let (client, device) = create_client();

    // Test at the boundary: 8 * 8 = 64, should use FFT
    let a_data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (1..=8).map(|i| i as f32).collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[8], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[8], &device);

    let c = client.polymul(&a, &b).unwrap();
    assert_eq!(c.shape()[0], 15); // 8 + 8 - 1

    // Verify first and last coefficients
    let data: Vec<f32> = c.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-4); // 1 * 1 = 1
    assert!((data[14] - 64.0).abs() < 1e-4); // 8 * 8 = 64
}

#[test]
fn test_polymul_very_large() {
    let (client, device) = create_client();

    // Test with large polynomials (50 coefficients each)
    // 50 * 50 = 2500 >> 64, definitely uses FFT
    let a_data: Vec<f32> = vec![1.0; 50];
    let b_data: Vec<f32> = vec![1.0; 50];

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[50], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[50], &device);

    let c = client.polymul(&a, &b).unwrap();

    // Result length: 50 + 50 - 1 = 99
    assert_eq!(c.shape()[0], 99);

    // Verify key coefficients
    let data: Vec<f32> = c.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-3); // First coefficient
    assert!((data[49] - 50.0).abs() < 1e-3); // Middle coefficient (peak)
    assert!((data[98] - 1.0).abs() < 1e-3); // Last coefficient
}

#[test]
fn test_polyfromroots_many_roots_uses_fft() {
    let (client, device) = create_client();

    // Create polynomial with many roots to test FFT convolution path
    // Each (x - rᵢ) factor multiplies by a degree-1 polynomial
    // With 20 roots, we have many convolutions, some will use FFT

    // Roots: 1, 2, 3, ..., 10 (real roots only for simplicity)
    let roots: Vec<f32> = (1..=10).map(|i| i as f32).collect();
    let roots_real = Tensor::<CpuRuntime>::from_slice(&roots, &[10], &device);
    let roots_imag = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 10], &[10], &device);

    let coeffs = client.polyfromroots(&roots_real, &roots_imag).unwrap();

    // Should have 11 coefficients (degree 10 polynomial)
    assert_eq!(coeffs.shape()[0], 11);

    // Verify it's monic (leading coefficient = 1)
    let data: Vec<f32> = coeffs.to_vec();
    assert!((data[10] - 1.0).abs() < 1e-4, "Should be monic polynomial");

    // Verify by evaluating at roots - should all be ~0
    let x_at_roots = Tensor::<CpuRuntime>::from_slice(&roots, &[10], &device);
    let values = client.polyval(&coeffs, &x_at_roots).unwrap();
    let values_data: Vec<f32> = values.to_vec();

    for (i, &v) in values_data.iter().enumerate() {
        assert!(v.abs() < 1e-2, "p(root_{}) should be ~0, got {}", i + 1, v);
    }
}
