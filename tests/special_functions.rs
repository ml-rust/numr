//! Integration tests for special mathematical functions
//!
//! Tests the SpecialFunctions trait across CPU backend.

use numr::ops::SpecialFunctions;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn get_client() -> <CpuRuntime as Runtime>::Client {
    let device = CpuDevice::new();
    CpuRuntime::default_client(&device)
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol || diff < tol * e.abs(),
            "Mismatch at index {}: actual={}, expected={}, diff={}",
            i,
            a,
            e,
            diff
        );
    }
}

// ============================================================================
// Error Function Tests
// ============================================================================

#[test]
fn test_erf_basic() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.5, 1.0, 2.0], &[4], &device);
    let result = client.erf(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // Expected values from standard mathematical tables
    let expected = [0.0, 0.5204998778, 0.8427007929, 0.9953222650];
    assert_close(&data, &expected, 1e-6);
}

#[test]
fn test_erf_negative() {
    let client = get_client();
    let device = CpuDevice::new();

    // erf is an odd function: erf(-x) = -erf(x)
    let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, -0.5, 0.5, 1.0], &[4], &device);
    let result = client.erf(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(
        (data[0] + data[3]).abs() < 1e-10,
        "erf(-1) should equal -erf(1)"
    );
    assert!(
        (data[1] + data[2]).abs() < 1e-10,
        "erf(-0.5) should equal -erf(0.5)"
    );
}

#[test]
fn test_erfc_basic() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0, 2.0], &[3], &device);
    let result = client.erfc(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // erfc(x) = 1 - erf(x)
    let expected = [1.0, 0.1572992070, 0.0046777350];
    assert_close(&data, &expected, 1e-6);
}

#[test]
fn test_erfinv_basic() {
    let client = get_client();
    let device = CpuDevice::new();

    // Test erfinv directly with known values
    // erfinv(0) = 0
    // erfinv(0.5) ≈ 0.4769
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.5, -0.5], &[3], &device);
    let result = client.erfinv(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].abs() < 1e-10, "erfinv(0) should be 0");
    assert!(
        (data[1] - 0.4769362762).abs() < 1e-3,
        "erfinv(0.5) mismatch"
    );
    assert!(
        (data[2] + 0.4769362762).abs() < 1e-3,
        "erfinv(-0.5) should be -erfinv(0.5)"
    );
}

// ============================================================================
// Gamma Function Tests
// ============================================================================

#[test]
fn test_gamma_integers() {
    let client = get_client();
    let device = CpuDevice::new();

    // Gamma(n) = (n-1)! for positive integers
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let result = client.gamma(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // Expected: 0!, 1!, 2!, 3!, 4!
    let expected = [1.0, 1.0, 2.0, 6.0, 24.0];
    assert_close(&data, &expected, 1e-10);
}

#[test]
fn test_gamma_half() {
    let client = get_client();
    let device = CpuDevice::new();

    // Gamma(1/2) = sqrt(pi)
    let x = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1], &device);
    let result = client.gamma(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    let sqrt_pi = std::f64::consts::PI.sqrt();
    assert!(
        (data[0] - sqrt_pi).abs() < 1e-10,
        "Gamma(1/2) should equal sqrt(pi)"
    );
}

#[test]
fn test_lgamma_large() {
    let client = get_client();
    let device = CpuDevice::new();

    // lgamma avoids overflow for large values
    let x = Tensor::<CpuRuntime>::from_slice(&[100.0f64, 150.0, 170.0], &[3], &device);
    let result = client.lgamma(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // All values should be finite
    assert!(
        data.iter().all(|&v| v.is_finite()),
        "lgamma should be finite for large inputs"
    );
    // Values should be positive and increasing
    assert!(
        data[0] < data[1] && data[1] < data[2],
        "lgamma should be increasing"
    );
}

#[test]
fn test_digamma_positive() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 5.0, 10.0], &[4], &device);
    let result = client.digamma(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // psi(1) = -gamma (Euler-Mascheroni constant)
    let euler_gamma = 0.5772156649015329;
    assert!(
        (data[0] + euler_gamma).abs() < 1e-6,
        "digamma(1) should equal -gamma"
    );

    // Digamma is increasing for x > 0
    assert!(data[1] > data[0], "digamma should be increasing");
    assert!(data[2] > data[1], "digamma should be increasing");
    assert!(data[3] > data[2], "digamma should be increasing");
}

// ============================================================================
// Beta Function Tests
// ============================================================================

#[test]
fn test_beta_symmetry() {
    let client = get_client();
    let device = CpuDevice::new();

    // B(a,b) = B(b,a)
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 3.0, 4.0], &[3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 2.0, 5.0], &[3], &device);

    let result1 = client.beta(&a, &b).unwrap();
    let result2 = client.beta(&b, &a).unwrap();

    let data1: Vec<f64> = result1.to_vec();
    let data2: Vec<f64> = result2.to_vec();

    assert_close(&data1, &data2, 1e-10);
}

#[test]
fn test_beta_integers() {
    let client = get_client();
    let device = CpuDevice::new();

    // B(a,b) = (a-1)!(b-1)!/(a+b-1)! for positive integers
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 3.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 4.0], &[2], &device);

    let result = client.beta(&a, &b).unwrap();
    let data: Vec<f64> = result.to_vec();

    // B(2,3) = 1!*2!/4! = 2/24 = 1/12
    // B(3,4) = 2!*3!/6! = 12/720 = 1/60
    let expected = [1.0 / 12.0, 1.0 / 60.0];
    assert_close(&data, &expected, 1e-10);
}

// ============================================================================
// Incomplete Functions Tests
// ============================================================================

#[test]
fn test_betainc_bounds() {
    let client = get_client();
    let device = CpuDevice::new();

    // I_0(a,b) = 0, I_1(a,b) = 1
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 2.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 3.0], &[2], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0], &[2], &device);

    let result = client.betainc(&a, &b, &x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].abs() < 1e-10, "betainc(a,b,0) should be 0");
    assert!((data[1] - 1.0).abs() < 1e-10, "betainc(a,b,1) should be 1");
}

#[test]
fn test_gammainc_bounds() {
    let client = get_client();
    let device = CpuDevice::new();

    // P(a, 0) = 0
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 5.0], &[2], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.0], &[2], &device);

    let result = client.gammainc(&a, &x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].abs() < 1e-10, "gammainc(a, 0) should be 0");
    assert!(data[1].abs() < 1e-10, "gammainc(a, 0) should be 0");
}

#[test]
fn test_gammaincc_complement() {
    let client = get_client();
    let device = CpuDevice::new();

    // P(a,x) + Q(a,x) = 1
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 3.0, 5.0], &[3], &device);
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

    let p = client.gammainc(&a, &x).unwrap();
    let q = client.gammaincc(&a, &x).unwrap();

    let p_data: Vec<f64> = p.to_vec();
    let q_data: Vec<f64> = q.to_vec();

    for i in 0..3 {
        assert!(
            (p_data[i] + q_data[i] - 1.0).abs() < 1e-10,
            "P(a,x) + Q(a,x) should equal 1, got {} + {} = {}",
            p_data[i],
            q_data[i],
            p_data[i] + q_data[i]
        );
    }
}

// ============================================================================
// F32 Type Tests
// ============================================================================

#[test]
fn test_erf_f32() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.5, 1.0, 2.0], &[4], &device);
    let result = client.erf(&x).unwrap();
    let data: Vec<f32> = result.to_vec();

    // F32 has lower precision
    let expected = [0.0f32, 0.5205, 0.8427, 0.9953];
    for (a, e) in data.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-3, "F32 erf mismatch: {} vs {}", a, e);
    }
}

#[test]
fn test_gamma_f32() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let result = client.gamma(&x).unwrap();
    let data: Vec<f32> = result.to_vec();

    let expected = [1.0f32, 1.0, 2.0, 6.0];
    for (a, e) in data.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-5, "F32 gamma mismatch: {} vs {}", a, e);
    }
}

// ============================================================================
// Multi-dimensional Tests
// ============================================================================

#[test]
fn test_erf_multidim() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.5, 1.0, 1.5, 2.0, 2.5], &[2, 3], &device);
    let result = client.erf(&x).unwrap();

    assert_eq!(result.shape(), &[2, 3], "Shape should be preserved");

    let data: Vec<f64> = result.to_vec();
    assert!(data[0].abs() < 1e-10, "erf(0) should be 0");
    assert!(data[4] > 0.99, "erf(2.0) should be close to 1");
}
