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

// ============================================================================
// Bessel Function Tests - First Kind (J0, J1)
// ============================================================================

#[test]
fn test_bessel_j0_at_zero() {
    let client = get_client();
    let device = CpuDevice::new();

    // J0(0) = 1
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
    let result = client.bessel_j0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(
        (data[0] - 1.0).abs() < 1e-7,
        "J0(0) should equal 1, got {}",
        data[0]
    );
}

#[test]
fn test_bessel_j0_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    // Known values from tables
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let result = client.bessel_j0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // J0 values from mathematical tables
    let expected = [
        0.7651976866,
        0.2238907791,
        -0.2600519549,
        -0.3971498099,
        -0.1775967713,
    ];
    assert_close(&data, &expected, 1e-5);
}

#[test]
fn test_bessel_j0_large_args() {
    let client = get_client();
    let device = CpuDevice::new();

    // Test asymptotic region
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f64, 20.0, 50.0], &[3], &device);
    let result = client.bessel_j0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // J0(10) ≈ -0.2459, J0 oscillates with decreasing amplitude
    assert!(data[0].abs() < 0.3, "J0(10) should have small amplitude");
    assert!(data[1].abs() < 0.2, "J0(20) should have smaller amplitude");
    assert!(
        data[2].abs() < 0.1,
        "J0(50) should have even smaller amplitude"
    );
}

#[test]
fn test_bessel_j0_even() {
    let client = get_client();
    let device = CpuDevice::new();

    // J0 is an even function: J0(-x) = J0(x)
    let x_pos = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.5, 5.0], &[3], &device);
    let x_neg = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, -2.5, -5.0], &[3], &device);

    let result_pos = client.bessel_j0(&x_pos).unwrap();
    let result_neg = client.bessel_j0(&x_neg).unwrap();

    let data_pos: Vec<f64> = result_pos.to_vec();
    let data_neg: Vec<f64> = result_neg.to_vec();

    assert_close(&data_pos, &data_neg, 1e-10);
}

#[test]
fn test_bessel_j1_at_zero() {
    let client = get_client();
    let device = CpuDevice::new();

    // J1(0) = 0
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
    let result = client.bessel_j1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(
        data[0].abs() < 1e-10,
        "J1(0) should equal 0, got {}",
        data[0]
    );
}

#[test]
fn test_bessel_j1_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let result = client.bessel_j1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // J1 values from tables
    let expected = [
        0.4400505857,
        0.5767248078,
        0.3390589585,
        -0.0660433280,
        -0.3275791376,
    ];
    assert_close(&data, &expected, 1e-5);
}

#[test]
fn test_bessel_j1_odd() {
    let client = get_client();
    let device = CpuDevice::new();

    // J1 is an odd function: J1(-x) = -J1(x)
    let x_pos = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.5, 5.0], &[3], &device);
    let x_neg = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, -2.5, -5.0], &[3], &device);

    let result_pos = client.bessel_j1(&x_pos).unwrap();
    let result_neg = client.bessel_j1(&x_neg).unwrap();

    let data_pos: Vec<f64> = result_pos.to_vec();
    let data_neg: Vec<f64> = result_neg.to_vec();

    for i in 0..3 {
        assert!(
            (data_pos[i] + data_neg[i]).abs() < 1e-10,
            "J1(-x) should equal -J1(x)"
        );
    }
}

// ============================================================================
// Bessel Function Tests - Second Kind (Y0, Y1)
// ============================================================================

#[test]
fn test_bessel_y0_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 5.0], &[4], &device);
    let result = client.bessel_y0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // Y0 values from tables
    let expected = [0.0882569642, 0.5103756726, 0.3768500100, -0.3085176252];
    assert_close(&data, &expected, 1e-5);
}

#[test]
fn test_bessel_y0_negative_returns_nan() {
    let client = get_client();
    let device = CpuDevice::new();

    // Y0(x) is undefined for x <= 0
    let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, 0.0], &[2], &device);
    let result = client.bessel_y0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].is_nan(), "Y0(-1) should be NaN");
    assert!(data[1].is_nan(), "Y0(0) should be NaN");
}

#[test]
fn test_bessel_y1_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 5.0], &[4], &device);
    let result = client.bessel_y1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // Y1 values from tables
    let expected = [-0.7812128213, -0.1070324315, 0.3246744248, 0.1478631434];
    assert_close(&data, &expected, 1e-5);
}

#[test]
fn test_bessel_y1_negative_returns_nan() {
    let client = get_client();
    let device = CpuDevice::new();

    // Y1(x) is undefined for x <= 0
    let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, 0.0], &[2], &device);
    let result = client.bessel_y1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].is_nan(), "Y1(-1) should be NaN");
    assert!(data[1].is_nan(), "Y1(0) should be NaN");
}

// ============================================================================
// Modified Bessel Function Tests - First Kind (I0, I1)
// ============================================================================

#[test]
fn test_bessel_i0_at_zero() {
    let client = get_client();
    let device = CpuDevice::new();

    // I0(0) = 1
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
    let result = client.bessel_i0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(
        (data[0] - 1.0).abs() < 1e-10,
        "I0(0) should equal 1, got {}",
        data[0]
    );
}

#[test]
fn test_bessel_i0_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 5.0], &[4], &device);
    let result = client.bessel_i0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // I0 values from tables (I0 grows exponentially)
    let expected = [1.2660658778, 2.2795853023, 4.8807925858, 27.2398718236];
    assert_close(&data, &expected, 1e-4);
}

#[test]
fn test_bessel_i0_even() {
    let client = get_client();
    let device = CpuDevice::new();

    // I0 is an even function: I0(-x) = I0(x)
    let x_pos = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.5, 5.0], &[3], &device);
    let x_neg = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, -2.5, -5.0], &[3], &device);

    let result_pos = client.bessel_i0(&x_pos).unwrap();
    let result_neg = client.bessel_i0(&x_neg).unwrap();

    let data_pos: Vec<f64> = result_pos.to_vec();
    let data_neg: Vec<f64> = result_neg.to_vec();

    assert_close(&data_pos, &data_neg, 1e-10);
}

#[test]
fn test_bessel_i0_positive() {
    let client = get_client();
    let device = CpuDevice::new();

    // I0(x) > 0 for all x
    let x = Tensor::<CpuRuntime>::from_slice(&[-5.0f64, -2.0, 0.0, 2.0, 5.0], &[5], &device);
    let result = client.bessel_i0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    for &val in &data {
        assert!(val > 0.0, "I0(x) should be positive, got {}", val);
    }
}

#[test]
fn test_bessel_i1_at_zero() {
    let client = get_client();
    let device = CpuDevice::new();

    // I1(0) = 0
    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
    let result = client.bessel_i1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(
        data[0].abs() < 1e-10,
        "I1(0) should equal 0, got {}",
        data[0]
    );
}

#[test]
fn test_bessel_i1_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 5.0], &[4], &device);
    let result = client.bessel_i1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // I1 values from tables
    let expected = [0.5651591040, 1.5906368547, 3.9533702174, 24.3356421088];
    assert_close(&data, &expected, 1e-4);
}

#[test]
fn test_bessel_i1_odd() {
    let client = get_client();
    let device = CpuDevice::new();

    // I1 is an odd function: I1(-x) = -I1(x)
    let x_pos = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.5, 5.0], &[3], &device);
    let x_neg = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, -2.5, -5.0], &[3], &device);

    let result_pos = client.bessel_i1(&x_pos).unwrap();
    let result_neg = client.bessel_i1(&x_neg).unwrap();

    let data_pos: Vec<f64> = result_pos.to_vec();
    let data_neg: Vec<f64> = result_neg.to_vec();

    for i in 0..3 {
        assert!(
            (data_pos[i] + data_neg[i]).abs() < 1e-10,
            "I1(-x) should equal -I1(x)"
        );
    }
}

// ============================================================================
// Modified Bessel Function Tests - Second Kind (K0, K1)
// ============================================================================

#[test]
fn test_bessel_k0_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 1.0, 2.0, 3.0], &[4], &device);
    let result = client.bessel_k0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // K0 values from tables (K0 decays exponentially)
    let expected = [0.9244190936, 0.4210244382, 0.1138938727, 0.0347395045];
    assert_close(&data, &expected, 1e-4);
}

#[test]
fn test_bessel_k0_negative_returns_nan() {
    let client = get_client();
    let device = CpuDevice::new();

    // K0(x) is undefined for x <= 0
    let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, 0.0], &[2], &device);
    let result = client.bessel_k0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].is_nan(), "K0(-1) should be NaN");
    assert!(data[1].is_nan(), "K0(0) should be NaN");
}

#[test]
fn test_bessel_k0_positive() {
    let client = get_client();
    let device = CpuDevice::new();

    // K0(x) > 0 for x > 0
    let x = Tensor::<CpuRuntime>::from_slice(&[0.1f64, 0.5, 1.0, 2.0, 5.0], &[5], &device);
    let result = client.bessel_k0(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    for &val in &data {
        assert!(val > 0.0, "K0(x) should be positive for x > 0, got {}", val);
    }
}

#[test]
fn test_bessel_k1_small_args() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 1.0, 2.0, 3.0], &[4], &device);
    let result = client.bessel_k1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    // K1 values from tables
    let expected = [1.6564411200, 0.6019072302, 0.1398658818, 0.0401564199];
    assert_close(&data, &expected, 1e-4);
}

#[test]
fn test_bessel_k1_negative_returns_nan() {
    let client = get_client();
    let device = CpuDevice::new();

    // K1(x) is undefined for x <= 0
    let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, 0.0], &[2], &device);
    let result = client.bessel_k1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].is_nan(), "K1(-1) should be NaN");
    assert!(data[1].is_nan(), "K1(0) should be NaN");
}

#[test]
fn test_bessel_k1_positive() {
    let client = get_client();
    let device = CpuDevice::new();

    // K1(x) > 0 for x > 0
    let x = Tensor::<CpuRuntime>::from_slice(&[0.1f64, 0.5, 1.0, 2.0, 5.0], &[5], &device);
    let result = client.bessel_k1(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    for &val in &data {
        assert!(val > 0.0, "K1(x) should be positive for x > 0, got {}", val);
    }
}

// ============================================================================
// Bessel Function F32 Tests
// ============================================================================

#[test]
fn test_bessel_j0_f32() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 2.0, 5.0], &[4], &device);
    let result = client.bessel_j0(&x).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Lower precision for F32
    let expected = [1.0f32, 0.7652, 0.2239, -0.1776];
    for (a, e) in data.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-3, "F32 J0 mismatch: {} vs {}", a, e);
    }
}

#[test]
fn test_bessel_i0_f32() {
    let client = get_client();
    let device = CpuDevice::new();

    let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4], &device);
    let result = client.bessel_i0(&x).unwrap();
    let data: Vec<f32> = result.to_vec();

    let expected = [1.0f32, 1.266, 2.280, 4.881];
    for (a, e) in data.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-2, "F32 I0 mismatch: {} vs {}", a, e);
    }
}

#[test]
fn test_erfinv_negative() {
    let client = get_client();
    let device = CpuDevice::new();

    // Test erfinv for negative values
    let x = Tensor::<CpuRuntime>::from_slice(&[-0.8f64], &[1], &device);
    let result = client.erfinv(&x).unwrap();
    let data: Vec<f64> = result.to_vec();

    let erfinv_n08 = data[0];
    assert!(erfinv_n08.is_finite(), "erfinv(-0.8) = {}", erfinv_n08);
    // erfinv(-0.8) ≈ -0.9062
    assert!(
        (erfinv_n08 + 0.9062).abs() < 0.01,
        "erfinv(-0.8) = {}, expected ≈ -0.9062",
        erfinv_n08
    );
}

// ============================================================================
// Inverse Incomplete Function Tests
// ============================================================================

#[test]
fn test_gammaincinv_roundtrip() {
    let client = get_client();
    let device = CpuDevice::new();

    // Test gammaincinv: gammainc(a, gammaincinv(a, p)) ≈ p
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 5.0, 10.0], &[4], &device);
    let p = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.3, 0.7, 0.9], &[4], &device);

    let x = client.gammaincinv(&a, &p).unwrap();
    let back = client.gammainc(&a, &x).unwrap();

    let p_data: Vec<f64> = p.to_vec();
    let back_data: Vec<f64> = back.to_vec();

    for i in 0..4 {
        assert!(
            (back_data[i] - p_data[i]).abs() < 1e-6,
            "gammaincinv roundtrip failed at {}: {} vs {}",
            i,
            back_data[i],
            p_data[i]
        );
    }
}

#[test]
fn test_gammaincinv_bounds() {
    let client = get_client();
    let device = CpuDevice::new();

    // gammaincinv(a, 0) = 0
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 5.0], &[2], &device);
    let p = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.0], &[2], &device);

    let result = client.gammaincinv(&a, &p).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].abs() < 1e-10, "gammaincinv(a, 0) should be 0");
    assert!(data[1].abs() < 1e-10, "gammaincinv(a, 0) should be 0");
}

#[test]
fn test_betaincinv_roundtrip() {
    let client = get_client();
    let device = CpuDevice::new();

    // Test betaincinv: betainc(a, b, betaincinv(a, b, p)) ≈ p
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 5.0, 0.5, 10.0], &[4], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 2.0, 5.0, 10.0], &[4], &device);
    let p = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.6, 0.2, 0.5], &[4], &device);

    let x = client.betaincinv(&a, &b, &p).unwrap();
    let back = client.betainc(&a, &b, &x).unwrap();

    let p_data: Vec<f64> = p.to_vec();
    let back_data: Vec<f64> = back.to_vec();

    for i in 0..4 {
        assert!(
            (back_data[i] - p_data[i]).abs() < 1e-6,
            "betaincinv roundtrip failed at {}: {} vs {}",
            i,
            back_data[i],
            p_data[i]
        );
    }
}

#[test]
fn test_betaincinv_bounds() {
    let client = get_client();
    let device = CpuDevice::new();

    // betaincinv(a, b, 0) = 0, betaincinv(a, b, 1) = 1
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 2.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 3.0], &[2], &device);
    let p = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0], &[2], &device);

    let result = client.betaincinv(&a, &b, &p).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert!(data[0].abs() < 1e-10, "betaincinv(a, b, 0) should be 0");
    assert!(
        (data[1] - 1.0).abs() < 1e-10,
        "betaincinv(a, b, 1) should be 1"
    );
}
