//! FFT integration tests
//!
//! Tests for FFT operations including:
//! - Basic FFT/IFFT on complex inputs
//! - Real FFT (rfft) and inverse real FFT (irfft)
//! - 2D FFT operations
//! - Frequency shift operations
//! - Batched FFT operations
//! - Numerical accuracy verification

use numr::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};
use numr::dtype::{Complex64, Complex128, DType};
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn get_cpu_client() -> CpuClient {
    let device = CpuDevice::new();
    CpuClient::new(device)
}

// ============================================================================
// Basic FFT Tests
// ============================================================================

#[test]
fn test_fft_impulse_response() {
    // FFT of impulse [1, 0, 0, 0] = [1, 1, 1, 1]
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);
    let result = client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    let result_data: Vec<Complex64> = result.to_vec();
    for c in &result_data {
        assert!((c.re - 1.0).abs() < 1e-5, "Expected 1.0, got {}", c.re);
        assert!(c.im.abs() < 1e-5, "Expected 0.0i, got {}i", c.im);
    }
}

#[test]
fn test_fft_constant_signal() {
    // FFT of constant [1, 1, 1, 1] = [4, 0, 0, 0]
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);
    let result = client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    let result_data: Vec<Complex64> = result.to_vec();
    assert!((result_data[0].re - 4.0).abs() < 1e-5);
    assert!(result_data[0].im.abs() < 1e-5);
    for c in &result_data[1..] {
        assert!(c.re.abs() < 1e-5);
        assert!(c.im.abs() < 1e-5);
    }
}

#[test]
fn test_fft_ifft_roundtrip() {
    // FFT followed by IFFT should recover original signal
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);

    // Forward FFT
    let fft_result = client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    // Inverse FFT with normalization
    let ifft_result = client
        .fft(
            &fft_result,
            FftDirection::Inverse,
            FftNormalization::Backward,
        )
        .unwrap();

    let result_data: Vec<Complex64> = ifft_result.to_vec();
    for (i, (got, expected)) in result_data.iter().zip(input_data.iter()).enumerate() {
        assert!(
            (got.re - expected.re).abs() < 1e-4,
            "Real mismatch at {}: {} vs {}",
            i,
            got.re,
            expected.re
        );
        assert!(
            (got.im - expected.im).abs() < 1e-4,
            "Imag mismatch at {}: {} vs {}",
            i,
            got.im,
            expected.im
        );
    }
}

#[test]
fn test_fft_ortho_normalization() {
    // With ortho normalization, FFT and IFFT are both scaled by 1/sqrt(N)
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);

    // Forward FFT with ortho
    let fft_result = client
        .fft(&input, FftDirection::Forward, FftNormalization::Ortho)
        .unwrap();

    // Inverse FFT with ortho
    let ifft_result = client
        .fft(&fft_result, FftDirection::Inverse, FftNormalization::Ortho)
        .unwrap();

    let result_data: Vec<Complex64> = ifft_result.to_vec();
    for (got, expected) in result_data.iter().zip(input_data.iter()) {
        assert!((got.re - expected.re).abs() < 1e-4);
        assert!((got.im - expected.im).abs() < 1e-4);
    }
}

// ============================================================================
// Parseval's Theorem Tests
// ============================================================================

#[test]
fn test_fft_parseval_theorem() {
    // Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(1.5, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);
    let result = client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    let energy_time: f32 = input_data.iter().map(|c| c.re * c.re + c.im * c.im).sum();
    let result_data: Vec<Complex64> = result.to_vec();
    let energy_freq: f32 = result_data.iter().map(|c| c.re * c.re + c.im * c.im).sum();

    // energy_time = (1/N) * energy_freq, so energy_freq = N * energy_time
    let expected_freq_energy = energy_time * 4.0;
    assert!(
        (energy_freq - expected_freq_energy).abs() < 1e-3,
        "Parseval failed: {} vs {}",
        energy_freq,
        expected_freq_energy
    );
}

// ============================================================================
// Real FFT Tests
// ============================================================================

#[test]
fn test_rfft_basic() {
    // Real FFT of [1, 2, 3, 4]
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [1.0f32, 2.0, 3.0, 4.0];
    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);

    let result = client.rfft(&input, FftNormalization::None).unwrap();

    assert_eq!(result.shape(), &[3]); // N/2 + 1

    let result_data: Vec<Complex64> = result.to_vec();

    // Expected (from numpy.fft.rfft): [10+0j, -2+2j, -2+0j]
    assert!((result_data[0].re - 10.0).abs() < 1e-4);
    assert!(result_data[0].im.abs() < 1e-4);
    assert!((result_data[1].re - (-2.0)).abs() < 1e-4);
    assert!((result_data[1].im - 2.0).abs() < 1e-4);
    assert!((result_data[2].re - (-2.0)).abs() < 1e-4);
    assert!(result_data[2].im.abs() < 1e-4);
}

#[test]
fn test_rfft_irfft_roundtrip() {
    let client = get_cpu_client();
    let device = client.device().clone();

    let original_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Tensor::<CpuRuntime>::from_slice(&original_data, &[8], &device);

    // rfft -> irfft should recover original
    let rfft_result = client.rfft(&input, FftNormalization::None).unwrap();
    let irfft_result = client
        .irfft(&rfft_result, Some(8), FftNormalization::Backward)
        .unwrap();

    let result_data: Vec<f32> = irfft_result.to_vec();
    for (i, (got, expected)) in result_data.iter().zip(original_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "Mismatch at {}: {} vs {}",
            i,
            got,
            expected
        );
    }
}

#[test]
fn test_rfft_f64() {
    // Test with f64 precision
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [1.0f64, 2.0, 3.0, 4.0];
    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);

    let result = client.rfft(&input, FftNormalization::None).unwrap();

    assert_eq!(result.dtype(), DType::Complex128);
    assert_eq!(result.shape(), &[3]);

    let result_data: Vec<Complex128> = result.to_vec();
    assert!((result_data[0].re - 10.0).abs() < 1e-10);
}

// ============================================================================
// 2D FFT Tests
// ============================================================================

#[test]
fn test_fft2_basic() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // 2x2 complex input
    let input_data = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[2, 2], &device);
    let result = client
        .fft2(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    assert_eq!(result.shape(), &[2, 2]);

    // Verify ifft2 recovers original
    let recovered = client
        .fft2(&result, FftDirection::Inverse, FftNormalization::Backward)
        .unwrap();

    let recovered_data: Vec<Complex64> = recovered.contiguous().to_vec();
    for (got, expected) in recovered_data.iter().zip(input_data.iter()) {
        assert!((got.re - expected.re).abs() < 1e-4);
        assert!((got.im - expected.im).abs() < 1e-4);
    }
}

#[test]
fn test_rfft2_basic() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // 4x4 real input
    let input_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4, 4], &device);

    let result = client.rfft2(&input, FftNormalization::None).unwrap();

    // Output shape: [4, 3] since last dimension is N/2 + 1 = 3
    assert_eq!(result.shape(), &[4, 3]);
    assert_eq!(result.dtype(), DType::Complex64);
}

// ============================================================================
// Batched FFT Tests
// ============================================================================

#[test]
fn test_fft_batched() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // Batch of 3 signals, each with 4 elements
    let input_data = [
        // Signal 1: impulse
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        // Signal 2: constant
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        // Signal 3: alternating
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[3, 4], &device);
    let result = client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    assert_eq!(result.shape(), &[3, 4]);

    let result_data: Vec<Complex64> = result.to_vec();

    // Signal 1 (impulse): FFT should be [1, 1, 1, 1]
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        assert!((result_data[i].re - 1.0).abs() < 1e-5);
        assert!(result_data[i].im.abs() < 1e-5);
    }

    // Signal 2 (constant): FFT should be [4, 0, 0, 0]
    assert!((result_data[4].re - 4.0).abs() < 1e-5);
    #[allow(clippy::needless_range_loop)]
    for i in 5..8 {
        assert!(result_data[i].re.abs() < 1e-5);
    }

    // Signal 3 (alternating): FFT should be [0, 0, 4, 0]
    assert!(result_data[8].re.abs() < 1e-5);
    assert!(result_data[9].re.abs() < 1e-5);
    assert!((result_data[10].re - 4.0).abs() < 1e-5);
    assert!(result_data[11].re.abs() < 1e-5);
}

// ============================================================================
// FFT Shift Tests
// ============================================================================

#[test]
fn test_fftshift() {
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);
    let result = client.fftshift(&input).unwrap();

    let result_data: Vec<Complex64> = result.to_vec();
    // [0, 1, 2, 3] -> [2, 3, 0, 1]
    assert!((result_data[0].re - 2.0).abs() < 1e-5);
    assert!((result_data[1].re - 3.0).abs() < 1e-5);
    assert!((result_data[2].re - 0.0).abs() < 1e-5);
    assert!((result_data[3].re - 1.0).abs() < 1e-5);
}

#[test]
fn test_fftshift_ifftshift_roundtrip() {
    let client = get_cpu_client();
    let device = client.device().clone();

    let original_data = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&original_data, &[4], &device);
    let shifted = client.fftshift(&input).unwrap();
    let unshifted = client.ifftshift(&shifted).unwrap();

    let result_data: Vec<Complex64> = unshifted.to_vec();
    for (got, expected) in result_data.iter().zip(original_data.iter()) {
        assert!((got.re - expected.re).abs() < 1e-5);
        assert!((got.im - expected.im).abs() < 1e-5);
    }
}

// ============================================================================
// Frequency Generation Tests
// ============================================================================

#[test]
fn test_fftfreq() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // For N=8, d=1: [0, 1, 2, 3, -4, -3, -2, -1] / 8
    let freqs = client.fftfreq(8, 1.0, DType::F32, &device).unwrap();

    assert_eq!(freqs.shape(), &[8]);

    let freq_data: Vec<f32> = freqs.to_vec();
    let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];

    for (got, exp) in freq_data.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-6);
    }
}

#[test]
fn test_rfftfreq() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // For N=8, d=1: [0, 1, 2, 3, 4] / 8
    let freqs = client.rfftfreq(8, 1.0, DType::F32, &device).unwrap();

    assert_eq!(freqs.shape(), &[5]); // N/2 + 1

    let freq_data: Vec<f32> = freqs.to_vec();
    let expected = [0.0, 0.125, 0.25, 0.375, 0.5];

    for (got, exp) in freq_data.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-6);
    }
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_fft_various_sizes() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // Test various power-of-2 sizes
    for size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let input_data: Vec<Complex64> = (0..size).map(|i| Complex64::new(i as f32, 0.0)).collect();

        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[size], &device);
        let fft_result = client
            .fft(&input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let ifft_result = client
            .fft(
                &fft_result,
                FftDirection::Inverse,
                FftNormalization::Backward,
            )
            .unwrap();

        let result_data: Vec<Complex64> = ifft_result.to_vec();
        for (got, expected) in result_data.iter().zip(input_data.iter()) {
            assert!(
                (got.re - expected.re).abs() < 1e-3,
                "Size {}: real mismatch",
                size
            );
            assert!(
                (got.im - expected.im).abs() < 1e-3,
                "Size {}: imag mismatch",
                size
            );
        }
    }
}

#[test]
fn test_fft_non_power_of_2_error() {
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data: Vec<Complex64> = (0..7).map(|i| Complex64::new(i as f32, 0.0)).collect();

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[7], &device);
    let result = client.fft(&input, FftDirection::Forward, FftNormalization::None);

    assert!(result.is_err(), "FFT should reject non-power-of-2 sizes");
}

#[test]
fn test_fft_wrong_dtype_error() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // Try FFT on real input (should fail, use rfft instead)
    let input_data = [1.0f32, 2.0, 3.0, 4.0];
    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);

    let result = client.fft(&input, FftDirection::Forward, FftNormalization::None);
    assert!(result.is_err(), "FFT should reject non-complex input");
}

#[test]
fn test_rfft_wrong_dtype_error() {
    let client = get_cpu_client();
    let device = client.device().clone();

    // Try rfft on complex input (should fail)
    let input_data = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[2], &device);

    let result = client.rfft(&input, FftNormalization::None);
    assert!(result.is_err(), "rfft should reject complex input");
}

// ============================================================================
// Complex128 (f64) Tests
// ============================================================================

#[test]
fn test_fft_complex128() {
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex128::new(1.0, 0.0),
        Complex128::new(0.0, 0.0),
        Complex128::new(0.0, 0.0),
        Complex128::new(0.0, 0.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);
    let result = client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    let result_data: Vec<Complex128> = result.to_vec();
    for c in &result_data {
        assert!((c.re - 1.0).abs() < 1e-10);
        assert!(c.im.abs() < 1e-10);
    }
}

#[test]
fn test_fft_complex128_roundtrip() {
    let client = get_cpu_client();
    let device = client.device().clone();

    let input_data = [
        Complex128::new(1.0, 2.0),
        Complex128::new(3.0, 4.0),
        Complex128::new(5.0, 6.0),
        Complex128::new(7.0, 8.0),
    ];

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[4], &device);
    let fft_result = client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();
    let ifft_result = client
        .fft(
            &fft_result,
            FftDirection::Inverse,
            FftNormalization::Backward,
        )
        .unwrap();

    let result_data: Vec<Complex128> = ifft_result.to_vec();
    for (got, expected) in result_data.iter().zip(input_data.iter()) {
        assert!((got.re - expected.re).abs() < 1e-10);
        assert!((got.im - expected.im).abs() < 1e-10);
    }
}

// ============================================================================
// CUDA Backend Parity Tests
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_parity {
    use super::*;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    fn create_cuda_client() -> Option<(CudaClient, CudaDevice)> {
        // Try to create CUDA device 0, return None if CUDA is unavailable
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        Some((client, device))
    }

    #[test]
    fn test_fft_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Test with various sizes
        for size in [4, 8, 16, 64, 256, 512, 1024] {
            let input_data: Vec<Complex64> = (0..size)
                .map(|i| Complex64::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
                .collect();

            let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[size], &cpu_device);
            let cuda_input = Tensor::<CudaRuntime>::from_slice(&input_data, &[size], &cuda_device);

            // Forward FFT
            let cpu_result = cpu_client
                .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let cuda_result = cuda_client
                .fft(&cuda_input, FftDirection::Forward, FftNormalization::None)
                .unwrap();

            let cpu_data: Vec<Complex64> = cpu_result.to_vec();
            let cuda_data: Vec<Complex64> = cuda_result.to_vec();

            // Use relative tolerance for larger values (FFT accumulates errors)
            for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
                let max_mag = cpu_val.re.abs().max(cuda_val.re.abs()).max(1.0);
                let tol = 1e-4 * max_mag.max(1.0);
                assert!(
                    (cpu_val.re - cuda_val.re).abs() < tol,
                    "FFT size {}, idx {}: CPU re={}, CUDA re={}, diff={}",
                    size,
                    i,
                    cpu_val.re,
                    cuda_val.re,
                    (cpu_val.re - cuda_val.re).abs()
                );
                let max_mag_im = cpu_val.im.abs().max(cuda_val.im.abs()).max(1.0);
                let tol_im = 1e-4 * max_mag_im.max(1.0);
                assert!(
                    (cpu_val.im - cuda_val.im).abs() < tol_im,
                    "FFT size {}, idx {}: CPU im={}, CUDA im={}, diff={}",
                    size,
                    i,
                    cpu_val.im,
                    cuda_val.im,
                    (cpu_val.im - cuda_val.im).abs()
                );
            }
        }
    }

    #[test]
    fn test_fft_roundtrip_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<Complex64> = (0..64)
            .map(|i| Complex64::new(i as f32, -(i as f32) * 0.5))
            .collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[64], &cpu_device);
        let cuda_input = Tensor::<CudaRuntime>::from_slice(&input_data, &[64], &cuda_device);

        // Forward then inverse
        let cpu_fft = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cpu_result = cpu_client
            .fft(&cpu_fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();

        let cuda_fft = cuda_client
            .fft(&cuda_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cuda_result = cuda_client
            .fft(&cuda_fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let cuda_data: Vec<Complex64> = cuda_result.to_vec();

        for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - cuda_val.re).abs() < 1e-4,
                "Roundtrip idx {}: CPU re={}, CUDA re={}",
                i,
                cpu_val.re,
                cuda_val.re
            );
            assert!(
                (cpu_val.im - cuda_val.im).abs() < 1e-4,
                "Roundtrip idx {}: CPU im={}, CUDA im={}",
                i,
                cpu_val.im,
                cuda_val.im
            );
        }
    }

    #[test]
    fn test_rfft_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[64], &cpu_device);
        let cuda_input = Tensor::<CudaRuntime>::from_slice(&input_data, &[64], &cuda_device);

        let cpu_result = cpu_client.rfft(&cpu_input, FftNormalization::None).unwrap();
        let cuda_result = cuda_client
            .rfft(&cuda_input, FftNormalization::None)
            .unwrap();

        assert_eq!(cpu_result.shape(), cuda_result.shape());

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let cuda_data: Vec<Complex64> = cuda_result.to_vec();

        for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - cuda_val.re).abs() < 1e-4,
                "rfft idx {}: CPU re={}, CUDA re={}",
                i,
                cpu_val.re,
                cuda_val.re
            );
            assert!(
                (cpu_val.im - cuda_val.im).abs() < 1e-4,
                "rfft idx {}: CPU im={}, CUDA im={}",
                i,
                cpu_val.im,
                cuda_val.im
            );
        }
    }

    #[test]
    fn test_irfft_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Start with rfft output (Hermitian symmetric)
        let n = 64;
        let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

        let cpu_real = Tensor::<CpuRuntime>::from_slice(&input_data, &[n], &cpu_device);
        let cuda_real = Tensor::<CudaRuntime>::from_slice(&input_data, &[n], &cuda_device);

        let cpu_freq = cpu_client.rfft(&cpu_real, FftNormalization::None).unwrap();
        let cuda_freq = cuda_client
            .rfft(&cuda_real, FftNormalization::None)
            .unwrap();

        let cpu_result = cpu_client
            .irfft(&cpu_freq, Some(n), FftNormalization::Backward)
            .unwrap();
        let cuda_result = cuda_client
            .irfft(&cuda_freq, Some(n), FftNormalization::Backward)
            .unwrap();

        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let cuda_data: Vec<f32> = cuda_result.to_vec();

        for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            assert!(
                (cpu_val - cuda_val).abs() < 1e-4,
                "irfft idx {}: CPU={}, CUDA={}",
                i,
                cpu_val,
                cuda_val
            );
        }
    }

    #[test]
    fn test_fftshift_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new(i as f32, -i as f32))
            .collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[16], &cpu_device);
        let cuda_input = Tensor::<CudaRuntime>::from_slice(&input_data, &[16], &cuda_device);

        let cpu_result = cpu_client.fftshift(&cpu_input).unwrap();
        let cuda_result = cuda_client.fftshift(&cuda_input).unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let cuda_data: Vec<Complex64> = cuda_result.to_vec();

        for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - cuda_val.re).abs() < 1e-5,
                "fftshift idx {}: CPU re={}, CUDA re={}",
                i,
                cpu_val.re,
                cuda_val.re
            );
            assert!(
                (cpu_val.im - cuda_val.im).abs() < 1e-5,
                "fftshift idx {}: CPU im={}, CUDA im={}",
                i,
                cpu_val.im,
                cuda_val.im
            );
        }
    }

    #[test]
    fn test_fft_batched_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Batch of 4 signals, each with 32 elements
        let batch_size = 4;
        let fft_size = 32;
        let input_data: Vec<Complex64> = (0..(batch_size * fft_size))
            .map(|i| Complex64::new((i as f32 * 0.1).sin(), (i as f32 * 0.2).cos()))
            .collect();

        let cpu_input =
            Tensor::<CpuRuntime>::from_slice(&input_data, &[batch_size, fft_size], &cpu_device);
        let cuda_input =
            Tensor::<CudaRuntime>::from_slice(&input_data, &[batch_size, fft_size], &cuda_device);

        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cuda_result = cuda_client
            .fft(&cuda_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let cuda_data: Vec<Complex64> = cuda_result.to_vec();

        for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - cuda_val.re).abs() < 1e-4,
                "Batched FFT idx {}: CPU re={}, CUDA re={}",
                i,
                cpu_val.re,
                cuda_val.re
            );
            assert!(
                (cpu_val.im - cuda_val.im).abs() < 1e-4,
                "Batched FFT idx {}: CPU im={}, CUDA im={}",
                i,
                cpu_val.im,
                cuda_val.im
            );
        }
    }

    #[test]
    fn test_fft_ortho_normalization_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<Complex64> = (0..64).map(|i| Complex64::new(i as f32, 0.0)).collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[64], &cpu_device);
        let cuda_input = Tensor::<CudaRuntime>::from_slice(&input_data, &[64], &cuda_device);

        // Test ortho normalization
        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::Ortho)
            .unwrap();
        let cuda_result = cuda_client
            .fft(&cuda_input, FftDirection::Forward, FftNormalization::Ortho)
            .unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let cuda_data: Vec<Complex64> = cuda_result.to_vec();

        for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - cuda_val.re).abs() < 1e-4,
                "Ortho FFT idx {}: CPU re={}, CUDA re={}",
                i,
                cpu_val.re,
                cuda_val.re
            );
            assert!(
                (cpu_val.im - cuda_val.im).abs() < 1e-4,
                "Ortho FFT idx {}: CPU im={}, CUDA im={}",
                i,
                cpu_val.im,
                cuda_val.im
            );
        }
    }

    #[test]
    fn test_fft_large_cpu_cuda_parity() {
        let Some((cuda_client, cuda_device)) = create_cuda_client() else {
            println!("Skipping CUDA parity test: no CUDA device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Test large FFT that triggers multi-stage kernel (> 1024)
        let size = 2048;
        let input_data: Vec<Complex64> = (0..size)
            .map(|i| Complex64::new((i as f32 * 0.01).sin(), (i as f32 * 0.01).cos()))
            .collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[size], &cpu_device);
        let cuda_input = Tensor::<CudaRuntime>::from_slice(&input_data, &[size], &cuda_device);

        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cuda_result = cuda_client
            .fft(&cuda_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let cuda_data: Vec<Complex64> = cuda_result.to_vec();

        for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - cuda_val.re).abs() < 1e-3,
                "Large FFT idx {}: CPU re={}, CUDA re={}",
                i,
                cpu_val.re,
                cuda_val.re
            );
            assert!(
                (cpu_val.im - cuda_val.im).abs() < 1e-3,
                "Large FFT idx {}: CPU im={}, CUDA im={}",
                i,
                cpu_val.im,
                cuda_val.im
            );
        }
    }
}

// ============================================================================
// WebGPU Backend Parity Tests
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu_parity {
    use super::*;
    use numr::runtime::Runtime;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime, is_wgpu_available};

    fn create_wgpu_client() -> Option<(WgpuClient, WgpuDevice)> {
        // Check if WebGPU is available
        if !is_wgpu_available() {
            return None;
        }
        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        Some((client, device))
    }

    #[test]
    fn test_fft_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Test with various sizes (small FFT sizes for WebGPU shared memory kernel)
        for size in [4, 8, 16, 64, 128, 256] {
            let input_data: Vec<Complex64> = (0..size)
                .map(|i| Complex64::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
                .collect();

            let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[size], &cpu_device);
            let wgpu_input = Tensor::<WgpuRuntime>::from_slice(&input_data, &[size], &wgpu_device);

            // Forward FFT
            let cpu_result = cpu_client
                .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let wgpu_result = wgpu_client
                .fft(&wgpu_input, FftDirection::Forward, FftNormalization::None)
                .unwrap();

            let cpu_data: Vec<Complex64> = cpu_result.to_vec();
            let wgpu_data: Vec<Complex64> = wgpu_result.to_vec();

            // Use relative tolerance for larger values (FFT accumulates errors)
            for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
                let max_mag = cpu_val.re.abs().max(wgpu_val.re.abs()).max(1.0);
                let tol = 1e-4 * max_mag.max(1.0);
                assert!(
                    (cpu_val.re - wgpu_val.re).abs() < tol,
                    "FFT size {}, idx {}: CPU re={}, WGPU re={}, diff={}",
                    size,
                    i,
                    cpu_val.re,
                    wgpu_val.re,
                    (cpu_val.re - wgpu_val.re).abs()
                );
                let max_mag_im = cpu_val.im.abs().max(wgpu_val.im.abs()).max(1.0);
                let tol_im = 1e-4 * max_mag_im.max(1.0);
                assert!(
                    (cpu_val.im - wgpu_val.im).abs() < tol_im,
                    "FFT size {}, idx {}: CPU im={}, WGPU im={}, diff={}",
                    size,
                    i,
                    cpu_val.im,
                    wgpu_val.im,
                    (cpu_val.im - wgpu_val.im).abs()
                );
            }
        }
    }

    #[test]
    fn test_fft_roundtrip_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<Complex64> = (0..64)
            .map(|i| Complex64::new(i as f32, -(i as f32) * 0.5))
            .collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[64], &cpu_device);
        let wgpu_input = Tensor::<WgpuRuntime>::from_slice(&input_data, &[64], &wgpu_device);

        // Forward then inverse
        let cpu_fft = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cpu_result = cpu_client
            .fft(&cpu_fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();

        let wgpu_fft = wgpu_client
            .fft(&wgpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let wgpu_result = wgpu_client
            .fft(&wgpu_fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let wgpu_data: Vec<Complex64> = wgpu_result.to_vec();

        for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - wgpu_val.re).abs() < 1e-3,
                "Roundtrip idx {}: CPU re={}, WGPU re={}",
                i,
                cpu_val.re,
                wgpu_val.re
            );
            assert!(
                (cpu_val.im - wgpu_val.im).abs() < 1e-3,
                "Roundtrip idx {}: CPU im={}, WGPU im={}",
                i,
                cpu_val.im,
                wgpu_val.im
            );
        }
    }

    #[test]
    fn test_rfft_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Test with various sizes
        for n in [16, 32, 64, 128] {
            let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

            let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[n], &cpu_device);
            let wgpu_input = Tensor::<WgpuRuntime>::from_slice(&input_data, &[n], &wgpu_device);

            let cpu_result = cpu_client.rfft(&cpu_input, FftNormalization::None).unwrap();
            let wgpu_result = wgpu_client
                .rfft(&wgpu_input, FftNormalization::None)
                .unwrap();

            let cpu_data: Vec<Complex64> = cpu_result.to_vec();
            let wgpu_data: Vec<Complex64> = wgpu_result.to_vec();

            for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
                let max_re = cpu_val.re.abs().max(wgpu_val.re.abs()).max(1.0);
                let tol_re = 1e-4 * max_re;
                assert!(
                    (cpu_val.re - wgpu_val.re).abs() < tol_re,
                    "rfft n={} idx {}: CPU re={}, WGPU re={}",
                    n,
                    i,
                    cpu_val.re,
                    wgpu_val.re
                );
                let max_im = cpu_val.im.abs().max(wgpu_val.im.abs()).max(1.0);
                let tol_im = 1e-4 * max_im;
                assert!(
                    (cpu_val.im - wgpu_val.im).abs() < tol_im,
                    "rfft n={} idx {}: CPU im={}, WGPU im={}",
                    n,
                    i,
                    cpu_val.im,
                    wgpu_val.im
                );
            }
        }
    }

    #[test]
    fn test_irfft_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Start with rfft output (Hermitian symmetric)
        let n = 64;
        let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

        let cpu_real = Tensor::<CpuRuntime>::from_slice(&input_data, &[n], &cpu_device);
        let wgpu_real = Tensor::<WgpuRuntime>::from_slice(&input_data, &[n], &wgpu_device);

        let cpu_freq = cpu_client.rfft(&cpu_real, FftNormalization::None).unwrap();
        let wgpu_freq = wgpu_client
            .rfft(&wgpu_real, FftNormalization::None)
            .unwrap();

        let cpu_result = cpu_client
            .irfft(&cpu_freq, Some(n), FftNormalization::Backward)
            .unwrap();
        let wgpu_result = wgpu_client
            .irfft(&wgpu_freq, Some(n), FftNormalization::Backward)
            .unwrap();

        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let wgpu_data: Vec<f32> = wgpu_result.to_vec();

        for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
            assert!(
                (cpu_val - wgpu_val).abs() < 1e-4,
                "irfft idx {}: CPU={}, WGPU={}",
                i,
                cpu_val,
                wgpu_val
            );
        }
    }

    #[test]
    fn test_fftshift_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new(i as f32, -i as f32))
            .collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[16], &cpu_device);
        let wgpu_input = Tensor::<WgpuRuntime>::from_slice(&input_data, &[16], &wgpu_device);

        let cpu_result = cpu_client.fftshift(&cpu_input).unwrap();
        let wgpu_result = wgpu_client.fftshift(&wgpu_input).unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let wgpu_data: Vec<Complex64> = wgpu_result.to_vec();

        for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - wgpu_val.re).abs() < 1e-5,
                "fftshift idx {}: CPU re={}, WGPU re={}",
                i,
                cpu_val.re,
                wgpu_val.re
            );
            assert!(
                (cpu_val.im - wgpu_val.im).abs() < 1e-5,
                "fftshift idx {}: CPU im={}, WGPU im={}",
                i,
                cpu_val.im,
                wgpu_val.im
            );
        }
    }

    #[test]
    fn test_fft_batched_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        // Batch of 4 signals, each with 32 elements
        let batch_size = 4;
        let fft_size = 32;
        let input_data: Vec<Complex64> = (0..(batch_size * fft_size))
            .map(|i| Complex64::new((i as f32 * 0.1).sin(), (i as f32 * 0.2).cos()))
            .collect();

        let cpu_input =
            Tensor::<CpuRuntime>::from_slice(&input_data, &[batch_size, fft_size], &cpu_device);
        let wgpu_input =
            Tensor::<WgpuRuntime>::from_slice(&input_data, &[batch_size, fft_size], &wgpu_device);

        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let wgpu_result = wgpu_client
            .fft(&wgpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let wgpu_data: Vec<Complex64> = wgpu_result.to_vec();

        for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - wgpu_val.re).abs() < 1e-4,
                "Batched FFT idx {}: CPU re={}, WGPU re={}",
                i,
                cpu_val.re,
                wgpu_val.re
            );
            assert!(
                (cpu_val.im - wgpu_val.im).abs() < 1e-4,
                "Batched FFT idx {}: CPU im={}, WGPU im={}",
                i,
                cpu_val.im,
                wgpu_val.im
            );
        }
    }

    #[test]
    fn test_fft_ortho_normalization_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<Complex64> = (0..64).map(|i| Complex64::new(i as f32, 0.0)).collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[64], &cpu_device);
        let wgpu_input = Tensor::<WgpuRuntime>::from_slice(&input_data, &[64], &wgpu_device);

        // Test ortho normalization
        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::Ortho)
            .unwrap();
        let wgpu_result = wgpu_client
            .fft(&wgpu_input, FftDirection::Forward, FftNormalization::Ortho)
            .unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let wgpu_data: Vec<Complex64> = wgpu_result.to_vec();

        for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - wgpu_val.re).abs() < 1e-4,
                "Ortho FFT idx {}: CPU re={}, WGPU re={}",
                i,
                cpu_val.re,
                wgpu_val.re
            );
            assert!(
                (cpu_val.im - wgpu_val.im).abs() < 1e-4,
                "Ortho FFT idx {}: CPU im={}, WGPU im={}",
                i,
                cpu_val.im,
                wgpu_val.im
            );
        }
    }

    #[test]
    fn test_ifftshift_cpu_wgpu_parity() {
        let Some((wgpu_client, wgpu_device)) = create_wgpu_client() else {
            println!("Skipping WebGPU parity test: no WebGPU device available");
            return;
        };
        let cpu_client = get_cpu_client();
        let cpu_device = cpu_client.device().clone();

        let input_data: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new(i as f32, -i as f32))
            .collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[16], &cpu_device);
        let wgpu_input = Tensor::<WgpuRuntime>::from_slice(&input_data, &[16], &wgpu_device);

        let cpu_result = cpu_client.ifftshift(&cpu_input).unwrap();
        let wgpu_result = wgpu_client.ifftshift(&wgpu_input).unwrap();

        let cpu_data: Vec<Complex64> = cpu_result.to_vec();
        let wgpu_data: Vec<Complex64> = wgpu_result.to_vec();

        for (i, (cpu_val, wgpu_val)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
            assert!(
                (cpu_val.re - wgpu_val.re).abs() < 1e-5,
                "ifftshift idx {}: CPU re={}, WGPU re={}",
                i,
                cpu_val.re,
                wgpu_val.re
            );
            assert!(
                (cpu_val.im - wgpu_val.im).abs() < 1e-5,
                "ifftshift idx {}: CPU im={}, WGPU im={}",
                i,
                cpu_val.im,
                wgpu_val.im
            );
        }
    }
}
