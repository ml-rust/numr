// Backend parity tests migrated from tests/fft_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};
use numr::dtype::Complex64;
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn get_cpu_client() -> CpuClient {
    let device = CpuDevice::new();
    CpuClient::new(device)
}

fn assert_complex_close(cpu: &[Complex64], other: &[Complex64], tol: f32, label: &str) {
    assert_eq!(cpu.len(), other.len(), "{} length mismatch", label);
    for (i, (c, g)) in cpu.iter().zip(other.iter()).enumerate() {
        assert!((c.re - g.re).abs() < tol, "{} re idx {}", label, i);
        assert!((c.im - g.im).abs() < tol, "{} im idx {}", label, i);
    }
}

#[test]
fn test_fft_forward_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    for size in [4, 8, 16, 64, 128, 256] {
        let input_data: Vec<Complex64> = (0..size)
            .map(|i| Complex64::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
            .collect();

        let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[size], &cpu_device);
        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cpu_data: Vec<Complex64> = cpu_result.to_vec();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &input_data,
                &[size],
                &cuda_device,
            );
            let result = cuda_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let data: Vec<Complex64> = result.to_vec();
            assert_complex_close(&cpu_data, &data, 1e-4, "fft cuda");
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
                &input_data,
                &[size],
                &wgpu_device,
            );
            let result = wgpu_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let data: Vec<Complex64> = result.to_vec();
            assert_complex_close(&cpu_data, &data, 1e-4, "fft wgpu");
        });
    }
}

#[test]
fn test_fft_roundtrip_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    let input_data: Vec<Complex64> = (0..64)
        .map(|i| Complex64::new(i as f32, -(i as f32) * 0.5))
        .collect();

    let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[64], &cpu_device);
    let cpu_fft = cpu_client
        .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
        .unwrap();
    let cpu_result = cpu_client
        .fft(&cpu_fft, FftDirection::Inverse, FftNormalization::Backward)
        .unwrap();
    let cpu_data: Vec<Complex64> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input_data,
            &[64],
            &cuda_device,
        );
        let fft = cuda_client
            .fft(&input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let result = cuda_client
            .fft(&fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-4, "roundtrip cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &input_data,
            &[64],
            &wgpu_device,
        );
        let fft = wgpu_client
            .fft(&input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let result = wgpu_client
            .fft(&fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-3, "roundtrip wgpu");
    });
}

#[test]
fn test_rfft_irfft_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();
    let n = 64;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

    let cpu_real = Tensor::<CpuRuntime>::from_slice(&input_data, &[n], &cpu_device);
    let cpu_freq = cpu_client.rfft(&cpu_real, FftNormalization::None).unwrap();
    let cpu_ir = cpu_client
        .irfft(&cpu_freq, Some(n), FftNormalization::Backward)
        .unwrap();
    let cpu_ir_data: Vec<f32> = cpu_ir.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let real =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&input_data, &[n], &cuda_device);
        let freq = cuda_client.rfft(&real, FftNormalization::None).unwrap();
        let ir = cuda_client
            .irfft(&freq, Some(n), FftNormalization::Backward)
            .unwrap();
        let data: Vec<f32> = ir.to_vec();
        for (c, g) in cpu_ir_data.iter().zip(data.iter()) {
            assert!((c - g).abs() < 1e-4);
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let real =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&input_data, &[n], &wgpu_device);
        let freq = wgpu_client.rfft(&real, FftNormalization::None).unwrap();
        let ir = wgpu_client
            .irfft(&freq, Some(n), FftNormalization::Backward)
            .unwrap();
        let data: Vec<f32> = ir.to_vec();
        for (c, g) in cpu_ir_data.iter().zip(data.iter()) {
            assert!((c - g).abs() < 1e-4);
        }
    });
}

#[test]
fn test_fftshift_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    let input_data: Vec<Complex64> = (0..16)
        .map(|i| Complex64::new(i as f32, -i as f32))
        .collect();
    let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[16], &cpu_device);
    let cpu_result = cpu_client.fftshift(&cpu_input).unwrap();
    let cpu_data: Vec<Complex64> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input_data,
            &[16],
            &cuda_device,
        );
        let result = cuda_client.fftshift(&input).unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-5, "fftshift cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &input_data,
            &[16],
            &wgpu_device,
        );
        let result = wgpu_client.fftshift(&input).unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-5, "fftshift wgpu");
    });
}
