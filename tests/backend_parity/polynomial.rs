// Backend parity tests migrated from tests/polynomial_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::algorithm::polynomial::PolynomialAlgorithms;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn assert_allclose(a: &[f32], b: &[f32], rtol: f32, atol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {}",
            msg,
            i,
            x,
            y
        );
    }
}

#[test]
fn test_polynomial_backend_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    #[cfg(feature = "cuda")]
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &cpu_device);
    #[cfg(feature = "cuda")]
    let b_cpu = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &cpu_device);
    #[cfg(feature = "cuda")]
    let cpu_polymul: Vec<f32> = cpu_client.polymul(&a_cpu, &b_cpu).unwrap().to_vec();

    #[cfg(feature = "wgpu")]
    let coeffs_cpu = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &cpu_device);
    #[cfg(feature = "wgpu")]
    let x_cpu = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.5, 2.5], &[3], &cpu_device);
    #[cfg(feature = "wgpu")]
    let cpu_polyval: Vec<f32> = cpu_client.polyval(&coeffs_cpu, &x_cpu).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a_cuda = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0],
            &[2],
            &cuda_device,
        );
        let b_cuda = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[3.0f32, 4.0],
            &[2],
            &cuda_device,
        );
        let cuda_polymul: Vec<f32> = cuda_client.polymul(&a_cuda, &b_cuda).unwrap().to_vec();
        assert_allclose(&cpu_polymul, &cuda_polymul, 1e-5, 1e-5, "CPU/CUDA polymul");

        let coeffs = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[6.0f32, -5.0, 1.0],
            &[3],
            &cuda_device,
        );
        let roots = cuda_client.polyroots(&coeffs).unwrap();
        let real: Vec<f32> = roots.roots_real.to_vec();
        let mut sorted: Vec<f32> = real.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 2.0).abs() < 1e-4);
        assert!((sorted[1] - 3.0).abs() < 1e-4);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let coeffs_wgpu = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0],
            &[3],
            &wgpu_device,
        );
        let x_wgpu = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[0.5f32, 1.5, 2.5],
            &[3],
            &wgpu_device,
        );
        let wgpu_polyval: Vec<f32> = wgpu_client.polyval(&coeffs_wgpu, &x_wgpu).unwrap().to_vec();
        assert_allclose(&cpu_polyval, &wgpu_polyval, 1e-5, 1e-5, "CPU/WGPU polyval");

        let coeffs = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[6.0f32, -5.0, 1.0],
            &[3],
            &wgpu_device,
        );
        let roots = wgpu_client.polyroots(&coeffs).unwrap();
        let real: Vec<f32> = roots.roots_real.to_vec();
        let mut sorted: Vec<f32> = real.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 2.0).abs() < 1e-4);
        assert!((sorted[1] - 3.0).abs() < 1e-4);

        let coeffs_f64 = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0],
            &[3],
            &wgpu_device,
        );
        assert!(wgpu_client.polyroots(&coeffs_f64).is_err());
    });
}
