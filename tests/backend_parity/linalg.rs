// Backend parity tests migrated from tests/linalg_statistics_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

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

#[test]
fn test_pinverse_cpu_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[4, 3], &cpu_device);
    let cpu_result: Vec<f32> = cpu_client.pinverse(&cpu_a, None).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_a =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[4, 3], &cuda_device);
        let cuda_result: Vec<f32> = cuda_client.pinverse(&cuda_a, None).unwrap().to_vec();
        assert_allclose_f32(
            &cpu_result,
            &cuda_result,
            1e-4,
            1e-4,
            "pinverse CPU vs CUDA",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_a =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[4, 3], &wgpu_device);
        let wgpu_result: Vec<f32> = wgpu_client.pinverse(&wgpu_a, None).unwrap().to_vec();
        assert_allclose_f32(
            &cpu_result,
            &wgpu_result,
            1e-3,
            1e-3,
            "pinverse CPU vs WGPU",
        );
    });
}

#[test]
fn test_cond_cpu_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = vec![4.0f32, 2.0, 2.0, 3.0];
    let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &cpu_device);
    let cpu_result: Vec<f32> = cpu_client.cond(&cpu_a).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_a =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[2, 2], &cuda_device);
        let cuda_result: Vec<f32> = cuda_client.cond(&cuda_a).unwrap().to_vec();
        assert_allclose_f32(&cpu_result, &cuda_result, 1e-4, 1e-4, "cond CPU vs CUDA");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_a =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[2, 2], &wgpu_device);
        let wgpu_result: Vec<f32> = wgpu_client.cond(&wgpu_a).unwrap().to_vec();
        assert_allclose_f32(&cpu_result, &wgpu_result, 1e-3, 1e-3, "cond CPU vs WGPU");
    });
}

#[test]
fn test_cov_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 3], &cpu_device);
    let cpu_result: Vec<f32> = cpu_client.cov(&cpu_a, Some(1)).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_a =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[3, 3], &cuda_device);
        let cuda_result: Vec<f32> = cuda_client.cov(&cuda_a, Some(1)).unwrap().to_vec();
        assert_allclose_f32(&cpu_result, &cuda_result, 1e-4, 1e-4, "cov CPU vs CUDA");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_a =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[3, 3], &wgpu_device);
        let wgpu_result: Vec<f32> = wgpu_client.cov(&wgpu_a, Some(1)).unwrap().to_vec();
        assert_allclose_f32(&cpu_result, &wgpu_result, 1e-3, 1e-3, "cov CPU vs WGPU");
    });
}

#[test]
fn test_corrcoef_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 3], &cpu_device);
    let cpu_result: Vec<f32> = cpu_client.corrcoef(&cpu_a).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_a =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[3, 3], &cuda_device);
        let cuda_result: Vec<f32> = cuda_client.corrcoef(&cuda_a).unwrap().to_vec();
        assert_allclose_f32(
            &cpu_result,
            &cuda_result,
            1e-4,
            1e-4,
            "corrcoef CPU vs CUDA",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_a =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[3, 3], &wgpu_device);
        let wgpu_result: Vec<f32> = wgpu_client.corrcoef(&wgpu_a).unwrap().to_vec();
        assert_allclose_f32(
            &cpu_result,
            &wgpu_result,
            1e-3,
            1e-3,
            "corrcoef CPU vs WGPU",
        );
    });
}

#[test]
fn test_corrcoef_zero_variance_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = vec![1.0f32, 2.0, 1.0, 3.0, 1.0, 4.0];
    let cpu_a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 2], &cpu_device);
    let cpu_result: Vec<f32> = cpu_client.corrcoef(&cpu_a).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_a =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[3, 2], &cuda_device);
        let cuda_result: Vec<f32> = cuda_client.corrcoef(&cuda_a).unwrap().to_vec();
        assert_allclose_f32(
            &cpu_result,
            &cuda_result,
            1e-5,
            1e-5,
            "corrcoef zero-variance CPU vs CUDA",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_a =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[3, 2], &wgpu_device);
        let wgpu_result: Vec<f32> = wgpu_client.corrcoef(&wgpu_a).unwrap().to_vec();
        assert_allclose_f32(
            &cpu_result,
            &wgpu_result,
            1e-4,
            1e-4,
            "corrcoef zero-variance CPU vs WGPU",
        );
    });
}
