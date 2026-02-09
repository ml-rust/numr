// Backend parity tests migrated from tests/sort_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::ops::*;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn assert_close(cpu: &[f32], other: &[f32], tol: f32) {
    assert_eq!(cpu.len(), other.len(), "Length mismatch");
    for (i, (c, g)) in cpu.iter().zip(other.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff <= tol,
            "Mismatch at index {}: CPU={}, GPU={}, diff={}",
            i,
            c,
            g,
            diff
        );
    }
}

#[test]
fn test_sort_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device);
    let cpu_sorted = cpu_client.sort(&cpu_tensor, 0, false).unwrap();
    let cpu_data: Vec<f32> = cpu_sorted.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[8], &cuda_device);
        let cuda_sorted = cuda_client.sort(&cuda_tensor, 0, false).unwrap();
        let cuda_data: Vec<f32> = cuda_sorted.to_vec();
        assert_close(&cpu_data, &cuda_data, 1e-6);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[8], &wgpu_device);
        let wgpu_sorted = wgpu_client.sort(&wgpu_tensor, 0, false).unwrap();
        let wgpu_data: Vec<f32> = wgpu_sorted.to_vec();
        assert_close(&cpu_data, &wgpu_data, 1e-6);
    });
}

#[test]
fn test_argsort_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0];
    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device);
    let cpu_indices = cpu_client.argsort(&cpu_tensor, 0, false).unwrap();
    let cpu_data: Vec<i64> = cpu_indices.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[5], &cuda_device);
        let cuda_indices = cuda_client.argsort(&cuda_tensor, 0, false).unwrap();
        let cuda_data: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_data, cuda_data);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[5], &wgpu_device);
        let wgpu_indices = wgpu_client.argsort(&wgpu_tensor, 0, false).unwrap();
        let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
        let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
        assert_eq!(cpu_data, wgpu_as_i64);
    });
}

#[test]
fn test_topk_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device);
    let (cpu_vals, cpu_indices) = cpu_client.topk(&cpu_tensor, 3, 0, true, true).unwrap();
    let cpu_v: Vec<f32> = cpu_vals.to_vec();
    let cpu_i: Vec<i64> = cpu_indices.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[8], &cuda_device);
        let (cuda_vals, cuda_indices) = cuda_client.topk(&cuda_tensor, 3, 0, true, true).unwrap();
        let cuda_v: Vec<f32> = cuda_vals.to_vec();
        assert_close(&cpu_v, &cuda_v, 1e-6);
        let cuda_i: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_i, cuda_i);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&data, &[8], &wgpu_device);
        let (wgpu_vals, wgpu_indices) = wgpu_client.topk(&wgpu_tensor, 3, 0, true, true).unwrap();
        let wgpu_v: Vec<f32> = wgpu_vals.to_vec();
        assert_close(&cpu_v, &wgpu_v, 1e-6);
        let wgpu_i: Vec<i32> = wgpu_indices.to_vec();
        let wgpu_as_i64: Vec<i64> = wgpu_i.iter().map(|&x| x as i64).collect();
        assert_eq!(cpu_i, wgpu_as_i64);
    });
}

#[test]
fn test_unique_parity() {
    #[cfg(feature = "cuda")]
    let cpu_device = CpuDevice::new();
    #[cfg(feature = "cuda")]
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    #[cfg(feature = "cuda")]
    let data = [1.0f32, 2.0, 2.0, 3.0, 1.0, 4.0];
    #[cfg(feature = "cuda")]
    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[6], &cpu_device);
    #[cfg(feature = "cuda")]
    let cpu_unique = cpu_client.unique(&cpu_tensor, true).unwrap();
    #[cfg(feature = "cuda")]
    let cpu_data: Vec<f32> = cpu_unique.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[6], &cuda_device);
        let cuda_unique = cuda_client.unique(&cuda_tensor, true).unwrap();
        let cuda_data: Vec<f32> = cuda_unique.to_vec();
        assert_close(&cpu_data, &cuda_data, 1e-6);
    });
}

#[test]
fn test_nonzero_parity() {
    #[cfg(feature = "cuda")]
    let cpu_device = CpuDevice::new();
    #[cfg(feature = "cuda")]
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    #[cfg(feature = "cuda")]
    let data = [0.0f32, 1.0, 0.0, 2.0, 3.0];
    #[cfg(feature = "cuda")]
    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device);
    #[cfg(feature = "cuda")]
    let cpu_indices = cpu_client.nonzero(&cpu_tensor).unwrap();
    #[cfg(feature = "cuda")]
    let cpu_data: Vec<i64> = cpu_indices.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &[5], &cuda_device);
        let cuda_indices = cuda_client.nonzero(&cuda_tensor).unwrap();
        let cuda_data: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_data, cuda_data);
    });
}

#[test]
fn test_searchsorted_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let sorted_data = [1.0f32, 3.0, 5.0, 7.0, 9.0];
    let values_data = [2.0f32, 4.0, 6.0, 8.0];

    let cpu_sorted = Tensor::<CpuRuntime>::from_slice(&sorted_data, &[5], &cpu_device);
    let cpu_values = Tensor::<CpuRuntime>::from_slice(&values_data, &[4], &cpu_device);
    let cpu_indices = cpu_client
        .searchsorted(&cpu_sorted, &cpu_values, false)
        .unwrap();
    let cpu_data: Vec<i64> = cpu_indices.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_sorted = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &sorted_data,
            &[5],
            &cuda_device,
        );
        let cuda_values = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &values_data,
            &[4],
            &cuda_device,
        );
        let cuda_indices = cuda_client
            .searchsorted(&cuda_sorted, &cuda_values, false)
            .unwrap();
        let cuda_data: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_data, cuda_data);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_sorted = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &sorted_data,
            &[5],
            &wgpu_device,
        );
        let wgpu_values = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &values_data,
            &[4],
            &wgpu_device,
        );
        let wgpu_indices = wgpu_client
            .searchsorted(&wgpu_sorted, &wgpu_values, false)
            .unwrap();
        let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
        let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
        assert_eq!(cpu_data, wgpu_as_i64);
    });
}
