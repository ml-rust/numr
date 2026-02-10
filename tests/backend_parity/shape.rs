// Backend parity tests for ShapeOps trait
//
// Tests verify that ShapeOps operations produce identical results across
// CPU, CUDA, and WebGPU backends.
//
// Migrated from scattered cuda_parity/wgpu_parity modules in shape_ops.rs.

use numr::ops::ShapeOps;
use numr::tensor::Tensor;

use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

fn test_repeat_on_backends(data: &[f32], shape: &[usize], repeats: &[usize]) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(data, shape, &cpu_device);
    let cpu_result = cpu_client.repeat(&cpu_tensor, repeats).unwrap();
    let cpu_data: Vec<f32> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor = Tensor::from_slice(data, shape, &cuda_device);
        let cuda_result = cuda_client.repeat(&cuda_tensor, repeats).unwrap();
        assert_eq!(cpu_result.shape(), cuda_result.shape());
        assert_parity_f32(&cpu_data, &cuda_result.to_vec::<f32>(), "repeat_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(data, shape, &wgpu_device);
        let wgpu_result = wgpu_client.repeat(&wgpu_tensor, repeats).unwrap();
        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        assert_parity_f32(&cpu_data, &wgpu_result.to_vec::<f32>(), "repeat_wgpu");
    });
}

fn test_pad_on_backends(data: &[f32], shape: &[usize], padding: &[usize], value: f64) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(data, shape, &cpu_device);
    let cpu_result = cpu_client.pad(&cpu_tensor, padding, value).unwrap();
    let cpu_data: Vec<f32> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor = Tensor::from_slice(data, shape, &cuda_device);
        let cuda_result = cuda_client.pad(&cuda_tensor, padding, value).unwrap();
        assert_eq!(cpu_result.shape(), cuda_result.shape());
        assert_parity_f32(&cpu_data, &cuda_result.to_vec::<f32>(), "pad_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(data, shape, &wgpu_device);
        let wgpu_result = wgpu_client.pad(&wgpu_tensor, padding, value).unwrap();
        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        assert_parity_f32(&cpu_data, &wgpu_result.to_vec::<f32>(), "pad_wgpu");
    });
}

fn test_roll_on_backends(data: &[f32], shape: &[usize], shift: isize, dim: isize) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(data, shape, &cpu_device);
    let cpu_result = cpu_client.roll(&cpu_tensor, shift, dim).unwrap();
    let cpu_data: Vec<f32> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor = Tensor::from_slice(data, shape, &cuda_device);
        let cuda_result = cuda_client.roll(&cuda_tensor, shift, dim).unwrap();
        assert_eq!(cpu_result.shape(), cuda_result.shape());
        assert_parity_f32(&cpu_data, &cuda_result.to_vec::<f32>(), "roll_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(data, shape, &wgpu_device);
        let wgpu_result = wgpu_client.roll(&wgpu_tensor, shift, dim).unwrap();
        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        assert_parity_f32(&cpu_data, &wgpu_result.to_vec::<f32>(), "roll_wgpu");
    });
}

fn test_flip_on_backends(data: &[f32], shape: &[usize], dim: isize) {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    let cpu_device = CpuDevice::new();
    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(data, shape, &cpu_device);
    let cpu_result = cpu_tensor.flip(dim).unwrap();
    let cpu_data: Vec<f32> = cpu_result.contiguous().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|_cuda_client, cuda_device| {
        let cuda_tensor =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(data, shape, &cuda_device);
        let cuda_result = cuda_tensor.flip(dim).unwrap();
        assert_eq!(cpu_result.shape(), cuda_result.shape());
        assert_parity_f32(
            &cpu_data,
            &cuda_result.contiguous().to_vec::<f32>(),
            "flip_cuda",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|_wgpu_client, wgpu_device| {
        let wgpu_tensor =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(data, shape, &wgpu_device);
        let wgpu_result = wgpu_tensor.flip(dim).unwrap();
        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        assert_parity_f32(
            &cpu_data,
            &wgpu_result.contiguous().to_vec::<f32>(),
            "flip_wgpu",
        );
    });
}

// ============================================================================
// Shape Operation Parity Tests
// ============================================================================

#[test]
fn test_repeat_parity() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    test_repeat_on_backends(&data, &[2, 3], &[2, 3]);
}

#[test]
fn test_pad_parity() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Pad last dim by (1, 2), second-to-last by (1, 1)
    test_pad_on_backends(&data, &[2, 3], &[1, 2, 1, 1], 0.0);
}

#[test]
fn test_roll_parity() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    test_roll_on_backends(&data, &[2, 3], 2, 1);
}

#[test]
fn test_flip_parity() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    test_flip_on_backends(&data, &[2, 3], 1);
}
