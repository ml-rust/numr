// Backend parity tests for MatmulOps::matmul_bias

use numr::ops::{BinaryOps, MatmulOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

fn cpu_reference(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    bias: &[f32],
) -> Vec<f32> {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::from_slice(a, a_shape, &cpu_device);
    let b_t = Tensor::from_slice(b, b_shape, &cpu_device);
    let bias_t = Tensor::from_slice(bias, &[bias.len()], &cpu_device);
    cpu_client
        .matmul_bias(&a_t, &b_t, &bias_t)
        .unwrap()
        .to_vec::<f32>()
}

#[test]
fn test_matmul_bias_2d_parity() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let bias = vec![1.0f32, 2.0];
    let cpu = cpu_reference(&a, &[2, 2], &b, &[2, 2], &bias);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a_t = Tensor::from_slice(&a, &[2, 2], &cuda_device);
        let b_t = Tensor::from_slice(&b, &[2, 2], &cuda_device);
        let bias_t = Tensor::from_slice(&bias, &[2], &cuda_device);
        let got: Vec<f32> = cuda_client
            .matmul_bias(&a_t, &b_t, &bias_t)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "matmul_bias_2d_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a_t = Tensor::from_slice(&a, &[2, 2], &wgpu_device);
        let b_t = Tensor::from_slice(&b, &[2, 2], &wgpu_device);
        let bias_t = Tensor::from_slice(&bias, &[2], &wgpu_device);
        let got: Vec<f32> = wgpu_client
            .matmul_bias(&a_t, &b_t, &bias_t)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "matmul_bias_2d_wgpu");
    });
}

#[test]
fn test_matmul_bias_batched_parity() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
    let bias = vec![0.5f32, 1.0];
    let cpu = cpu_reference(&a, &[2, 2, 2], &b, &[2, 2, 2], &bias);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a_t = Tensor::from_slice(&a, &[2, 2, 2], &cuda_device);
        let b_t = Tensor::from_slice(&b, &[2, 2, 2], &cuda_device);
        let bias_t = Tensor::from_slice(&bias, &[2], &cuda_device);
        let got: Vec<f32> = cuda_client
            .matmul_bias(&a_t, &b_t, &bias_t)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "matmul_bias_batched_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a_t = Tensor::from_slice(&a, &[2, 2, 2], &wgpu_device);
        let b_t = Tensor::from_slice(&b, &[2, 2, 2], &wgpu_device);
        let bias_t = Tensor::from_slice(&bias, &[2], &wgpu_device);
        let got: Vec<f32> = wgpu_client
            .matmul_bias(&a_t, &b_t, &bias_t)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "matmul_bias_batched_wgpu");
    });
}

#[test]
fn test_matmul_bias_matches_matmul_plus_bias() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu_device);
    let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu_device);
    let bias = Tensor::from_slice(&[0.1f32, 0.2], &[2], &cpu_device);
    let fused: Vec<f32> = cpu_client.matmul_bias(&a, &b, &bias).unwrap().to_vec();
    let reference: Vec<f32> = cpu_client
        .add(
            &cpu_client.matmul(&a, &b).unwrap(),
            &bias.broadcast_to(&[2, 2]).unwrap(),
        )
        .unwrap()
        .to_vec();
    assert_parity_f32(&fused, &reference, "matmul_bias_matches_reference_cpu");
}

#[test]
fn test_cpu_matmul_bias_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(1024)));

    let a_shape = [4, 20, 16];
    let b_shape = [4, 16, 10];
    let bias_shape = [10];
    let a_numel: usize = a_shape.iter().product();
    let b_numel: usize = b_shape.iter().product();
    let bias_numel: usize = bias_shape.iter().product();

    let a_data: Vec<f32> = (0..a_numel)
        .map(|i| (i as f32 * 0.009).sin() + (i as f32 * 0.004).cos())
        .collect();
    let b_data: Vec<f32> = (0..b_numel)
        .map(|i| (i as f32 * 0.015).cos() - (i as f32 * 0.006).sin())
        .collect();
    let bias_data: Vec<f32> = (0..bias_numel).map(|i| (i as f32 * 0.021).sin()).collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &a_shape, &device);
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &b_shape, &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&bias_data, &bias_shape, &device);

    let base: Vec<f32> = default_client.matmul_bias(&a, &b, &bias).unwrap().to_vec();
    let cfg: Vec<f32> = configured_client
        .matmul_bias(&a, &b, &bias)
        .unwrap()
        .to_vec();
    assert_parity_f32(&base, &cfg, "cpu_matmul_bias_parallelism_config");
}
