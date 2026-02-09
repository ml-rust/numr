// Backend parity tests for ConvOps

use numr::ops::{ConvOps, PaddingMode};
use numr::tensor::Tensor;

use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

#[test]
fn test_conv1d_moving_average_parity() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let weight = [1.0f32, 1.0, 1.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_in = Tensor::from_slice(&input, &[1, 1, 5], &cpu_device);
    let cpu_w = Tensor::from_slice(&weight, &[1, 1, 3], &cpu_device);
    let cpu: Vec<f32> = cpu_client
        .conv1d(&cpu_in, &cpu_w, None, 1, PaddingMode::Valid, 1, 1)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&input, &[1, 1, 5], &cuda_device);
        let w = Tensor::from_slice(&weight, &[1, 1, 3], &cuda_device);
        let got: Vec<f32> = cuda_client
            .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 1)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "conv1d_moving_average_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&input, &[1, 1, 5], &wgpu_device);
        let w = Tensor::from_slice(&weight, &[1, 1, 3], &wgpu_device);
        let got: Vec<f32> = wgpu_client
            .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 1)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "conv1d_moving_average_wgpu");
    });
}

#[test]
fn test_conv2d_box_blur_parity() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = [1.0f32; 4];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_in = Tensor::from_slice(&input, &[1, 1, 3, 3], &cpu_device);
    let cpu_w = Tensor::from_slice(&weight, &[1, 1, 2, 2], &cpu_device);
    let cpu: Vec<f32> = cpu_client
        .conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&input, &[1, 1, 3, 3], &cuda_device);
        let w = Tensor::from_slice(&weight, &[1, 1, 2, 2], &cuda_device);
        let got: Vec<f32> = cuda_client
            .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "conv2d_box_blur_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&input, &[1, 1, 3, 3], &wgpu_device);
        let w = Tensor::from_slice(&weight, &[1, 1, 2, 2], &wgpu_device);
        let got: Vec<f32> = wgpu_client
            .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "conv2d_box_blur_wgpu");
    });
}

#[test]
fn test_depthwise_conv2d_parity() {
    let input = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    let weight = [1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_in = Tensor::from_slice(&input, &[1, 2, 3, 3], &cpu_device);
    let cpu_w = Tensor::from_slice(&weight, &[2, 1, 2, 2], &cpu_device);
    let cpu: Vec<f32> = cpu_client
        .depthwise_conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1))
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&input, &[1, 2, 3, 3], &cuda_device);
        let w = Tensor::from_slice(&weight, &[2, 1, 2, 2], &cuda_device);
        let got: Vec<f32> = cuda_client
            .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "depthwise_conv2d_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&input, &[1, 2, 3, 3], &wgpu_device);
        let w = Tensor::from_slice(&weight, &[2, 1, 2, 2], &wgpu_device);
        let got: Vec<f32> = wgpu_client
            .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "depthwise_conv2d_wgpu");
    });
}

#[test]
fn test_conv2d_invalid_groups_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_in = Tensor::from_slice(&vec![0.0f32; 5 * 8 * 8], &[1, 5, 8, 8], &cpu_device);
    let cpu_w = Tensor::from_slice(&vec![0.0f32; 10 * 3 * 3 * 3], &[10, 3, 3, 3], &cpu_device);
    assert!(
        cpu_client
            .conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1), 2,)
            .is_err()
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&vec![0.0f32; 5 * 8 * 8], &[1, 5, 8, 8], &cuda_device);
        let w = Tensor::from_slice(&vec![0.0f32; 10 * 3 * 3 * 3], &[10, 3, 3, 3], &cuda_device);
        assert!(
            cuda_client
                .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 2)
                .is_err()
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&vec![0.0f32; 5 * 8 * 8], &[1, 5, 8, 8], &wgpu_device);
        let w = Tensor::from_slice(&vec![0.0f32; 10 * 3 * 3 * 3], &[10, 3, 3, 3], &wgpu_device);
        assert!(
            wgpu_client
                .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 2)
                .is_err()
        );
    });
}
