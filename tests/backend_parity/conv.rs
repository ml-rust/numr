// Backend parity tests for ConvOps
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

#[test]
fn test_conv1d_moving_average_parity() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 1, 5], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .conv1d(&cpu_in, &cpu_w, None, 1, PaddingMode::Valid, 1, 1)
            .unwrap_or_else(|e| panic!("CPU conv1d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 1, 5], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 1)
                    .unwrap_or_else(|e| panic!("CUDA conv1d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv1d_moving_average CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 1, 5], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 1)
                    .unwrap_or_else(|e| panic!("WebGPU conv1d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv1d_moving_average WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_conv2d_box_blur_parity() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = vec![1.0; 4];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 1, 3, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[1, 1, 2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap_or_else(|e| panic!("CPU conv2d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 1, 3, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 2, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
                    .unwrap_or_else(|e| panic!("CUDA conv2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv2d_box_blur CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 1, 3, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 2, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
                    .unwrap_or_else(|e| panic!("WebGPU conv2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv2d_box_blur WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_depthwise_conv2d_parity() {
    let input = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 2, 3, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[2, 1, 2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .depthwise_conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1))
            .unwrap_or_else(|e| panic!("CPU depthwise_conv2d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 2, 3, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[2, 1, 2, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
                    .unwrap_or_else(|e| panic!("CUDA depthwise_conv2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("depthwise_conv2d CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 2, 3, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[2, 1, 2, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
                    .unwrap_or_else(|e| {
                        panic!("WebGPU depthwise_conv2d failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("depthwise_conv2d WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_conv2d_invalid_groups_parity() {
    let input_data = vec![0.0; 5 * 8 * 8];
    let weight_data = vec![0.0; 10 * 3 * 3 * 3];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input_data, &[1, 5, 8, 8], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(
            &weight_data,
            &[10, 3, 3, 3],
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        assert!(
            cpu_client
                .conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1), 2,)
                .is_err()
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(
                    &input_data,
                    &[1, 5, 8, 8],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(
                    &weight_data,
                    &[10, 3, 3, 3],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                assert!(
                    cuda_client
                        .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 2)
                        .is_err()
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(
                    &input_data,
                    &[1, 5, 8, 8],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(
                    &weight_data,
                    &[10, 3, 3, 3],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                assert!(
                    wgpu_client
                        .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 2)
                        .is_err()
                );
            });
        }
    }
}
