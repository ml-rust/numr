// Backend parity tests for GemmEpilogueOps
//
// This module tests matmul_bias_activation, matmul_bias_residual, and
// matmul_bias_activation_bwd across all supported dtypes and backends,
// ensuring numerical consistency across CPU, CUDA, and WebGPU.

use numr::ops::{ActivationOps, BinaryOps, GemmActivation, GemmEpilogueOps, MatmulOps};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, assert_tensor_allclose_tol, create_cpu_client,
    gemm_long_k_tolerance, is_dtype_supported, parity_dtypes, values_close,
};

// ============================================================================
// matmul_bias_activation: 2D parity across activations, dtypes, backends
// ============================================================================

#[test]
fn test_gemm_bias_activation_none_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::None, "gemm_bias_act_none_2d");
}

#[test]
fn test_gemm_bias_activation_relu_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::ReLU, "gemm_bias_act_relu_2d");
}

#[test]
fn test_gemm_bias_activation_gelu_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::GELU, "gemm_bias_act_gelu_2d");
}

#[test]
fn test_gemm_bias_activation_silu_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::SiLU, "gemm_bias_act_silu_2d");
}

#[test]
fn test_gemm_bias_activation_sigmoid_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::Sigmoid, "gemm_bias_act_sigmoid_2d");
}

#[test]
fn test_gemm_bias_activation_tanh_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::Tanh, "gemm_bias_act_tanh_2d");
}

fn gemm_bias_activation_2d_parity(activation: GemmActivation, label: &str) {
    // [2, 3] @ [3, 2] + [2] -> [2, 2]
    let a = vec![1.0f64, 2.0, -1.0, 3.0, -2.0, 4.0];
    let b = vec![0.5f64, -0.3, 0.1, 0.7, -0.2, 0.4];
    let bias = vec![-0.1f64, 0.2];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{label} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{label} WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// 128x128x8 tile coverage (F32): every case above uses [2,3]@[3,2] or
// [2,2,3]@[2,3,2], which `f32_batched_tile_config` (m<=64 || n<=64) always
// routes to the 64x64x32 tile. These force the 128x128x8 specialized kernel.
// ============================================================================

/// Deterministic F32 fill, distinct phase per operand so a/b/bias/residual
/// don't correlate into an accidental zero. Matches the helper in matmul_bias.rs.
fn deterministic_f32(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((i as f32 * 0.013 + phase).sin() * 0.5) + ((i as f32 * 0.0047 + phase).cos() * 0.3)
        })
        .collect()
}

/// GELU with a bias offset that drives every pre-activation value deeply
/// negative, at a 128-tile size. A/B are kept small-amplitude so the bias
/// term (not the matmul accumulation) controls the sign and magnitude.
#[test]
fn gemm_bias_act_f32_128_tile_gelu_large_negative_match_cpu() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| (i as f32 * 0.013).sin() * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 * 0.017).cos() * 0.01)
        .collect();
    let bias_data = vec![-50.0f32; n];

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &a_data,
        &[m, k],
        &cpu_device,
    )
    .unwrap();
    let b_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &b_data,
        &[k, n],
        &cpu_device,
    )
    .unwrap();
    let bias_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &bias_data,
        &[n],
        &cpu_device,
    )
    .unwrap();
    let cpu_result = cpu_client
        .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::GELU)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::GELU)
                .unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "gemm_bias_act_f32_128_tile_gelu_large_negative CUDA vs CPU",
            );
        });
    }
}

/// GELU with a bias offset that drives every pre-activation value deeply
/// positive, at a 128-tile size — the counterpart of the negative case above.
#[test]
fn gemm_bias_act_f32_128_tile_gelu_large_positive_match_cpu() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| (i as f32 * 0.013).sin() * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 * 0.017).cos() * 0.01)
        .collect();
    let bias_data = vec![50.0f32; n];

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &a_data,
        &[m, k],
        &cpu_device,
    )
    .unwrap();
    let b_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &b_data,
        &[k, n],
        &cpu_device,
    )
    .unwrap();
    let bias_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &bias_data,
        &[n],
        &cpu_device,
    )
    .unwrap();
    let cpu_result = cpu_client
        .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::GELU)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::GELU)
                .unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "gemm_bias_act_f32_128_tile_gelu_large_positive CUDA vs CPU",
            );
        });
    }
}

/// Every `GemmActivation` variant at a 128-tile size, ragged K so the
/// activation epilogue is checked against the partial-tile path too.
#[test]
fn gemm_bias_act_f32_128_tile_all_activations_match_cpu() {
    let (m, k, n) = (128usize, 130usize, 128usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);
    // K=130 dot products, same bias-fold-order argument as
    // `matmul_bias_f32_128_tile_ragged_large_match_cpu` (see
    // `gemm_long_k_tolerance`'s doc comment), compounded by GELU/SiLU/Sigmoid
    // /Tanh evaluating `tanhf` differently per backend (CUDA's device
    // `tanhf` vs CPU's AVX2 polynomial `tanh_f32`) — both legitimate
    // approximations of the same transcendental function, not a bug.
    let operand_scale = a_data
        .iter()
        .chain(b_data.iter())
        .fold(0.0f64, |acc, &v| acc.max(v.abs() as f64));
    let (gemm_rtol, gemm_atol) = gemm_long_k_tolerance(numr::dtype::DType::F32, k, operand_scale);

    for activation in [
        GemmActivation::None,
        GemmActivation::ReLU,
        GemmActivation::GELU,
        GemmActivation::SiLU,
        GemmActivation::Sigmoid,
        GemmActivation::Tanh,
    ] {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &a_data,
            &[m, k],
            &cpu_device,
        )
        .unwrap();
        let b_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &b_data,
            &[k, n],
            &cpu_device,
        )
        .unwrap();
        let bias_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &bias_data,
            &[n],
            &cpu_device,
        )
        .unwrap();
        let cpu_result = cpu_client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", numr::dtype::DType::F32) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                    &a_data,
                    &[m, k],
                    &cuda_device,
                )
                .unwrap();
                let b_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                    &b_data,
                    &[k, n],
                    &cuda_device,
                )
                .unwrap();
                let bias_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                    &bias_data,
                    &[n],
                    &cuda_device,
                )
                .unwrap();
                let result = cuda_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
                    .unwrap();
                assert_tensor_allclose_tol(
                    &result,
                    &cpu_result,
                    gemm_rtol,
                    gemm_atol,
                    &format!(
                        "gemm_bias_act_f32_128_tile_all_activations[{activation:?}] CUDA vs CPU"
                    ),
                );
            });
        }
    }
}

/// Both CPU and CUDA F32 results checked against an F64 ground truth,
/// computed from the SAME inputs (cast up, never regenerated) at the SAME
/// shape/activations as `gemm_bias_act_f32_128_tile_all_activations_match_cpu`.
///
/// This is a stronger statement than CPU-vs-CUDA agreement: two backends
/// could in principle agree with each other while both being wrong. Diagnosed
/// while investigating that test's GELU failure under `tolerance_for_dtype`
/// — this is the permanent record of that diagnosis, not a throwaway.
/// `gemm_long_k_tolerance` is the same accumulation-scaled bound used there;
/// a real accumulation or activation bug (as opposed to a benign
/// bias-fold-order reassociation plus differing `tanhf` implementations)
/// would blow through it, because it targets f64 truth rather than another
/// F32 backend that could share the same bug.
#[test]
fn gemm_bias_act_f32_128_tile_all_activations_match_f64_truth() {
    let (m, k, n) = (128usize, 130usize, 128usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);
    let operand_scale = a_data
        .iter()
        .chain(b_data.iter())
        .fold(0.0f64, |acc, &v| acc.max(v.abs() as f64));
    let (_, gemm_atol) = gemm_long_k_tolerance(numr::dtype::DType::F32, k, operand_scale);

    let a_f64: Vec<f64> = a_data.iter().map(|&v| v as f64).collect();
    let b_f64: Vec<f64> = b_data.iter().map(|&v| v as f64).collect();
    let bias_f64: Vec<f64> = bias_data.iter().map(|&v| v as f64).collect();

    for activation in [
        GemmActivation::None,
        GemmActivation::ReLU,
        GemmActivation::GELU,
        GemmActivation::SiLU,
        GemmActivation::Sigmoid,
        GemmActivation::Tanh,
    ] {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_t64 = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &a_f64,
            &[m, k],
            &cpu_device,
        )
        .unwrap();
        let b_t64 = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &b_f64,
            &[k, n],
            &cpu_device,
        )
        .unwrap();
        let bias_t64 = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &bias_f64,
            &[n],
            &cpu_device,
        )
        .unwrap();
        let truth: Vec<f64> = cpu_client
            .matmul_bias_activation(&a_t64, &b_t64, &bias_t64, activation)
            .unwrap()
            .to_vec();

        let a_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &a_data,
            &[m, k],
            &cpu_device,
        )
        .unwrap();
        let b_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &b_data,
            &[k, n],
            &cpu_device,
        )
        .unwrap();
        let bias_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &bias_data,
            &[n],
            &cpu_device,
        )
        .unwrap();
        let cpu_f32: Vec<f32> = cpu_client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
            .unwrap()
            .to_vec();

        for (i, (&c, &t)) in cpu_f32.iter().zip(truth.iter()).enumerate() {
            assert!(
                values_close(c as f64, t, 0.0, gemm_atol),
                "gemm_bias_act_f32_128_tile_all_activations[{activation:?}] CPU vs F64 truth: element {i} differs: {c} vs {t} (tol={gemm_atol:.2e})"
            );
        }

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", numr::dtype::DType::F32) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                    &a_data,
                    &[m, k],
                    &cuda_device,
                )
                .unwrap();
                let b_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                    &b_data,
                    &[k, n],
                    &cuda_device,
                )
                .unwrap();
                let bias_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                    &bias_data,
                    &[n],
                    &cuda_device,
                )
                .unwrap();
                let cuda_f32: Vec<f32> = cuda_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
                    .unwrap()
                    .to_vec();
                for (i, (&c, &t)) in cuda_f32.iter().zip(truth.iter()).enumerate() {
                    assert!(
                        values_close(c as f64, t, 0.0, gemm_atol),
                        "gemm_bias_act_f32_128_tile_all_activations[{activation:?}] CUDA vs F64 truth: element {i} differs: {c} vs {t} (tol={gemm_atol:.2e})"
                    );
                }
            });
        }
    }
}

/// K == 0 at a 128-tile size for `matmul_bias_activation`: the tiled kernel
/// used to leave C unwritten for K==0 and now must write the epilogue
/// (activation applied to zeros plus bias), same as CPU.
#[test]
fn gemm_bias_act_f32_128_tile_k_zero_match_cpu() {
    let (m, k, n) = (100usize, 0usize, 100usize);
    let a_data: Vec<f32> = Vec::new();
    let b_data: Vec<f32> = Vec::new();
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &a_data,
        &[m, k],
        &cpu_device,
    )
    .unwrap();
    let b_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &b_data,
        &[k, n],
        &cpu_device,
    )
    .unwrap();
    let bias_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &bias_data,
        &[n],
        &cpu_device,
    )
    .unwrap();
    let cpu_result = cpu_client
        .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
                .unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "gemm_bias_act_f32_128_tile_k_zero CUDA vs CPU",
            );
        });
    }
}

/// `matmul_bias_residual` at a 128-tile size: residual is indexed elementwise
/// over the full output, so a partial-tile bug in the residual add would
/// show up here even if bias/activation are correct.
#[test]
fn gemm_bias_residual_f32_128_tile_match_cpu() {
    let (m, k, n) = (256usize, 128usize, 256usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);
    let residual_data = deterministic_f32(m * n, 5.3);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &a_data,
        &[m, k],
        &cpu_device,
    )
    .unwrap();
    let b_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &b_data,
        &[k, n],
        &cpu_device,
    )
    .unwrap();
    let bias_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &bias_data,
        &[n],
        &cpu_device,
    )
    .unwrap();
    let res_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &residual_data,
        &[m, n],
        &cpu_device,
    )
    .unwrap();
    let cpu_result = cpu_client
        .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let res_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &residual_data,
                &[m, n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                .unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "gemm_bias_residual_f32_128_tile CUDA vs CPU",
            );
        });
    }
}

/// `matmul_bias_residual` at a 128-tile size with M, N, and K all ragged:
/// the case most likely to catch an out-of-range residual read at the
/// partial-tile boundary.
#[test]
fn gemm_bias_residual_f32_128_tile_ragged_match_cpu() {
    let (m, k, n) = (130usize, 70usize, 130usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);
    let residual_data = deterministic_f32(m * n, 5.3);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &a_data,
        &[m, k],
        &cpu_device,
    )
    .unwrap();
    let b_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &b_data,
        &[k, n],
        &cpu_device,
    )
    .unwrap();
    let bias_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &bias_data,
        &[n],
        &cpu_device,
    )
    .unwrap();
    let res_t = numr::tensor::Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
        &residual_data,
        &[m, n],
        &cpu_device,
    )
    .unwrap();
    let cpu_result = cpu_client
        .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let res_t = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &residual_data,
                &[m, n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                .unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "gemm_bias_residual_f32_128_tile_ragged CUDA vs CPU",
            );
        });
    }
}

// ============================================================================
// matmul_bias_activation: batched 3D parity
// ============================================================================

#[test]
fn test_gemm_bias_activation_batched_3d_parity() {
    // [2, 2, 3] @ [2, 3, 2] + [2] -> [2, 2, 2]
    let a = vec![
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b = vec![
        0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ];
    let bias = vec![0.01f64, 0.02];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 2, 3], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[2, 3, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t =
                    tensor_from_f64(&a, &[2, 2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t =
                    tensor_from_f64(&b, &[2, 3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_act_batched CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t =
                    tensor_from_f64(&a, &[2, 2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t =
                    tensor_from_f64(&b, &[2, 3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_act_batched WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// matmul_bias_residual: 2D parity across dtypes and backends
// ============================================================================

#[test]
fn test_gemm_bias_residual_2d_parity() {
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![0.5f64, -0.3, 0.1, 0.7, -0.2, 0.4];
    let bias = vec![-0.1f64, 0.2];
    let residual = vec![1.0f64, 2.0, 3.0, 4.0];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let res_t = tensor_from_f64(&residual, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client
            .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let res_t =
                    tensor_from_f64(&residual, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client
                    .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_residual_2d CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let res_t =
                    tensor_from_f64(&residual, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client
                    .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_residual_2d WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// matmul_bias_activation_bwd: parity across dtypes and backends
// ============================================================================

#[test]
fn test_gemm_bias_activation_bwd_none_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::None, "gemm_bias_act_bwd_none");
}

#[test]
fn test_gemm_bias_activation_bwd_relu_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::ReLU, "gemm_bias_act_bwd_relu");
}

#[test]
fn test_gemm_bias_activation_bwd_sigmoid_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::Sigmoid, "gemm_bias_act_bwd_sigmoid");
}

#[test]
fn test_gemm_bias_activation_bwd_tanh_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::Tanh, "gemm_bias_act_bwd_tanh");
}

#[test]
fn test_gemm_bias_activation_bwd_silu_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::SiLU, "gemm_bias_act_bwd_silu");
}

#[test]
fn test_gemm_bias_activation_bwd_gelu_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::GELU, "gemm_bias_act_bwd_gelu");
}

fn gemm_bias_activation_bwd_parity(activation: GemmActivation, label: &str) {
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![0.5f64, 0.3, -0.1, 0.7];
    let bias = vec![0.0f64, 0.0];
    let grad = vec![1.0f64, 1.0, 1.0, 1.0];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let grad_t = tensor_from_f64(&grad, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let (cpu_da, cpu_db, cpu_dbias) = cpu_client
            .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let grad_t =
                    tensor_from_f64(&grad, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let (da, db, dbias) = cuda_client
                    .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                    .unwrap();
                assert_tensor_allclose(
                    &da,
                    &cpu_da,
                    dtype,
                    &format!("{label} d_a CUDA vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &db,
                    &cpu_db,
                    dtype,
                    &format!("{label} d_b CUDA vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &dbias,
                    &cpu_dbias,
                    dtype,
                    &format!("{label} d_bias CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t = tensor_from_f64(&a, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t = tensor_from_f64(&b, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let grad_t =
                    tensor_from_f64(&grad, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let (da, db, dbias) = wgpu_client
                    .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                    .unwrap();
                assert_tensor_allclose(
                    &da,
                    &cpu_da,
                    dtype,
                    &format!("{label} d_a WebGPU vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &db,
                    &cpu_db,
                    dtype,
                    &format!("{label} d_b WebGPU vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &dbias,
                    &cpu_dbias,
                    dtype,
                    &format!("{label} d_bias WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// matmul_bias_activation_bwd: batched 3D parity
// ============================================================================

#[test]
fn test_gemm_bias_activation_bwd_batched_3d_parity() {
    let a = vec![
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b = vec![
        0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ];
    let bias = vec![0.01f64, 0.02];
    let grad = vec![1.0f64; 8];

    for activation in [
        GemmActivation::None,
        GemmActivation::ReLU,
        GemmActivation::SiLU,
    ] {
        for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
            let (cpu_client, cpu_device) = create_cpu_client();
            let a_t = tensor_from_f64(&a, &[2, 2, 3], dtype, &cpu_device, &cpu_client).unwrap();
            let b_t = tensor_from_f64(&b, &[2, 3, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
            let grad_t =
                tensor_from_f64(&grad, &[2, 2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let (cpu_da, cpu_db, cpu_dbias) = cpu_client
                .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                .unwrap();

            assert_eq!(cpu_da.shape(), &[2, 2, 3]);
            assert_eq!(cpu_db.shape(), &[2, 3, 2]);
            assert_eq!(cpu_dbias.shape(), &[2]);

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2, 2], dtype, &cuda_device, &cuda_client)
                            .unwrap();
                    let label = format!("bwd_batched_{activation:?}");
                    let (da, db, dbias) = cuda_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias CUDA vs CPU [{dtype:?}]"),
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|wgpu_client, wgpu_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2, 2], dtype, &wgpu_device, &wgpu_client)
                            .unwrap();
                    let label = format!("bwd_batched_{activation:?}");
                    let (da, db, dbias) = wgpu_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias WebGPU vs CPU [{dtype:?}]"),
                    );
                });
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Assert all elements of a tensor are finite, reading as the correct native dtype.
fn assert_finite<R: numr::runtime::Runtime>(
    tensor: &numr::tensor::Tensor<R>,
    dtype: numr::dtype::DType,
    label: &str,
) {
    use numr::dtype::DType;
    macro_rules! check {
        ($T:ty) => {
            for (i, val) in tensor.to_vec::<$T>().iter().enumerate() {
                let v = *val as f64;
                assert!(
                    v.is_finite(),
                    "non-finite {label} [{dtype:?}] at index {i}: {v}"
                );
            }
        };
    }
    match dtype {
        DType::F64 => check!(f64),
        DType::F32 => check!(f32),
        #[cfg(feature = "f16")]
        DType::F16 => {
            for (i, val) in tensor.to_vec::<half::f16>().iter().enumerate() {
                let v = f32::from(*val) as f64;
                assert!(
                    v.is_finite(),
                    "non-finite {label} [{dtype:?}] at index {i}: {v}"
                );
            }
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            for (i, val) in tensor.to_vec::<half::bf16>().iter().enumerate() {
                let v = f32::from(*val) as f64;
                assert!(
                    v.is_finite(),
                    "non-finite {label} [{dtype:?}] at index {i}: {v}"
                );
            }
        }
        _ => {} // integer/bool dtypes are always "finite"
    }
}

// ============================================================================
// matmul_bias_activation_bwd: negative values / edge cases
// ============================================================================

#[test]
fn test_gemm_bias_activation_bwd_negative_values_parity() {
    let a = vec![-1.0f64, 2.0, 3.0, -4.0];
    let b = vec![-1.0f64, 0.5, 0.5, -1.0];
    let bias = vec![-0.5f64, 0.5];
    let grad = vec![1.0f64, 1.0, 1.0, 1.0];

    for activation in [
        GemmActivation::None,
        GemmActivation::ReLU,
        GemmActivation::Sigmoid,
        GemmActivation::Tanh,
        GemmActivation::SiLU,
        GemmActivation::GELU,
    ] {
        for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
            let (cpu_client, cpu_device) = create_cpu_client();
            let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
            let grad_t = tensor_from_f64(&grad, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let (cpu_da, cpu_db, cpu_dbias) = cpu_client
                .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                .unwrap();

            // Verify finiteness on CPU reference (must read as native dtype,
            // not f64, because to_vec is a raw byte copy with no conversion)
            assert_finite(&cpu_da, dtype, &format!("d_a for {activation:?}"));
            assert_finite(&cpu_db, dtype, &format!("d_b for {activation:?}"));
            assert_finite(&cpu_dbias, dtype, &format!("d_bias for {activation:?}"));

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let label = format!("bwd_neg_{activation:?}");
                    let (da, db, dbias) = cuda_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias CUDA vs CPU [{dtype:?}]"),
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|wgpu_client, wgpu_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let label = format!("bwd_neg_{activation:?}");
                    let (da, db, dbias) = wgpu_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias WebGPU vs CPU [{dtype:?}]"),
                    );
                });
            }
        }
    }
}

// ============================================================================
// matmul_bias_activation_bwd: long-row sums in a half dtype
// ============================================================================
//
// `d_b = A^T @ grad_pre` and `d_bias = sum(grad_pre, dim=0)` both sum over M.
// With M = 4096 rows of `a = 1/4096` and `grad = 1`, the true `d_b` is 1.0
// and the true `d_bias` is 4096. An accumulator held in the storage dtype
// stalls long before that: BF16 counting 1.0 per row cannot pass 256, and its
// 1/4096 steps stop moving a sum once it reaches about 1/16. Every operand
// value is exact in F16 and BF16, so the F64 CPU run of the same values is
// the reference and only the final rounding of each output separates them.

/// Reads any float tensor back as `f64` via a CPU cast.
#[cfg(feature = "f16")]
fn read_f64<R: numr::runtime::Runtime<DType = numr::dtype::DType>>(
    t: &numr::tensor::Tensor<R>,
    client: &impl numr::ops::TypeConversionOps<R>,
) -> Vec<f64> {
    client
        .cast(t, numr::dtype::DType::F64)
        .unwrap()
        .to_vec::<f64>()
}

#[cfg(feature = "f16")]
fn assert_bwd_long_m_half(activation: GemmActivation, label: &str) {
    use numr::dtype::DType;

    let (m, k, n) = (4096usize, 8usize, 8usize);
    let a = vec![1.0f64 / 4096.0; m * k];
    let b = vec![0.5f64; k * n];
    let bias = vec![0.0f64; n];
    let grad = vec![1.0f64; m * n];

    let (cpu_client, cpu_device) = create_cpu_client();
    let run_cpu = |dtype: DType| {
        let a_t = tensor_from_f64(&a, &[m, k], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[k, n], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[n], dtype, &cpu_device, &cpu_client).unwrap();
        let grad_t = tensor_from_f64(&grad, &[m, n], dtype, &cpu_device, &cpu_client).unwrap();
        let (da, db, dbias) = cpu_client
            .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
            .unwrap();
        (
            read_f64(&da, &cpu_client),
            read_f64(&db, &cpu_client),
            read_f64(&dbias, &cpu_client),
        )
    };
    let (ref_da, ref_db, ref_dbias) = run_cpu(DType::F64);

    // The outputs round once at the store: 2^-9 relative for BF16.
    let check = |got: &[f64], want: &[f64], what: &str, dtype: DType, backend: &str| {
        assert_eq!(
            got.len(),
            want.len(),
            "{label} {what} {backend} [{dtype:?}]"
        );
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                values_close(*g, *w, 4e-3, 1e-6),
                "{label} {what} {backend} [{dtype:?}] at {i}: {g} vs F64 {w}"
            );
        }
    };

    for dtype in [DType::F16, DType::BF16] {
        if !is_dtype_supported("cpu", dtype) {
            continue;
        }
        let (da, db, dbias) = run_cpu(dtype);
        check(&da, &ref_da, "d_a", dtype, "CPU");
        check(&db, &ref_db, "d_b", dtype, "CPU");
        check(&dbias, &ref_dbias, "d_bias", dtype, "CPU");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[m, k], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[k, n], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[n], dtype, &cuda_device, &cuda_client).unwrap();
                let grad_t =
                    tensor_from_f64(&grad, &[m, n], dtype, &cuda_device, &cuda_client).unwrap();
                let (da, db, dbias) = cuda_client
                    .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                    .unwrap();
                check(&read_f64(&da, &cuda_client), &ref_da, "d_a", dtype, "CUDA");
                check(&read_f64(&db, &cuda_client), &ref_db, "d_b", dtype, "CUDA");
                check(
                    &read_f64(&dbias, &cuda_client),
                    &ref_dbias,
                    "d_bias",
                    dtype,
                    "CUDA",
                );
            });
        }
    }
}

#[cfg(feature = "f16")]
#[test]
fn test_gemm_bias_activation_bwd_long_m_half_none() {
    assert_bwd_long_m_half(GemmActivation::None, "bwd_long_m_none");
}

#[cfg(feature = "f16")]
#[test]
fn test_gemm_bias_activation_bwd_long_m_half_gelu() {
    assert_bwd_long_m_half(GemmActivation::GELU, "bwd_long_m_gelu");
}

// ============================================================================
// CPU-only reference tests: fused == unfused
// ============================================================================

#[test]
fn test_gemm_bias_activation_none_matches_matmul_bias() {
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    let (client, dev) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.3, 0.1, 0.7, -0.2, 0.4], &[3, 2], &dev)
        .unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[-0.1f32, 0.2], &[2], &dev).unwrap();

    let fused: Vec<f32> = client
        .matmul_bias_activation(&a, &b, &bias, GemmActivation::None)
        .unwrap()
        .to_vec();
    let reference: Vec<f32> = client.matmul_bias(&a, &b, &bias).unwrap().to_vec();

    crate::backend_parity::helpers::assert_parity_f32(
        &fused,
        &reference,
        "gemm_bias_act_none_matches_matmul_bias",
    );
}

#[test]
fn test_gemm_bias_activation_relu_matches_unfused() {
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    let (client, dev) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, -1.0, 3.0, -2.0, 4.0], &[2, 3], &dev)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.3, 0.1, 0.7, -0.2, 0.4], &[3, 2], &dev)
        .unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[-0.5f32, 0.3], &[2], &dev).unwrap();

    let fused: Vec<f32> = client
        .matmul_bias_activation(&a, &b, &bias, GemmActivation::ReLU)
        .unwrap()
        .to_vec();
    let pre = client.matmul_bias(&a, &b, &bias).unwrap();
    let unfused: Vec<f32> = client.relu(&pre).unwrap().to_vec();

    crate::backend_parity::helpers::assert_parity_f32(
        &fused,
        &unfused,
        "gemm_bias_act_relu_matches_unfused",
    );
}

#[test]
fn test_gemm_bias_residual_matches_unfused() {
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    let (client, dev) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.3, 0.1, 0.7, -0.2, 0.4], &[3, 2], &dev)
        .unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[-0.1f32, 0.2], &[2], &dev).unwrap();
    let residual =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &dev).unwrap();

    let fused: Vec<f32> = client
        .matmul_bias_residual(&a, &b, &bias, &residual)
        .unwrap()
        .to_vec();
    let pre = client.matmul_bias(&a, &b, &bias).unwrap();
    let unfused: Vec<f32> = client.add(&pre, &residual).unwrap().to_vec();

    crate::backend_parity::helpers::assert_parity_f32(
        &fused,
        &unfused,
        "gemm_bias_residual_matches_unfused",
    );
}

// ============================================================================
// Rank-1 `b`: `[m, k] @ [k] + [1]` is `[m, 1]` on every backend
// ============================================================================

/// The shared rule (`matmul_mkn`) makes a rank-1 `b` a `[k, 1]` column, so the
/// bias is `[1]` and the output `[m, 1]`. CPU is checked against an F64 sum,
/// CUDA and WebGPU against CPU, for the activation and the residual epilogues.
#[test]
fn gemm_epilogue_rank1_b_match_reference_and_cpu() {
    let (m, k) = (5usize, 7usize);
    let a: Vec<f64> = (0..m * k)
        .map(|i| ((i * 5) % 11) as f64 * 0.25 - 1.0)
        .collect();
    let b: Vec<f64> = (0..k).map(|i| ((i * 3) % 5) as f64 * 0.5 - 1.0).collect();
    let bias = [0.125f64];
    let residual: Vec<f64> = (0..m).map(|i| i as f64 * 0.1 - 0.2).collect();
    let dtype = numr::dtype::DType::F32;

    let pre: Vec<f64> = (0..m)
        .map(|i| (0..k).map(|p| a[i * k + p] * b[p]).sum::<f64>() + bias[0])
        .collect();
    let want_relu: Vec<f64> = pre.iter().map(|v| v.max(0.0)).collect();
    let want_res: Vec<f64> = pre.iter().zip(&residual).map(|(v, r)| v + r).collect();

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = tensor_from_f64(&a, &[m, k], dtype, &cpu_device, &cpu_client).unwrap();
    let b_t = tensor_from_f64(&b, &[k], dtype, &cpu_device, &cpu_client).unwrap();
    let bias_t = tensor_from_f64(&bias, &[1], dtype, &cpu_device, &cpu_client).unwrap();
    let res_t = tensor_from_f64(&residual, &[m, 1], dtype, &cpu_device, &cpu_client).unwrap();
    let cpu_relu = cpu_client
        .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
        .expect("CPU rank-1 b matmul_bias_activation");
    let cpu_res = cpu_client
        .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
        .expect("CPU rank-1 b matmul_bias_residual");
    assert_eq!(cpu_relu.shape(), &[m, 1]);
    assert_eq!(cpu_res.shape(), &[m, 1]);
    let want_relu_t =
        tensor_from_f64(&want_relu, &[m, 1], dtype, &cpu_device, &cpu_client).unwrap();
    let want_res_t = tensor_from_f64(&want_res, &[m, 1], dtype, &cpu_device, &cpu_client).unwrap();
    assert_tensor_allclose(&cpu_relu, &want_relu_t, dtype, "rank-1 b ReLU CPU vs F64");
    assert_tensor_allclose(&cpu_res, &want_res_t, dtype, "rank-1 b residual CPU vs F64");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let a_t = tensor_from_f64(&a, &[m, k], dtype, &device, &client).unwrap();
        let b_t = tensor_from_f64(&b, &[k], dtype, &device, &client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[1], dtype, &device, &client).unwrap();
        let res_t = tensor_from_f64(&residual, &[m, 1], dtype, &device, &client).unwrap();
        let relu = client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
            .expect("CUDA rank-1 b matmul_bias_activation");
        let res = client
            .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
            .expect("CUDA rank-1 b matmul_bias_residual");
        assert_tensor_allclose(&relu, &cpu_relu, dtype, "rank-1 b ReLU CUDA vs CPU");
        assert_tensor_allclose(&res, &cpu_res, dtype, "rank-1 b residual CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let a_t = tensor_from_f64(&a, &[m, k], dtype, &device, &client).unwrap();
        let b_t = tensor_from_f64(&b, &[k], dtype, &device, &client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[1], dtype, &device, &client).unwrap();
        let res_t = tensor_from_f64(&residual, &[m, 1], dtype, &device, &client).unwrap();
        let relu = client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
            .expect("WebGPU rank-1 b matmul_bias_activation");
        let res = client
            .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
            .expect("WebGPU rank-1 b matmul_bias_residual");
        assert_tensor_allclose(&relu, &cpu_relu, dtype, "rank-1 b ReLU WebGPU vs CPU");
        assert_tensor_allclose(&res, &cpu_res, dtype, "rank-1 b residual WebGPU vs CPU");
    });
}

// ============================================================================
// Batch broadcast: `[2,m,k] @ [k,n]` and `[m,k] @ [2,k,n]`
// ============================================================================

/// F64 reference for one broadcast case, forward and backward.
///
/// `a_batch`/`b_batch` are 1 for a plain matrix and 2 for a batched operand;
/// the output batch is 2. Returns `(out, d_a, d_b, d_bias)` with `d_a`/`d_b`
/// summed back over a broadcast batch, as the gradient of a shared operand is.
struct BroadcastRef {
    out: Vec<f64>,
    d_a: Vec<f64>,
    d_b: Vec<f64>,
    d_bias: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn broadcast_reference(
    a: &[f64],
    a_batch: usize,
    b: &[f64],
    b_batch: usize,
    bias: &[f64],
    residual: Option<&[f64]>,
    grad: &[f64],
    relu: bool,
    m: usize,
    k: usize,
    n: usize,
) -> BroadcastRef {
    let batch = 2usize;
    let mut out = vec![0.0f64; batch * m * n];
    let mut d_a = vec![0.0f64; a_batch * m * k];
    let mut d_b = vec![0.0f64; b_batch * k * n];
    let mut d_bias = vec![0.0f64; n];
    for bi in 0..batch {
        let a_off = if a_batch == 1 { 0 } else { bi * m * k };
        let b_off = if b_batch == 1 { 0 } else { bi * k * n };
        for i in 0..m {
            for j in 0..n {
                let pre = (0..k)
                    .map(|p| a[a_off + i * k + p] * b[b_off + p * n + j])
                    .sum::<f64>()
                    + bias[j];
                let o = bi * m * n + i * n + j;
                out[o] = match residual {
                    Some(r) => pre + r[o],
                    None if relu => pre.max(0.0),
                    None => pre,
                };
                let deriv = if relu && pre <= 0.0 { 0.0 } else { 1.0 };
                let gp = grad[o] * deriv;
                d_bias[j] += gp;
                for p in 0..k {
                    d_a[a_off + i * k + p] += gp * b[b_off + p * n + j];
                    d_b[b_off + p * n + j] += a[a_off + i * k + p] * gp;
                }
            }
        }
    }
    BroadcastRef {
        out,
        d_a,
        d_b,
        d_bias,
    }
}

/// Values exactly representable in F16 so the F64 reference sees the same
/// operands the half run does.
fn quarter_grid(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 7 + seed * 3) % 9) as f64 * 0.25 - 1.0)
        .collect()
}

/// One broadcast layout on one dtype: activation forward + backward and
/// residual forward, CPU against the F64 reference, CUDA against CPU.
fn assert_gemm_epilogue_broadcast(a_batch: usize, b_batch: usize, dtype: numr::dtype::DType) {
    let (m, k, n) = (3usize, 4usize, 5usize);
    let a = quarter_grid(a_batch * m * k, 1);
    let b = quarter_grid(b_batch * k * n, 2);
    let bias = quarter_grid(n, 3);
    let residual = quarter_grid(2 * m * n, 4);
    let grad = quarter_grid(2 * m * n, 5);
    let a_shape: Vec<usize> = if a_batch == 1 {
        vec![m, k]
    } else {
        vec![a_batch, m, k]
    };
    let b_shape: Vec<usize> = if b_batch == 1 {
        vec![k, n]
    } else {
        vec![b_batch, k, n]
    };
    let out_shape = [2usize, m, n];
    let label = format!("gemm epilogue broadcast a={a_shape:?} b={b_shape:?} [{dtype:?}]");

    let act_ref = broadcast_reference(&a, a_batch, &b, b_batch, &bias, None, &grad, true, m, k, n);
    let res_ref = broadcast_reference(
        &a,
        a_batch,
        &b,
        b_batch,
        &bias,
        Some(&residual),
        &grad,
        false,
        m,
        k,
        n,
    );

    let (cpu_client, cpu_device) = create_cpu_client();
    let mk = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client).unwrap()
    };
    let (a_t, b_t, bias_t) = (mk(&a, &a_shape), mk(&b, &b_shape), mk(&bias, &[n]));
    let (res_t, grad_t) = (mk(&residual, &out_shape), mk(&grad, &out_shape));

    let cpu_act = cpu_client
        .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
        .unwrap_or_else(|e| panic!("CPU {label}: activation forward: {e}"));
    let cpu_res = cpu_client
        .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
        .unwrap_or_else(|e| panic!("CPU {label}: residual forward: {e}"));
    let (cpu_da, cpu_db, cpu_dbias) = cpu_client
        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, GemmActivation::ReLU)
        .unwrap_or_else(|e| panic!("CPU {label}: backward: {e}"));
    assert_eq!(cpu_act.shape(), &out_shape, "CPU {label}: activation shape");
    assert_eq!(cpu_res.shape(), &out_shape, "CPU {label}: residual shape");
    assert_eq!(cpu_da.shape(), a_shape.as_slice(), "CPU {label}: d_a shape");
    assert_eq!(cpu_db.shape(), b_shape.as_slice(), "CPU {label}: d_b shape");
    assert_eq!(cpu_dbias.shape(), &[n], "CPU {label}: d_bias shape");

    type CpuTensor = numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>;
    let checks: [(&CpuTensor, &[f64], &[usize], &str); 5] = [
        (&cpu_act, &act_ref.out, &out_shape, "activation forward"),
        (&cpu_res, &res_ref.out, &out_shape, "residual forward"),
        (&cpu_da, &act_ref.d_a, &a_shape, "d_a"),
        (&cpu_db, &act_ref.d_b, &b_shape, "d_b"),
        (&cpu_dbias, &act_ref.d_bias, &[n], "d_bias"),
    ];
    for (got, want, shape, what) in checks {
        let want_t = mk(want, shape);
        assert_tensor_allclose(got, &want_t, dtype, &format!("CPU {label}: {what} vs F64"));
    }

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|client, device| {
            let mk = |data: &[f64], shape: &[usize]| {
                tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
            };
            let (a_t, b_t, bias_t) = (mk(&a, &a_shape), mk(&b, &b_shape), mk(&bias, &[n]));
            let (res_t, grad_t) = (mk(&residual, &out_shape), mk(&grad, &out_shape));
            let act = client
                .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
                .unwrap_or_else(|e| panic!("CUDA {label}: activation forward: {e}"));
            let res = client
                .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                .unwrap_or_else(|e| panic!("CUDA {label}: residual forward: {e}"));
            let (da, db, dbias) = client
                .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, GemmActivation::ReLU)
                .unwrap_or_else(|e| panic!("CUDA {label}: backward: {e}"));
            assert_eq!(da.shape(), a_shape.as_slice(), "CUDA {label}: d_a shape");
            assert_eq!(db.shape(), b_shape.as_slice(), "CUDA {label}: d_b shape");
            assert_tensor_allclose(&act, &cpu_act, dtype, &format!("CUDA {label}: activation"));
            assert_tensor_allclose(&res, &cpu_res, dtype, &format!("CUDA {label}: residual"));
            assert_tensor_allclose(&da, &cpu_da, dtype, &format!("CUDA {label}: d_a"));
            assert_tensor_allclose(&db, &cpu_db, dtype, &format!("CUDA {label}: d_b"));
            assert_tensor_allclose(&dbias, &cpu_dbias, dtype, &format!("CUDA {label}: d_bias"));
        });
    }

    // WebGPU carries no batch broadcast: the broadcast layouts are refused
    // outright, and the fully batched one must match CPU, per-slice `d_b`
    // included.
    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|client, device| {
            let mk = |data: &[f64], shape: &[usize]| {
                tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
            };
            let (a_t, b_t, bias_t) = (mk(&a, &a_shape), mk(&b, &b_shape), mk(&bias, &[n]));
            let (res_t, grad_t) = (mk(&residual, &out_shape), mk(&grad, &out_shape));
            let act = client.matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU);
            let res = client.matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t);
            let bwd = client.matmul_bias_activation_bwd(
                &grad_t,
                &a_t,
                &b_t,
                &bias_t,
                GemmActivation::ReLU,
            );
            if a_batch != b_batch {
                assert!(act.is_err(), "WebGPU {label}: activation must refuse");
                assert!(res.is_err(), "WebGPU {label}: residual must refuse");
                assert!(bwd.is_err(), "WebGPU {label}: backward must refuse");
                return;
            }
            let act = act.unwrap_or_else(|e| panic!("WebGPU {label}: activation forward: {e}"));
            let res = res.unwrap_or_else(|e| panic!("WebGPU {label}: residual forward: {e}"));
            let (da, db, dbias) = bwd.unwrap_or_else(|e| panic!("WebGPU {label}: backward: {e}"));
            assert_tensor_allclose(
                &act,
                &cpu_act,
                dtype,
                &format!("WebGPU {label}: activation"),
            );
            assert_tensor_allclose(&res, &cpu_res, dtype, &format!("WebGPU {label}: residual"));
            assert_tensor_allclose(&da, &cpu_da, dtype, &format!("WebGPU {label}: d_a"));
            assert_tensor_allclose(&db, &cpu_db, dtype, &format!("WebGPU {label}: d_b"));
            assert_tensor_allclose(
                &dbias,
                &cpu_dbias,
                dtype,
                &format!("WebGPU {label}: d_bias"),
            );
        });
    }
}

/// `[2,m,k] @ [k,n]`, `[m,k] @ [2,k,n]` and the fully batched `[2,m,k] @
/// [2,k,n]`: every backend reads each operand at its own batch index, and a
/// batched `b` gets one gradient slice per batch rather than a sum in slice 0.
#[test]
fn gemm_epilogue_batch_broadcast_match_reference_and_cpu() {
    for dtype in [numr::dtype::DType::F32, numr::dtype::DType::F16] {
        if !is_dtype_supported("cpu", dtype) {
            continue;
        }
        assert_gemm_epilogue_broadcast(2, 1, dtype);
        assert_gemm_epilogue_broadcast(1, 2, dtype);
        assert_gemm_epilogue_broadcast(2, 2, dtype);
    }
}
