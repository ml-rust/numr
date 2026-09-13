// Backend parity tests for the WMMA tensor-core epilogue in
// GemmEpilogueOps::matmul_bias_activation and ::matmul_bias_residual
// (F16/BF16).
//
// `use_wmma` (src/runtime/cuda/kernels/loader/matmul_wmma_policy.rs) selects the
// fused WMMA kernel for any m >= 1 whenever `caps.f16_mma`/`caps.bf16`. An N
// or K that is not a multiple of 8 stages that operand one element at a time;
// when the GEMM is heavy enough per copied element,
// src/ops/cuda/gemm_epilogue.rs pads it to the next multiple of 8 first — A,
// B and the bias vector in 1-D, and the residual in 2-D, since it is
// [M,N]-shaped — and slices back afterward. M is never padded. The existing gemm_epilogue
// tests are F32-only, so none of them reach this path. These cases do:
// aligned sizes that dispatch straight to WMMA, sizes that force the padding
// path, each ragged stride class on its own, small m, and batched shapes.

#[cfg(feature = "f16")]
use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(all(feature = "f16", feature = "cuda"))]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "f16")]
use crate::common::create_cpu_client;
#[cfg(all(feature = "f16", feature = "cuda"))]
use crate::common::{assert_tensor_allclose, is_dtype_supported};
#[cfg(feature = "f16")]
use numr::dtype::DType;
#[cfg(feature = "f16")]
use numr::ops::{GemmActivation, GemmEpilogueOps};

#[cfg(feature = "f16")]
fn deterministic_f64(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            ((i as f64 * 0.013 + phase).sin() * 0.5) + ((i as f64 * 0.0047 + phase).cos() * 0.3)
        })
        .collect()
}

/// Build CPU (reference) and, when CUDA is available, CUDA
/// `matmul_bias_activation` results and assert they agree within
/// `tolerance_for_dtype`.
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
fn assert_bias_act_parity(
    dtype: DType,
    activation: GemmActivation,
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
    bias_data: &[f64],
    test_name: &str,
) {
    let bias_shape = [bias_data.len()];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = tensor_from_f64(a_data, a_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let b_t = tensor_from_f64(b_data, b_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let bias_t = tensor_from_f64(bias_data, &bias_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let cpu_result = cpu_client
        .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = tensor_from_f64(a_data, a_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let b_t = tensor_from_f64(b_data, b_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let bias_t =
                tensor_from_f64(bias_data, &bias_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let result = cuda_client
                .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
                .unwrap();
            assert_tensor_allclose(&result, &cpu_result, dtype, test_name);
        });
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = test_name;
    }
}

/// Same for `matmul_bias_residual`. The residual is `[M,N]`-shaped (with the
/// leading batch dims of the output, when there are any).
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
fn assert_bias_residual_parity(
    dtype: DType,
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
    bias_data: &[f64],
    res_data: &[f64],
    res_shape: &[usize],
    test_name: &str,
) {
    let bias_shape = [bias_data.len()];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = tensor_from_f64(a_data, a_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let b_t = tensor_from_f64(b_data, b_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let bias_t = tensor_from_f64(bias_data, &bias_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let res_t = tensor_from_f64(res_data, res_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let cpu_result = cpu_client
        .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = tensor_from_f64(a_data, a_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let b_t = tensor_from_f64(b_data, b_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let bias_t =
                tensor_from_f64(bias_data, &bias_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let res_t =
                tensor_from_f64(res_data, res_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let result = cuda_client
                .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                .unwrap();
            assert_tensor_allclose(&result, &cpu_result, dtype, test_name);
        });
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = test_name;
    }
}

/// 2-D bias+activation case at the given shape.
#[cfg(feature = "f16")]
fn assert_bias_act_2d(
    dtype: DType,
    activation: GemmActivation,
    m: usize,
    k: usize,
    n: usize,
    test_name: &str,
) {
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_bias_act_parity(
        dtype,
        activation,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        test_name,
    );
}

/// 2-D bias+residual case at the given shape.
#[cfg(feature = "f16")]
fn assert_bias_residual_2d(dtype: DType, m: usize, k: usize, n: usize, test_name: &str) {
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    let res_data = deterministic_f64(m * n, 5.3);
    assert_bias_residual_parity(
        dtype,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        &res_data,
        &[m, n],
        test_name,
    );
}

#[cfg(feature = "f16")]
const ALL_ACTIVATIONS: [(GemmActivation, &str); 6] = [
    (GemmActivation::None, "none"),
    (GemmActivation::ReLU, "relu"),
    (GemmActivation::GELU, "gelu"),
    (GemmActivation::SiLU, "silu"),
    (GemmActivation::Sigmoid, "sigmoid"),
    (GemmActivation::Tanh, "tanh"),
];

// --- bias + activation: aligned, reaches WMMA directly ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_f16_wmma_aligned_128_all_activations_match_cpu() {
    for (activation, name) in ALL_ACTIVATIONS {
        assert_bias_act_2d(
            DType::F16,
            activation,
            128,
            128,
            128,
            &format!("gemm_bias_act_f16_wmma_aligned_128_{name} CUDA vs CPU"),
        );
    }
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_bf16_wmma_aligned_128_all_activations_match_cpu() {
    for (activation, name) in ALL_ACTIVATIONS {
        assert_bias_act_2d(
            DType::BF16,
            activation,
            128,
            128,
            128,
            &format!("gemm_bias_act_bf16_wmma_aligned_128_{name} CUDA vs CPU"),
        );
    }
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_f16_wmma_partial_tile_match_cpu() {
    // 16-aligned but not a whole 128x128 block: exercises the epilogue bounds
    // check on a ragged block edge.
    assert_bias_act_2d(
        DType::F16,
        GemmActivation::GELU,
        144,
        48,
        80,
        "gemm_bias_act_f16_wmma_partial_tile CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_bf16_wmma_partial_tile_match_cpu() {
    assert_bias_act_2d(
        DType::BF16,
        GemmActivation::SiLU,
        144,
        48,
        80,
        "gemm_bias_act_bf16_wmma_partial_tile CUDA vs CPU",
    );
}

// --- bias + activation: N and K not multiples of 8 at shapes too light ---
// --- to pad: scalar staging on both operands. M is ragged, never padded. ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_f16_wmma_padded_130x70x50_match_cpu() {
    for (activation, name) in ALL_ACTIVATIONS {
        assert_bias_act_2d(
            DType::F16,
            activation,
            130,
            70,
            50,
            &format!("gemm_bias_act_f16_wmma_padded_130x70x50_{name} CUDA vs CPU"),
        );
    }
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_bf16_wmma_padded_130x70x50_match_cpu() {
    for (activation, name) in ALL_ACTIVATIONS {
        assert_bias_act_2d(
            DType::BF16,
            activation,
            130,
            70,
            50,
            &format!("gemm_bias_act_bf16_wmma_padded_130x70x50_{name} CUDA vs CPU"),
        );
    }
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_f16_wmma_padded_100_match_cpu() {
    assert_bias_act_2d(
        DType::F16,
        GemmActivation::GELU,
        100,
        100,
        100,
        "gemm_bias_act_f16_wmma_padded_100 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_bf16_wmma_padded_100_match_cpu() {
    assert_bias_act_2d(
        DType::BF16,
        GemmActivation::GELU,
        100,
        100,
        100,
        "gemm_bias_act_bf16_wmma_padded_100 CUDA vs CPU",
    );
}

// --- bias + activation: small m, reaches WMMA as it is ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_f16_wmma_small_m_aligned16_match_cpu() {
    // m == 16, k/n aligned: dispatches to WMMA with no padding.
    assert_bias_act_2d(
        DType::F16,
        GemmActivation::ReLU,
        16,
        32,
        32,
        "gemm_bias_act_f16_wmma_small_m_aligned16 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_bf16_wmma_small_m_padded8_match_cpu() {
    // m == 8, k == 30, n == 20: ragged K and N, too light to pad.
    assert_bias_act_2d(
        DType::BF16,
        GemmActivation::ReLU,
        8,
        30,
        20,
        "gemm_bias_act_bf16_wmma_small_m_padded8 CUDA vs CPU",
    );
}

// --- bias + activation: m=1, single-token LLM decode. Launched as it is; ---
// --- K/N are multiples of 8 so nothing pads.                             ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_f16_wmma_m1_decode_match_cpu() {
    assert_bias_act_2d(
        DType::F16,
        GemmActivation::GELU,
        1,
        64,
        128,
        "gemm_bias_act_f16_wmma_m1_decode CUDA vs CPU",
    );
}

// --- bias + activation: batched ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_f16_wmma_batched_aligned_match_cpu() {
    let (batch, m, k, n) = (3usize, 64usize, 32usize, 48usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_bias_act_parity(
        DType::F16,
        GemmActivation::GELU,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        "gemm_bias_act_f16_wmma_batched_aligned CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_act_bf16_wmma_batched_aligned_match_cpu() {
    let (batch, m, k, n) = (3usize, 64usize, 32usize, 48usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_bias_act_parity(
        DType::BF16,
        GemmActivation::SiLU,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        "gemm_bias_act_bf16_wmma_batched_aligned CUDA vs CPU",
    );
}

// --- bias + residual: aligned, reaches WMMA directly ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_f16_wmma_aligned_128_match_cpu() {
    assert_bias_residual_2d(
        DType::F16,
        128,
        128,
        128,
        "gemm_bias_residual_f16_wmma_aligned_128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_bf16_wmma_aligned_128_match_cpu() {
    assert_bias_residual_2d(
        DType::BF16,
        128,
        128,
        128,
        "gemm_bias_residual_bf16_wmma_aligned_128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_f16_wmma_partial_tile_match_cpu() {
    assert_bias_residual_2d(
        DType::F16,
        144,
        48,
        80,
        "gemm_bias_residual_f16_wmma_partial_tile CUDA vs CPU",
    );
}

// --- bias + residual: N and K not multiples of 8 at shapes too light to ---
// --- pad. The 2-D residual pad spec is covered by the heavy ragged      ---
// --- cases at the end of this file.                                     ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_f16_wmma_padded_130x70x50_match_cpu() {
    assert_bias_residual_2d(
        DType::F16,
        130,
        70,
        50,
        "gemm_bias_residual_f16_wmma_padded_130x70x50 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_bf16_wmma_padded_130x70x50_match_cpu() {
    assert_bias_residual_2d(
        DType::BF16,
        130,
        70,
        50,
        "gemm_bias_residual_bf16_wmma_padded_130x70x50 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_f16_wmma_padded_100_match_cpu() {
    assert_bias_residual_2d(
        DType::F16,
        100,
        100,
        100,
        "gemm_bias_residual_f16_wmma_padded_100 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_bf16_wmma_padded_100_match_cpu() {
    assert_bias_residual_2d(
        DType::BF16,
        100,
        100,
        100,
        "gemm_bias_residual_bf16_wmma_padded_100 CUDA vs CPU",
    );
}

// --- bias + residual: small m, reaches WMMA as it is ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_f16_wmma_small_m_aligned16_match_cpu() {
    // m == 16, k/n aligned: dispatches to WMMA with no padding.
    assert_bias_residual_2d(
        DType::F16,
        16,
        32,
        32,
        "gemm_bias_residual_f16_wmma_small_m_aligned16 CUDA vs CPU",
    );
}

// --- bias + residual: batched ---

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_f16_wmma_batched_aligned_match_cpu() {
    let (batch, m, k, n) = (3usize, 64usize, 32usize, 48usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    let res_data = deterministic_f64(batch * m * n, 5.3);
    assert_bias_residual_parity(
        DType::F16,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &res_data,
        &[batch, m, n],
        "gemm_bias_residual_f16_wmma_batched_aligned CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_bias_residual_bf16_wmma_batched_aligned_match_cpu() {
    let (batch, m, k, n) = (3usize, 64usize, 32usize, 48usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    let res_data = deterministic_f64(batch * m * n, 5.3);
    assert_bias_residual_parity(
        DType::BF16,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &res_data,
        &[batch, m, n],
        "gemm_bias_residual_bf16_wmma_batched_aligned CUDA vs CPU",
    );
}

// --- ragged strides: each class on its own, bias+activation and           ---
// --- bias+residual. M is never a condition; a ragged K or N pads only     ---
// --- when the pad pass pays.                                              ---
// ---   m=37: ragged M only, no padding                                    ---
// ---   n=40: N ≡ 8 mod 16, no padding, scalar-staged N edge tile           ---
// ---   k=24: K ≡ 8 mod 16, no padding, scalar-staged K tail                ---
// ---   n=35 at m=64: N not a multiple of 8, too light to pad              ---
// ---   k=21 at m=64: K not a multiple of 8, too light to pad              ---
// ---   n=35 at m=520: pads N (bias and residual too) to 40                 ---
// ---   k=21 at m=n=1040: pads K to 24 on both operands                     ---
// A residual padded as if it were 1-D would shift every row, so the padded
// residual cases fail in the interior, not only at the sliced-off edge, if
// the padding spec is wrong.

#[cfg(feature = "f16")]
const RAGGED_STRIDE_SHAPES: [(usize, usize, usize, &str); 7] = [
    (37, 64, 128, "ragged_m"),
    (64, 64, 40, "n40"),
    (64, 24, 128, "k24"),
    (64, 64, 35, "n35_unpadded"),
    (64, 21, 128, "k21_unpadded"),
    (520, 64, 35, "n35_padded"),
    (1040, 21, 1040, "k21_padded"),
];

#[cfg(feature = "f16")]
fn assert_gemm_epilogue_wmma_ragged_strides(dtype: DType, tag: &str) {
    for (m, k, n, class) in RAGGED_STRIDE_SHAPES {
        assert_bias_act_2d(
            dtype,
            GemmActivation::GELU,
            m,
            k,
            n,
            &format!("gemm_bias_act_{tag}_wmma_{class} CUDA vs CPU"),
        );
        assert_bias_residual_2d(
            dtype,
            m,
            k,
            n,
            &format!("gemm_bias_residual_{tag}_wmma_{class} CUDA vs CPU"),
        );
    }
}

#[cfg(feature = "f16")]
#[test]
fn gemm_epilogue_f16_wmma_ragged_strides_match_cpu() {
    assert_gemm_epilogue_wmma_ragged_strides(DType::F16, "f16");
}

#[cfg(feature = "f16")]
#[test]
fn gemm_epilogue_bf16_wmma_ragged_strides_match_cpu() {
    assert_gemm_epilogue_wmma_ragged_strides(DType::BF16, "bf16");
}

// --- 3-D batched with ragged M and N ≡ 8 mod 16, through the batched WMMA ---
// --- launchers as it is.                                                  ---

#[cfg(feature = "f16")]
fn assert_gemm_epilogue_wmma_batched_ragged(dtype: DType, tag: &str) {
    let (batch, m, k, n) = (3usize, 37usize, 64usize, 40usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    let res_data = deterministic_f64(batch * m * n, 5.3);
    assert_bias_act_parity(
        dtype,
        GemmActivation::SiLU,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &format!("gemm_bias_act_{tag}_wmma_batched_ragged_m_n CUDA vs CPU"),
    );
    assert_bias_residual_parity(
        dtype,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &res_data,
        &[batch, m, n],
        &format!("gemm_bias_residual_{tag}_wmma_batched_ragged_m_n CUDA vs CPU"),
    );
}

#[cfg(feature = "f16")]
#[test]
fn gemm_epilogue_f16_wmma_batched_ragged_m_n_match_cpu() {
    assert_gemm_epilogue_wmma_batched_ragged(DType::F16, "f16");
}

#[cfg(feature = "f16")]
#[test]
fn gemm_epilogue_bf16_wmma_batched_ragged_m_n_match_cpu() {
    assert_gemm_epilogue_wmma_batched_ragged(DType::BF16, "bf16");
}
