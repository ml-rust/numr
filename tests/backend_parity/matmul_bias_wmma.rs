// Backend parity tests for the WMMA tensor-core epilogue in
// MatmulOps::matmul_bias (F16/BF16).
//
// `use_wmma` (src/runtime/cuda/kernels/loader/matmul_wmma_policy.rs) dispatches the
// fused bias-in-epilogue WMMA kernel for any m >= 1 whenever
// `caps.f16_mma`/`caps.bf16`. An N or K that is not a multiple of 8 stages
// that operand one element at a time; when the GEMM is heavy enough per
// copied element, src/ops/cuda/matmul.rs pads it (A, B, and the bias vector)
// to the next multiple of 8 first and slices back afterward. M is never
// padded.
// `matmul_bias.rs`'s existing tests all use 2x2 shapes, so none of them ever
// reached this path. These cases do: aligned sizes that dispatch straight to
// WMMA, a ragged block edge, sizes that force the padding path, each ragged
// stride class on its own, small m, batched broadcast at a WMMA-eligible
// size, and a bias-dominant case where a dropped or mis-indexed bias fails
// loudly.

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
use numr::ops::MatmulOps;

#[cfg(feature = "f16")]
fn deterministic_f64(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            ((i as f64 * 0.013 + phase).sin() * 0.5) + ((i as f64 * 0.0047 + phase).cos() * 0.3)
        })
        .collect()
}

/// Build CPU (reference) and, when CUDA is available, CUDA `matmul_bias`
/// results for the given dtype/shapes and assert they agree within
/// `tolerance_for_dtype`. Shared by every WMMA-epilogue case below.
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
fn assert_matmul_bias_wmma_parity(
    dtype: DType,
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
    bias_data: &[f64],
    bias_shape: &[usize],
    test_name: &str,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = tensor_from_f64(a_data, a_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let b_t = tensor_from_f64(b_data, b_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let bias_t = tensor_from_f64(bias_data, bias_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = tensor_from_f64(a_data, a_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let b_t = tensor_from_f64(b_data, b_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let bias_t =
                tensor_from_f64(bias_data, bias_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(&result, &cpu_result, dtype, test_name);
        });
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = test_name;
    }
}

/// Plain 2D case: deterministic A/B/bias at the given shape.
#[cfg(feature = "f16")]
fn assert_matmul_bias_wmma_2d(dtype: DType, m: usize, k: usize, n: usize, test_name: &str) {
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        dtype,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        &[n],
        test_name,
    );
}

// --- Case 1: aligned, reaches WMMA directly (M/N/K all 16-multiples) ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_aligned_128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        128,
        128,
        128,
        "matmul_bias_f16_wmma_aligned_128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_aligned_256x512x128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        256,
        128,
        512,
        "matmul_bias_f16_wmma_aligned_256x512x128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_aligned_128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        128,
        128,
        128,
        "matmul_bias_bf16_wmma_aligned_128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_aligned_256x512x128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        256,
        128,
        512,
        "matmul_bias_bf16_wmma_aligned_256x512x128 CUDA vs CPU",
    );
}

// --- Case 2: aligned but ragged against the 128x128 block tile: catches ---
// --- an out-of-range bias[col] read or a mishandled partial tile.         ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_partial_tile_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        144,
        144,
        144,
        "matmul_bias_f16_wmma_partial_tile CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_partial_tile_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        144,
        144,
        144,
        "matmul_bias_bf16_wmma_partial_tile CUDA vs CPU",
    );
}

// --- Case 3: N and K not multiples of 8 at shapes too light to pad:    ---
// --- scalar staging on both operands. M is ragged and never padded.    ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_padded_100_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        100,
        100,
        100,
        "matmul_bias_f16_wmma_padded_100 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_padded_130x70x50_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        130,
        50,
        70,
        "matmul_bias_f16_wmma_padded_130x70x50 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_padded_100_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        100,
        100,
        100,
        "matmul_bias_bf16_wmma_padded_100 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_padded_130x70x50_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        130,
        50,
        70,
        "matmul_bias_bf16_wmma_padded_130x70x50 CUDA vs CPU",
    );
}

// --- Case 4: small M, still reaches WMMA. `use_wmma` does not test M:    ---
// --- any m >= 1 dispatches to WMMA as it is.                             ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_small_m_aligned16_match_cpu() {
    // m == 16: one full row block.
    assert_matmul_bias_wmma_2d(
        DType::F16,
        16,
        64,
        64,
        "matmul_bias_f16_wmma_small_m_aligned16 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_small_m_padded8_match_cpu() {
    // m == 8: half a row block, masked by the kernel.
    assert_matmul_bias_wmma_2d(
        DType::F16,
        8,
        64,
        64,
        "matmul_bias_f16_wmma_small_m_padded8 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_small_m_aligned16_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        16,
        64,
        64,
        "matmul_bias_bf16_wmma_small_m_aligned16 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_small_m_padded8_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        8,
        64,
        64,
        "matmul_bias_bf16_wmma_small_m_padded8 CUDA vs CPU",
    );
}

// --- Case 5: batched with broadcast, at a WMMA-eligible size (M/N/K all  ---
// --- 16-multiples). Bias is indexed by global column only and           ---
// --- must broadcast across rows AND batch slices.                       ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_batched_a_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::F16,
        &a_data,
        &[1, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &[n],
        "matmul_bias_f16_wmma_batched_a_broadcast CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_batched_b_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::F16,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[1, k, n],
        &bias_data,
        &[n],
        "matmul_bias_f16_wmma_batched_b_broadcast CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_batched_a_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::BF16,
        &a_data,
        &[1, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &[n],
        "matmul_bias_bf16_wmma_batched_a_broadcast CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_batched_b_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::BF16,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[1, k, n],
        &bias_data,
        &[n],
        "matmul_bias_bf16_wmma_batched_b_broadcast CUDA vs CPU",
    );
}

// --- Case 6: bias dominates the result at a WMMA-eligible size (A/B tiny, ---
// --- bias large), so a dropped or mis-indexed bias fails loudly instead  ---
// --- of hiding under matmul noise.                                       ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_bias_dominant_match_cpu() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a_data: Vec<f64> = deterministic_f64(m * k, 0.0)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let b_data: Vec<f64> = deterministic_f64(k * n, 1.7)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let bias_data: Vec<f64> = deterministic_f64(n, 3.1)
        .iter()
        .map(|v| v * 50.0 + 100.0)
        .collect();
    assert_matmul_bias_wmma_parity(
        DType::F16,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        &[n],
        "matmul_bias_f16_wmma_bias_dominant CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_bias_dominant_match_cpu() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a_data: Vec<f64> = deterministic_f64(m * k, 0.0)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let b_data: Vec<f64> = deterministic_f64(k * n, 1.7)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let bias_data: Vec<f64> = deterministic_f64(n, 3.1)
        .iter()
        .map(|v| v * 50.0 + 100.0)
        .collect();
    assert_matmul_bias_wmma_parity(
        DType::BF16,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        &[n],
        "matmul_bias_bf16_wmma_bias_dominant CUDA vs CPU",
    );
}

// --- Case 7: m=1, single-token LLM decode. Launched as it is; K/N are   ---
// --- multiples of 8 so nothing pads.                                    ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_m1_decode_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        1,
        64,
        128,
        "matmul_bias_f16_wmma_m1_decode CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_m1_decode_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        1,
        64,
        128,
        "matmul_bias_bf16_wmma_m1_decode CUDA vs CPU",
    );
}

// --- Case 8: each ragged-stride class on its own. M is never a           ---
// --- condition; a ragged K or N pads only when the pad pass pays.        ---
// ---   m=37: ragged M only, no padding                                   ---
// ---   n=40: N ≡ 8 mod 16, no padding, scalar-staged N edge tile          ---
// ---   k=24: K ≡ 8 mod 16, no padding, scalar-staged K tail               ---
// ---   n=35 at m=64: N not a multiple of 8, too light to pad             ---
// ---   k=21 at m=64: K not a multiple of 8, too light to pad             ---
// ---   n=35 at m=520: heavy enough, pads N (and the bias) to 40           ---
// ---   k=21 at m=n=1040: heavy enough, pads K to 24 on both operands      ---

#[cfg(feature = "f16")]
fn assert_matmul_bias_wmma_ragged_strides(dtype: DType, tag: &str) {
    for (m, k, n, class) in [
        (37usize, 64usize, 128usize, "ragged_m"),
        (64, 64, 40, "n40"),
        (64, 24, 128, "k24"),
        (64, 64, 35, "n35_unpadded"),
        (64, 21, 128, "k21_unpadded"),
        (520, 64, 35, "n35_padded"),
        (1040, 21, 1040, "k21_padded"),
    ] {
        assert_matmul_bias_wmma_2d(
            dtype,
            m,
            k,
            n,
            &format!("matmul_bias_{tag}_wmma_{class} CUDA vs CPU"),
        );
    }
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_ragged_strides_match_cpu() {
    assert_matmul_bias_wmma_ragged_strides(DType::F16, "f16");
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_ragged_strides_match_cpu() {
    assert_matmul_bias_wmma_ragged_strides(DType::BF16, "bf16");
}

// --- Case 9: 3-D batched with ragged M and N ≡ 8 mod 16, through the      ---
// --- batched WMMA launcher as it is.                                     ---

#[cfg(feature = "f16")]
fn assert_matmul_bias_wmma_batched_ragged(dtype: DType, test_name: &str) {
    let (batch, m, k, n) = (3usize, 37usize, 64usize, 40usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        dtype,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &[n],
        test_name,
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_batched_ragged_m_n_match_cpu() {
    assert_matmul_bias_wmma_batched_ragged(
        DType::F16,
        "matmul_bias_f16_wmma_batched_ragged_m_n CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_batched_ragged_m_n_match_cpu() {
    assert_matmul_bias_wmma_batched_ragged(
        DType::BF16,
        "matmul_bias_bf16_wmma_batched_ragged_m_n CUDA vs CPU",
    );
}
