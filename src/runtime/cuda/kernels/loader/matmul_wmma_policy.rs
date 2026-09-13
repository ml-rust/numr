//! Dispatch policy for the tensor-core WMMA GEMM path.
//!
//! Decides which dtype/device/shape combinations take the WMMA launchers in
//! `matmul_wmma.rs`, and when a ragged row stride is worth padding first.
//! `use_wmma` is the single source of truth for the decision; the launcher
//! dispatch and the pre-launch padding decision in `src/ops/cuda/matmul.rs`
//! both derive from it.

use crate::dtype::DType;
use crate::runtime::traits::profile::DeviceCaps;

/// Halves moved by one 128-bit staging access in the WMMA kernels.
///
/// Mirrors `WMMA_VEC_HALVES` in `matmul_wmma_stage.cuh`. The kernel takes
/// its 128-bit staging path only when the operand row stride (K for A, N for
/// B) is a multiple of this value; otherwise every tile of that operand is
/// staged one element at a time. The two constants must change together: a
/// mismatch makes the host pad for a fast path the kernel never takes, or
/// skip padding the kernel needed.
pub(crate) const WMMA_STAGE_HALVES: usize = 8;

/// Ratio of GEMM multiply-adds (`m*n*k`) to elements the pad would copy at
/// which padding a ragged row stride pays.
///
/// Scalar staging costs a roughly fixed fraction of the GEMM's work, so it
/// scales with `m*n*k`. Padding costs one pass over each copied operand, plus
/// the narrow of the output. Below this ratio the unpadded launch wins; at or
/// above it the padded one does. Swept on device; tune it with a throughput
/// probe, not by reasoning.
pub(crate) const WMMA_PAD_MIN_WORK_PER_COPIED_ELEMENT: usize = 512;

/// True when this dtype has WMMA kernels on this device.
///
/// - F16 needs `caps.f16_mma` (sm_70+)
/// - BF16 needs `caps.bf16` (sm_80+; the BF16 WMMA kernels are compiled only
///   from sm_80, see `matmul_wmma.cu`)
#[inline]
fn wmma_dtype_ok(dtype: DType, caps: DeviceCaps) -> bool {
    match dtype {
        DType::F16 => caps.f16_mma,
        DType::BF16 => caps.bf16,
        _ => false,
    }
}

/// Elements the pad copies to bring every row stride to a multiple of
/// [`WMMA_STAGE_HALVES`]. Zero when neither stride is ragged.
///
/// A ragged N pads B only (`k*n`). A ragged K pads BOTH operands (`m*k` for
/// A, and `k*n` for B, whose row count changes). `None` on overflow, which
/// the callers read as "pad does not pay": a problem that large launches
/// unpadded rather than allocating.
#[inline]
fn wmma_pad_copied_elements(m: usize, n: usize, k: usize) -> Option<usize> {
    let kn = k.checked_mul(n)?;
    let mut copied = 0usize;
    if !n.is_multiple_of(WMMA_STAGE_HALVES) {
        copied = copied.checked_add(kn)?;
    }
    if !k.is_multiple_of(WMMA_STAGE_HALVES) {
        copied = copied.checked_add(m.checked_mul(k)?)?.checked_add(kn)?;
    }
    Some(copied)
}

/// True when a row stride is ragged AND the GEMM does at least
/// [`WMMA_PAD_MIN_WORK_PER_COPIED_ELEMENT`] multiply-adds per element the
/// pad would copy. Any overflow reads as false.
#[inline]
fn wmma_pad_pays(m: usize, n: usize, k: usize) -> bool {
    let Some(copied) = wmma_pad_copied_elements(m, n, k) else {
        return false;
    };
    if copied == 0 {
        return false;
    }
    let work = m.checked_mul(n).and_then(|mn| mn.checked_mul(k));
    let threshold = WMMA_PAD_MIN_WORK_PER_COPIED_ELEMENT.checked_mul(copied);
    match (work, threshold) {
        (Some(work), Some(threshold)) => work >= threshold,
        _ => false,
    }
}

/// Returns true when the WMMA path is taken for this dtype, device, and
/// shape as it is. This is the SINGLE source of truth for the decision: the
/// launcher dispatch in `matmul_wmma.rs` and the pre-launch padding decision
/// in `src/ops/cuda/matmul.rs` (via [`use_wmma_after_padding`]) both derive
/// from it.
///
/// The kernel is correct for any M, N, K >= 1: `WMMA_STAGE_TILE` zero-fills
/// past the M, N and K edges and the epilogue masks its store per element.
/// So the shape test here is a speed policy, not a correctness condition. A
/// row stride (K for A, N for B) that is not a multiple of
/// [`WMMA_STAGE_HALVES`] makes the kernel stage that operand one element at
/// a time. That still beats a pad copy unless the GEMM is heavy enough that
/// the scalar staging costs more than the copy pass ([`wmma_pad_pays`]). So
/// every shape launches WMMA directly, except the ones where padding pays;
/// those come back through [`use_wmma_after_padding`] with aligned strides.
/// M is never a condition: a ragged M costs nothing on the kernel.
#[inline]
pub(crate) fn use_wmma(dtype: DType, caps: DeviceCaps, m: usize, n: usize, k: usize) -> bool {
    wmma_dtype_ok(dtype, caps) && m >= 1 && !wmma_pad_pays(m, n, k)
}

/// `(n_pad, k_pad)`: N and K rounded up to the next multiple of
/// [`WMMA_STAGE_HALVES`]. M is never padded.
#[inline]
pub(crate) fn wmma_padded_dims(n: usize, k: usize) -> (usize, usize) {
    (
        n.next_multiple_of(WMMA_STAGE_HALVES),
        k.next_multiple_of(WMMA_STAGE_HALVES),
    )
}

/// Returns true when the op must pad N and K per [`wmma_padded_dims`] before
/// dispatch: the dtype has WMMA kernels on this device and padding pays
/// ([`wmma_pad_pays`]). Exactly the complement of [`use_wmma`] for a WMMA
/// dtype, so the padding decision in `src/ops/cuda/matmul.rs` cannot
/// disagree with the dispatch decision here (e.g. padding BF16 operands on a
/// device without `caps.bf16`, then not taking the WMMA path: a wasted
/// allocation and copy). The padded dims always satisfy `use_wmma`: their
/// strides are aligned, so nothing is left to pay for.
#[inline]
pub(crate) fn use_wmma_after_padding(
    dtype: DType,
    caps: DeviceCaps,
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    let pad = wmma_dtype_ok(dtype, caps) && m >= 1 && wmma_pad_pays(m, n, k);
    if pad {
        let (n_pad, k_pad) = wmma_padded_dims(n, k);
        debug_assert!(use_wmma(dtype, caps, m, n_pad, k_pad));
    }
    pad
}

#[cfg(test)]
mod tests {
    use super::*;

    // Turing (e.g. T4, RTX 20xx): f16 tensor cores, no native bf16 —
    // BF16 WMMA kernels are not even compiled for this arch (matmul_wmma.cu).
    fn turing_caps() -> DeviceCaps {
        DeviceCaps {
            dp4a: true,
            int8_mma: true,
            int8_mma_m16n8k32: false,
            f16_mma: true,
            bf16: false,
            fp8: false,
        }
    }

    // Ampere (e.g. A100, RTX 30xx): both f16 and native bf16 tensor cores.
    fn ampere_caps() -> DeviceCaps {
        DeviceCaps {
            dp4a: true,
            int8_mma: true,
            int8_mma_m16n8k32: true,
            f16_mma: true,
            bf16: true,
            fp8: true,
        }
    }

    fn no_caps() -> DeviceCaps {
        DeviceCaps::default()
    }

    // Ragged N, heavy enough that padding pays.
    const HEAVY_RAGGED_N: (usize, usize, usize) = (4104, 4099, 4096);
    // Ragged N, skinny: the scalar-staged B tiles cost less than copying B.
    const SKINNY_RAGGED_N: (usize, usize, usize) = (128, 57603, 2048);
    // Ragged K, heavy enough that padding both operands pays.
    const HEAVY_RAGGED_K: (usize, usize, usize) = (4096, 4096, 4097);

    // ---- dtype and caps ----

    #[test]
    fn turing_f16_aligned_uses_wmma() {
        assert!(use_wmma(DType::F16, turing_caps(), 32, 32, 32));
    }

    #[test]
    fn turing_bf16_aligned_does_not_use_wmma() {
        // BF16 WMMA symbols do not exist in the sm_75 cubin: requesting them
        // would be a missing-symbol launch failure, not a slow fallback.
        assert!(!use_wmma(DType::BF16, turing_caps(), 32, 32, 32));
    }

    #[test]
    fn ampere_f16_and_bf16_aligned_use_wmma() {
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 32, 32));
        assert!(use_wmma(DType::BF16, ampere_caps(), 32, 32, 32));
    }

    #[test]
    fn no_caps_never_uses_wmma() {
        assert!(!use_wmma(DType::F16, no_caps(), 32, 32, 32));
        assert!(!use_wmma(DType::BF16, no_caps(), 32, 32, 32));
    }

    #[test]
    fn non_f16_bf16_dtype_never_uses_wmma() {
        assert!(!use_wmma(DType::F32, ampere_caps(), 32, 32, 32));
    }

    // ---- shape policy ----

    #[test]
    fn ragged_m_uses_wmma() {
        // M is never a condition: the kernel zero-fills and masks the M edge.
        assert!(use_wmma(DType::F16, ampere_caps(), 33, 32, 32));
        assert!(use_wmma(DType::F16, ampere_caps(), 1, 32, 32));
        assert!(use_wmma(DType::F16, ampere_caps(), 8, 32, 32));
        assert!(!use_wmma(DType::F16, ampere_caps(), 0, 32, 32));
    }

    #[test]
    fn n_or_k_stage_multiple_but_not_16_uses_wmma() {
        // A row stride that is a multiple of WMMA_STAGE_HALVES keeps the
        // vector staging path, so nothing is ragged.
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 40, 32));
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 32, 24));
        assert!(use_wmma(DType::F16, ampere_caps(), 37, 40, 24));
        assert!(!use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            37,
            40,
            24
        ));
    }

    #[test]
    fn small_ragged_strides_launch_unpadded() {
        // Ragged N and K, but the GEMM is too light for a pad pass to pay.
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 33, 32));
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 32, 33));
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 35, 21));
        assert!(!use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            32,
            35,
            21
        ));
    }

    #[test]
    fn skinny_ragged_n_launches_unpadded() {
        let (m, n, k) = SKINNY_RAGGED_N;
        assert!(use_wmma(DType::F16, ampere_caps(), m, n, k));
        assert!(!use_wmma_after_padding(DType::F16, ampere_caps(), m, n, k));
    }

    #[test]
    fn heavy_ragged_n_pads() {
        let (m, n, k) = HEAVY_RAGGED_N;
        assert!(!use_wmma(DType::F16, ampere_caps(), m, n, k));
        assert!(use_wmma_after_padding(DType::F16, ampere_caps(), m, n, k));
    }

    #[test]
    fn heavy_ragged_k_pads() {
        let (m, n, k) = HEAVY_RAGGED_K;
        assert!(!use_wmma(DType::F16, ampere_caps(), m, n, k));
        assert!(use_wmma_after_padding(DType::F16, ampere_caps(), m, n, k));
    }

    #[test]
    fn copied_elements_count_each_padded_operand() {
        // Ragged N: B only.
        assert_eq!(wmma_pad_copied_elements(10, 35, 16), Some(16 * 35));
        // Ragged K: A and B.
        assert_eq!(
            wmma_pad_copied_elements(10, 32, 21),
            Some(10 * 21 + 21 * 32)
        );
        // Both ragged: B is counted once per ragged stride, as two copies
        // of it are never made but both dims move it.
        assert_eq!(
            wmma_pad_copied_elements(10, 35, 21),
            Some(21 * 35 + 10 * 21 + 21 * 35)
        );
        // Aligned: nothing copied.
        assert_eq!(wmma_pad_copied_elements(37, 40, 24), Some(0));
    }

    #[test]
    fn ragged_k_counts_both_operands() {
        // Square m == n == 1000. A ragged N copies B only, so the work per
        // copied element is m == 1000 and padding pays. A ragged K copies A
        // and B, halving that to m*n/(m+n) == 500, and padding does not pay.
        assert!(wmma_pad_pays(1000, 1003, 1000));
        assert!(!wmma_pad_pays(1000, 1000, 1003));
    }

    #[test]
    fn pad_threshold_is_inclusive() {
        // Ragged N only: work / copied == m exactly.
        let t = WMMA_PAD_MIN_WORK_PER_COPIED_ELEMENT;
        assert!(wmma_pad_pays(t, 35, 64));
        assert!(!wmma_pad_pays(t - 1, 35, 64));
    }

    #[test]
    fn huge_dims_do_not_pad() {
        // Overflow anywhere in the work or copy count reads as "does not
        // pay": launch unpadded rather than allocate.
        let huge = usize::MAX / 2;
        assert!(!wmma_pad_pays(huge, 35, huge));
        assert!(!wmma_pad_pays(huge, huge, 21));
        assert!(use_wmma(DType::F16, ampere_caps(), huge, 35, huge));
        assert!(!use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            huge,
            35,
            huge
        ));
    }

    #[test]
    fn padded_dims_round_to_stage_halves_never_16() {
        assert_eq!(wmma_padded_dims(35, 21), (40, 24));
        assert_eq!(wmma_padded_dims(40, 24), (40, 24));
        assert_eq!(wmma_padded_dims(1, 1), (8, 8));
        assert_eq!(wmma_padded_dims(33, 17), (40, 24));
    }

    #[test]
    fn padded_dims_always_satisfy_use_wmma() {
        for (m, n, k) in [HEAVY_RAGGED_N, HEAVY_RAGGED_K, (4104, 4099, 4097)] {
            assert!(use_wmma_after_padding(DType::F16, ampere_caps(), m, n, k));
            let (n_pad, k_pad) = wmma_padded_dims(n, k);
            assert!(use_wmma(DType::F16, ampere_caps(), m, n_pad, k_pad));
            assert!(!use_wmma_after_padding(
                DType::F16,
                ampere_caps(),
                m,
                n_pad,
                k_pad
            ));
        }
    }

    // ---- padding is caps-gated ----

    #[test]
    fn padding_turing_f16_reaches_wmma() {
        let (m, n, k) = HEAVY_RAGGED_N;
        assert!(use_wmma_after_padding(DType::F16, turing_caps(), m, n, k));
    }

    #[test]
    fn padding_turing_bf16_does_not_reach_wmma() {
        // Padding a BF16 operand on Turing must NOT be done: the WMMA path
        // never fires afterward (no caps.bf16), so padding would only cost
        // an allocation and a copy for nothing.
        let (m, n, k) = HEAVY_RAGGED_N;
        assert!(!use_wmma_after_padding(DType::BF16, turing_caps(), m, n, k));
        assert!(!use_wmma(DType::BF16, turing_caps(), m, n, k));
    }

    #[test]
    fn padding_ampere_bf16_reaches_wmma() {
        let (m, n, k) = HEAVY_RAGGED_N;
        assert!(use_wmma_after_padding(DType::BF16, ampere_caps(), m, n, k));
    }

    #[test]
    fn padding_no_caps_never_fires() {
        let (m, n, k) = HEAVY_RAGGED_N;
        assert!(!use_wmma_after_padding(DType::F16, no_caps(), m, n, k));
    }

    #[test]
    fn padding_already_aligned_dims_is_a_noop() {
        assert!(!use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            32,
            32,
            32
        ));
        assert!(!use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            4104,
            4096,
            4096
        ));
    }
}
