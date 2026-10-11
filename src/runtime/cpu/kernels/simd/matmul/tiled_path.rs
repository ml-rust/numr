//! Which shapes take the tiled matmul path.
//!
//! These queries are architecture-neutral. Callers on every target read them,
//! including targets that compile no SIMD matmul kernel.

use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// Small matrix threshold - below this, register-blocked SIMD is faster than tiled
pub(super) const SMALL_MATRIX_THRESHOLD: usize = 128 * 128 * 128 + 1;

/// Do the transposed-B entry points take the tiled path for this shape?
///
/// The tiled path is the one that packs B panels straight out of the `[N, K]`
/// buffer, producing panels byte-identical to a materialized `[K, N]` operand —
/// so on that path, and only on that path, `matmul_bt_f32` and `matmul_f32`
/// agree bit for bit. Anywhere else the transposed operand runs a different
/// kernel (the dot-product GEMV-BT one), which agrees within tolerance but
/// accumulates in its own order. Callers that require identity gate on this.
pub fn matmul_bt_is_tiled(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(n).saturating_mul(k) >= SMALL_MATRIX_THRESHOLD
        && tiled_level_available(detect_simd())
}

/// Fewest output columns that keep an `m × n × k` product on the tiled path.
///
/// A caller splitting N across threads uses this as its chunk floor. A chunk
/// narrower than this drops below `SMALL_MATRIX_THRESHOLD` and runs the
/// small-matrix kernel instead, which both accumulates in a different order
/// and — for a transposed B — is a *different* kernel from the contiguous
/// case, breaking the bit-for-bit agreement [`matmul_bt_is_tiled`] promises.
///
/// Returns `usize::MAX` when `m * k` is zero, so no split can clear the floor.
#[cfg(feature = "rayon")]
pub fn min_tiled_columns(m: usize, k: usize) -> usize {
    let rows = m.saturating_mul(k);
    if rows == 0 {
        return usize::MAX;
    }
    SMALL_MATRIX_THRESHOLD.div_ceil(rows)
}

/// Does this SIMD level have a tiled microkernel, rather than the scalar path?
#[cfg(target_arch = "x86_64")]
fn tiled_level_available(level: SimdLevel) -> bool {
    matches!(level, SimdLevel::Avx512 | SimdLevel::Avx2Fma)
}

/// Does this SIMD level have a tiled microkernel, rather than the scalar path?
#[cfg(target_arch = "aarch64")]
fn tiled_level_available(level: SimdLevel) -> bool {
    matches!(level, SimdLevel::Neon | SimdLevel::NeonFp16)
}

/// Does this SIMD level have a tiled microkernel, rather than the scalar path?
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn tiled_level_available(_level: SimdLevel) -> bool {
    false
}
