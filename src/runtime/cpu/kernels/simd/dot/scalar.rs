//! Scalar fallback and shared accumulator helpers for i8xi8 dot products.
//!
//! [`DOT_SPILL_ITERS`] and [`saturate_i64_to_i32`] are shared with every SIMD
//! backend (x86_64 AVX2/AVX-512, aarch64 NEON), which each accumulate in i32
//! SIMD lanes and periodically spill into the same i64 total this scalar path
//! uses directly.

/// SIMD iterations between spills of the i32 lane accumulator into an i64 total.
///
/// Every backend here accumulates products in i32 SIMD lanes, and every one of
/// them adds at most `2^16` to a single lane per iteration: a product of two i8
/// is bounded by `128 * 128 = 2^14`, and each lane receives at most four of them
/// per iteration. `2^14 * 2^16 = 2^30` stays inside i32, so spilling this often
/// is provably safe with a full bit of headroom.
///
/// Without a spill the lanes wrap after roughly a million elements and the dot
/// product returns a value with the wrong sign — silently, in release.
pub(in crate::runtime::cpu::kernels::simd) const DOT_SPILL_ITERS: usize = 16_384;

/// Narrow an exact i64 total to the i32 this op returns, clamping on overflow.
///
/// Saturating rather than wrapping, matching
/// [`crate::runtime::cpu::kernels::wide_acc`]: a wrapped total reports the
/// wrong sign and magnitude, while a clamped one is at least ordered correctly
/// and stays a total function.
#[inline]
pub(in crate::runtime::cpu::kernels::simd) fn saturate_i64_to_i32(acc: i64) -> i32 {
    acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Scalar fallback for i8 dot product.
///
/// Accumulates in i64 and clamps once at the end. An i32 accumulator wraps
/// after about 131k terms (`i32::MAX / 128^2`), which is well inside the sizes
/// a quantized matmul reaches along K. The i64 accumulator cannot overflow for
/// any reachable length: it would take `2^63 / 2^14` terms.
#[inline]
pub(super) unsafe fn i8xi8_dot_scalar(a: *const i8, b: *const i8, len: usize) -> i32 {
    let mut acc = 0i64;
    for i in 0..len {
        acc += (*a.add(i) as i64) * (*b.add(i) as i64);
    }
    saturate_i64_to_i32(acc)
}
