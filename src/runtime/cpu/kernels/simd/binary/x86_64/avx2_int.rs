//! AVX2 binary operation kernels for i32
//!
//! Processes 8 i32s per iteration using 256-bit vectors.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ops::BinaryOp;

const I32_LANES: usize = 8;

macro_rules! impl_binary_i32_avx2 {
    ($name:ident, $vec_op:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(a: *const i32, b: *const i32, out: *mut i32, chunks: usize) {
            for i in 0..chunks {
                let offset = i * I32_LANES;
                let va = _mm256_loadu_si256(a.add(offset) as *const __m256i);
                let vb = _mm256_loadu_si256(b.add(offset) as *const __m256i);
                let vr = $vec_op(va, vb);
                _mm256_storeu_si256(out.add(offset) as *mut __m256i, vr);
            }
        }
    };
}

impl_binary_i32_avx2!(binary_add_i32, _mm256_add_epi32);
impl_binary_i32_avx2!(binary_sub_i32, _mm256_sub_epi32);
impl_binary_i32_avx2!(binary_mul_i32, _mm256_mullo_epi32);
impl_binary_i32_avx2!(binary_max_i32, _mm256_max_epi32);
impl_binary_i32_avx2!(binary_min_i32, _mm256_min_epi32);

/// AVX2 binary operation for i32
///
/// # Safety
/// - CPU must support AVX2
/// - All pointers must be valid for `len` elements
#[target_feature(enable = "avx2")]
pub unsafe fn binary_i32(op: BinaryOp, a: *const i32, b: *const i32, out: *mut i32, len: usize) {
    let chunks = len / I32_LANES;
    let remainder = len % I32_LANES;

    match op {
        BinaryOp::Add => binary_add_i32(a, b, out, chunks),
        BinaryOp::Sub => binary_sub_i32(a, b, out, chunks),
        BinaryOp::Mul => binary_mul_i32(a, b, out, chunks),
        BinaryOp::Max => binary_max_i32(a, b, out, chunks),
        BinaryOp::Min => binary_min_i32(a, b, out, chunks),
        // Div, Pow, Atan2 have no integer SIMD — use scalar fallback
        _ => {
            super::super::binary_scalar_i32(op, a, b, out, len);
            return;
        }
    }

    // Handle tail with scalar
    if remainder > 0 {
        let offset = chunks * I32_LANES;
        super::super::binary_scalar_i32(
            op,
            a.add(offset),
            b.add(offset),
            out.add(offset),
            remainder,
        );
    }
}
