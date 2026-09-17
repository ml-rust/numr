//! f16 / bf16 wrappers for reductions.
//!
//! Converts to f32, runs the f32 SIMD reduction, converts the result back.

#[cfg(feature = "f16")]
use super::dispatch::reduce_f32;
#[cfg(feature = "f16")]
use crate::ops::ReduceOp;

#[cfg(feature = "f16")]
/// f16 wrapper for reduce: converts input to f32, runs f32 reduce, converts output back.
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
pub unsafe fn reduce_f16(
    op: ReduceOp,
    a: *const half::f16,
    out: *mut half::f16,
    reduce_size: usize,
    outer_size: usize,
) {
    use super::super::half_convert_utils::*;
    let input_len = outer_size * reduce_size;
    let mut a_f32 = vec![0.0f32; input_len];
    let mut out_f32 = vec![0.0f32; outer_size];
    convert_f16_to_f32(a as *const u16, a_f32.as_mut_ptr(), input_len);
    reduce_f32(
        op,
        a_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        reduce_size,
        outer_size,
    );
    convert_f32_to_f16(out_f32.as_ptr(), out as *mut u16, outer_size);
}

#[cfg(feature = "f16")]
/// bf16 wrapper for reduce: converts input to f32, runs f32 reduce, converts output back.
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
pub unsafe fn reduce_bf16(
    op: ReduceOp,
    a: *const half::bf16,
    out: *mut half::bf16,
    reduce_size: usize,
    outer_size: usize,
) {
    use super::super::half_convert_utils::*;
    let input_len = outer_size * reduce_size;
    let mut a_f32 = vec![0.0f32; input_len];
    let mut out_f32 = vec![0.0f32; outer_size];
    convert_bf16_to_f32(a as *const u16, a_f32.as_mut_ptr(), input_len);
    reduce_f32(
        op,
        a_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        reduce_size,
        outer_size,
    );
    convert_f32_to_bf16(out_f32.as_ptr(), out as *mut u16, outer_size);
}
