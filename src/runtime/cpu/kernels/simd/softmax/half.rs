//! f16/bf16 softmax wrappers: convert to f32, run the f32 kernel, convert back.

#[cfg(feature = "f16")]
use super::dispatch::softmax_f32;

#[cfg(feature = "f16")]
/// f16 wrapper for softmax: processes one row at a time via f32 conversion.
///
/// # Safety
/// - `a` and `out` must point to `outer_size * dim_size` elements
pub unsafe fn softmax_f16(
    a: *const half::f16,
    out: *mut half::f16,
    outer_size: usize,
    dim_size: usize,
) {
    use crate::runtime::cpu::kernels::simd::half_convert_utils::*;
    let row_len = dim_size;
    let mut a_buf = vec![0.0f32; row_len];
    let mut out_buf = vec![0.0f32; row_len];
    for i in 0..outer_size {
        let offset = i * dim_size;
        convert_f16_to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), row_len);
        softmax_f32(a_buf.as_ptr(), out_buf.as_mut_ptr(), 1, dim_size);
        convert_f32_to_f16(out_buf.as_ptr(), out.add(offset) as *mut u16, row_len);
    }
}

#[cfg(feature = "f16")]
/// bf16 wrapper for softmax: processes one row at a time via f32 conversion.
///
/// # Safety
/// - `a` and `out` must point to `outer_size * dim_size` elements
pub unsafe fn softmax_bf16(
    a: *const half::bf16,
    out: *mut half::bf16,
    outer_size: usize,
    dim_size: usize,
) {
    use crate::runtime::cpu::kernels::simd::half_convert_utils::*;
    let row_len = dim_size;
    let mut a_buf = vec![0.0f32; row_len];
    let mut out_buf = vec![0.0f32; row_len];
    for i in 0..outer_size {
        let offset = i * dim_size;
        convert_bf16_to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), row_len);
        softmax_f32(a_buf.as_ptr(), out_buf.as_mut_ptr(), 1, dim_size);
        convert_f32_to_bf16(out_buf.as_ptr(), out.add(offset) as *mut u16, row_len);
    }
}
