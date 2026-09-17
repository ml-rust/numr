//! f16 / bf16 wrappers for softmax backward.
//!
//! Converts to f32, runs the f32 SIMD kernel, converts the result back.

#[cfg(feature = "f16")]
use super::dispatch::softmax_bwd_f32;

#[cfg(feature = "f16")]
/// f16 wrapper for softmax backward: processes one row at a time via f32 conversion.
///
/// # Safety
/// - All pointers must point to `outer_size * dim_size` elements
pub unsafe fn softmax_bwd_f16(
    grad: *const half::f16,
    output: *const half::f16,
    d_input: *mut half::f16,
    outer_size: usize,
    dim_size: usize,
) {
    use super::super::half_convert_utils::*;
    let row_len = dim_size;
    let mut grad_buf = vec![0.0f32; row_len];
    let mut out_buf = vec![0.0f32; row_len];
    let mut result_buf = vec![0.0f32; row_len];
    for i in 0..outer_size {
        let offset = i * dim_size;
        convert_f16_to_f32(
            grad.add(offset) as *const u16,
            grad_buf.as_mut_ptr(),
            row_len,
        );
        convert_f16_to_f32(
            output.add(offset) as *const u16,
            out_buf.as_mut_ptr(),
            row_len,
        );
        softmax_bwd_f32(
            grad_buf.as_ptr(),
            out_buf.as_ptr(),
            result_buf.as_mut_ptr(),
            1,
            dim_size,
        );
        convert_f32_to_f16(
            result_buf.as_ptr(),
            d_input.add(offset) as *mut u16,
            row_len,
        );
    }
}

#[cfg(feature = "f16")]
/// bf16 wrapper for softmax backward: processes one row at a time via f32 conversion.
///
/// # Safety
/// - All pointers must point to `outer_size * dim_size` elements
pub unsafe fn softmax_bwd_bf16(
    grad: *const half::bf16,
    output: *const half::bf16,
    d_input: *mut half::bf16,
    outer_size: usize,
    dim_size: usize,
) {
    use super::super::half_convert_utils::*;
    let row_len = dim_size;
    let mut grad_buf = vec![0.0f32; row_len];
    let mut out_buf = vec![0.0f32; row_len];
    let mut result_buf = vec![0.0f32; row_len];
    for i in 0..outer_size {
        let offset = i * dim_size;
        convert_bf16_to_f32(
            grad.add(offset) as *const u16,
            grad_buf.as_mut_ptr(),
            row_len,
        );
        convert_bf16_to_f32(
            output.add(offset) as *const u16,
            out_buf.as_mut_ptr(),
            row_len,
        );
        softmax_bwd_f32(
            grad_buf.as_ptr(),
            out_buf.as_ptr(),
            result_buf.as_mut_ptr(),
            1,
            dim_size,
        );
        convert_f32_to_bf16(
            result_buf.as_ptr(),
            d_input.add(offset) as *mut u16,
            row_len,
        );
    }
}
