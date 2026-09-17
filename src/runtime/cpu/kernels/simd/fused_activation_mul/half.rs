//! f16/bf16 block-convert-compute wrappers for fused activation-multiplication.

/// Generate f16/bf16 wrappers for binary fused ops: `fn(a, b, out, len)`
macro_rules! _half_binary_fused {
    ($fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            a: *const $half_ty,
            b: *const $half_ty,
            out: *mut $half_ty,
            len: usize,
        ) {
            use super::super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut b_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $to_f32(b.add(offset) as *const u16, b_buf.as_mut_ptr(), chunk);
                $f32_fn(a_buf.as_ptr(), b_buf.as_ptr(), out_buf.as_mut_ptr(), chunk);
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
}

macro_rules! half_binary_fused {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_binary_fused!([<$name _f16>], half::f16,
                super::super::half_convert_utils::convert_f16_to_f32,
                super::super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_binary_fused!([<$name _bf16>], half::bf16,
                super::super::half_convert_utils::convert_bf16_to_f32,
                super::super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

half_binary_fused!(silu_mul, super::dispatch::silu_mul_f32);
half_binary_fused!(gelu_mul, super::dispatch::gelu_mul_f32);
half_binary_fused!(relu_mul, super::dispatch::relu_mul_f32);
half_binary_fused!(sigmoid_mul, super::dispatch::sigmoid_mul_f32);
