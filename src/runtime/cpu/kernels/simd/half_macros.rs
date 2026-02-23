//! Macros for generating f16/bf16 block-convert-compute wrappers.
//!
//! These macros eliminate boilerplate by generating both f16 and bf16 variants
//! of functions that operate via the block-convert-compute pattern:
//! 1. Convert half-precision input(s) to f32 in L1-sized stack blocks
//! 2. Call the existing f32 SIMD kernel
//! 3. Convert f32 output back to half-precision
//!
//! # Available Macros
//!
//! | Macro | Pattern | Example |
//! |-------|---------|---------|
//! | `half_unary!` | `fn(in, out, len)` | sigmoid, relu, erf |
//! | `half_unary_op!` | `fn(op, in, out, len)` | unary(UnaryOp) |
//! | `half_unary_param!` | `fn(in, out, len, p)` | leaky_relu, elu |
//! | `half_binary_op!` | `fn(op, a, b, out, len)` | binary, compare |
//! | `half_scalar_op!` | `fn(op, a, s, out, len)` | scalar ops |
//! | `half_unary_scalar!` | `fn(a, s, out, len)` | rsub_scalar |
//! | `half_where!` | `fn(cond, x, y, out, len)` | where_select |
//! | `half_clamp!` | `fn(a, out, len, min, max)` | clamp |

/// Internal: generate a single half-precision variant (f16 or bf16).
/// All public macros delegate to this to avoid duplicating the block-convert loop.
macro_rules! _half_variant {
    // 1-input, no extra args: fn(input, output, len)
    (unary, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(input: *const $half_ty, output: *mut $half_ty, len: usize) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(input.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $f32_fn(a_buf.as_ptr(), out_buf.as_mut_ptr(), chunk);
                $from_f32(out_buf.as_ptr(), output.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
    // 1-input with leading op: fn(op, input, output, len)
    (unary_op, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path, $op_ty:ty) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            op: $op_ty,
            input: *const $half_ty,
            output: *mut $half_ty,
            len: usize,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(input.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $f32_fn(op, a_buf.as_ptr(), out_buf.as_mut_ptr(), chunk);
                $from_f32(out_buf.as_ptr(), output.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
    // 1-input with trailing f32 param: fn(input, output, len, param)
    (unary_param, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            input: *const $half_ty,
            output: *mut $half_ty,
            len: usize,
            param: f32,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(input.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $f32_fn(a_buf.as_ptr(), out_buf.as_mut_ptr(), chunk, param);
                $from_f32(out_buf.as_ptr(), output.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
    // 2-input with leading op: fn(op, a, b, output, len)
    (binary_op, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path, $op_ty:ty) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            op: $op_ty,
            a: *const $half_ty,
            b: *const $half_ty,
            out: *mut $half_ty,
            len: usize,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut b_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $to_f32(b.add(offset) as *const u16, b_buf.as_mut_ptr(), chunk);
                $f32_fn(
                    op,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    out_buf.as_mut_ptr(),
                    chunk,
                );
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
    // 1-input with op + scalar: fn(op, a, scalar, output, len)
    (scalar_op, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path, $op_ty:ty) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            op: $op_ty,
            a: *const $half_ty,
            scalar: f32,
            out: *mut $half_ty,
            len: usize,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $f32_fn(op, a_buf.as_ptr(), scalar, out_buf.as_mut_ptr(), chunk);
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
    // 1-input with scalar (no op): fn(a, scalar, output, len)
    (unary_scalar, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(a: *const $half_ty, scalar: f32, out: *mut $half_ty, len: usize) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $f32_fn(a_buf.as_ptr(), scalar, out_buf.as_mut_ptr(), chunk);
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
    // where/select: fn(cond, x, y, output, len)
    (where_select, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            cond: *const u8,
            x: *const $half_ty,
            y: *const $half_ty,
            out: *mut $half_ty,
            len: usize,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut x_buf = [0.0f32; HALF_BLOCK];
            let mut y_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(x.add(offset) as *const u16, x_buf.as_mut_ptr(), chunk);
                $to_f32(y.add(offset) as *const u16, y_buf.as_mut_ptr(), chunk);
                $f32_fn(
                    cond.add(offset),
                    x_buf.as_ptr(),
                    y_buf.as_ptr(),
                    out_buf.as_mut_ptr(),
                    chunk,
                );
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
    // clamp: fn(a, output, len, min, max)
    (clamp, $fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            a: *const $half_ty,
            out: *mut $half_ty,
            len: usize,
            min_val: f32,
            max_val: f32,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $f32_fn(
                    a_buf.as_ptr(),
                    out_buf.as_mut_ptr(),
                    chunk,
                    min_val,
                    max_val,
                );
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
}

/// Generate f16/bf16 wrappers for unary: `fn(input, output, len)`
macro_rules! half_unary {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_variant!(unary, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_variant!(unary, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

/// Generate f16/bf16 wrappers for unary with leading op: `fn(op, input, output, len)`
macro_rules! half_unary_op {
    ($name:ident, $f32_fn:path, $op_ty:ty) => {
        paste::paste! {
            _half_variant!(unary_op, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn, $op_ty);
            _half_variant!(unary_op, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn, $op_ty);
        }
    };
}

/// Generate f16/bf16 wrappers for unary with trailing f32 param: `fn(input, output, len, param)`
macro_rules! half_unary_param {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_variant!(unary_param, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_variant!(unary_param, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

/// Generate f16/bf16 wrappers for binary with op: `fn(op, a, b, output, len)`
macro_rules! half_binary_op {
    ($name:ident, $f32_fn:path, $op_ty:ty) => {
        paste::paste! {
            _half_variant!(binary_op, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn, $op_ty);
            _half_variant!(binary_op, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn, $op_ty);
        }
    };
}

/// Generate f16/bf16 wrappers for scalar op: `fn(op, a, scalar, output, len)`
macro_rules! half_scalar_op {
    ($name:ident, $f32_fn:path, $op_ty:ty) => {
        paste::paste! {
            _half_variant!(scalar_op, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn, $op_ty);
            _half_variant!(scalar_op, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn, $op_ty);
        }
    };
}

/// Generate f16/bf16 wrappers for simple scalar fn: `fn(a, scalar, output, len)`
macro_rules! half_unary_scalar {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_variant!(unary_scalar, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_variant!(unary_scalar, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

/// Generate f16/bf16 wrappers for where/select: `fn(cond, x, y, output, len)`
macro_rules! half_where {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_variant!(where_select, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_variant!(where_select, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

/// Generate f16/bf16 wrappers for clamp: `fn(a, output, len, min, max)`
macro_rules! half_clamp {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_variant!(clamp, [<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_variant!(clamp, [<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}
