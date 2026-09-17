//! Generic unary-operation dispatch and scalar fallback.

use crate::dtype::{DType, Element};
use crate::ops::UnaryOp;

/// Execute a unary operation element-wise with automatic SIMD dispatch
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_op_kernel<T: Element>(op: UnaryOp, a: *const T, out: *mut T, len: usize) {
    // Complex types need component-wise operations, not magnitude-based to_f64/from_f64
    match T::DTYPE {
        DType::Complex64 => {
            super::complex::unary_op_complex64(op, a as *const f32, out as *mut f32, len);
            return;
        }
        DType::Complex128 => {
            super::complex::unary_op_complex128(op, a as *const f64, out as *mut f64, len);
            return;
        }
        _ => {}
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::unary;

        match T::DTYPE {
            DType::F32 => {
                unary::unary_f32(op, a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                unary::unary_f64(op, a as *const f64, out as *mut f64, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                unary::unary_f16(op, a as *const half::f16, out as *mut half::f16, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                unary::unary_bf16(op, a as *const half::bf16, out as *mut half::bf16, len);
                return;
            }
            _ => {}
        }
    }

    unary_op_scalar(op, a, out, len);
}

/// Scalar unary operation for all Element types
#[inline]
unsafe fn unary_op_scalar<T: Element>(op: UnaryOp, a: *const T, out: *mut T, len: usize) {
    // Integer neg, abs and sign WRAP and are computed in the element type, and
    // the rounding family is the exact identity there. The f64 round trip below
    // saturates and loses every bit past the 53rd, so it answers `i32::MAX` for
    // `neg(i32::MIN)` where CUDA and WebGPU both answer `i32::MIN`, and it does
    // not round-trip `floor(9007199254740993i64)`. See `int::unary_int_kernel`.
    if unsafe { super::int::unary_int_kernel(op, a, out, len) } {
        return;
    }

    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    match op {
        UnaryOp::Neg => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(-v);
            }
        }
        UnaryOp::Abs => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.abs());
            }
        }
        UnaryOp::Sqrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sqrt());
            }
        }
        UnaryOp::Exp => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp());
            }
        }
        UnaryOp::Log => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ln());
            }
        }
        UnaryOp::Sin => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sin());
            }
        }
        UnaryOp::Cos => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cos());
            }
        }
        UnaryOp::Tan => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tan());
            }
        }
        UnaryOp::Atan => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.atan());
            }
        }
        UnaryOp::Tanh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tanh());
            }
        }
        UnaryOp::Recip => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(1.0 / v);
            }
        }
        UnaryOp::Square => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v * v);
            }
        }
        UnaryOp::Floor => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.floor());
            }
        }
        UnaryOp::Ceil => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ceil());
            }
        }
        UnaryOp::Round => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.round());
            }
        }
        UnaryOp::RoundTiesEven => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.round_ties_even());
            }
        }
        UnaryOp::Sign => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                let sign = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out_slice[i] = T::from_f64(sign);
            }
        }
        UnaryOp::Rsqrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(1.0 / v.sqrt());
            }
        }
        UnaryOp::Cbrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cbrt());
            }
        }
        UnaryOp::Exp2 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp2());
            }
        }
        UnaryOp::Expm1 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp_m1());
            }
        }
        UnaryOp::Log2 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.log2());
            }
        }
        UnaryOp::Log10 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.log10());
            }
        }
        UnaryOp::Log1p => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ln_1p());
            }
        }
        UnaryOp::Asin => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.asin());
            }
        }
        UnaryOp::Acos => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.acos());
            }
        }
        UnaryOp::Sinh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sinh());
            }
        }
        UnaryOp::Cosh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cosh());
            }
        }
        UnaryOp::Asinh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.asinh());
            }
        }
        UnaryOp::Acosh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.acosh());
            }
        }
        UnaryOp::Atanh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.atanh());
            }
        }
        UnaryOp::Trunc => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.trunc());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_neg() {
        let a = [1.0f32, -2.0, 3.0, -4.0];
        let mut out = [0.0f32; 4];

        unsafe {
            unary_op_kernel(UnaryOp::Neg, a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_unary_sqrt() {
        let a = [1.0f32, 4.0, 9.0, 16.0];
        let mut out = [0.0f32; 4];

        unsafe {
            unary_op_kernel(UnaryOp::Sqrt, a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }
}
