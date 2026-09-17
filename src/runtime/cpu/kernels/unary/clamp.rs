//! Clamp-to-range kernel with automatic SIMD dispatch.

use crate::dtype::{DType, Element};

/// Clamp values to a range: out[i] = min(max(a[i], min_val), max_val)
///
/// On x86-64, dispatches to SIMD for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn clamp_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    min_val: f64,
    max_val: f64,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::clamp;

        match T::DTYPE {
            DType::F32 => {
                clamp::clamp_f32(
                    a as *const f32,
                    out as *mut f32,
                    len,
                    min_val as f32,
                    max_val as f32,
                );
                return;
            }
            DType::F64 => {
                clamp::clamp_f64(a as *const f64, out as *mut f64, len, min_val, max_val);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                clamp::clamp_f16(
                    a as *const half::f16,
                    out as *mut half::f16,
                    len,
                    min_val as f32,
                    max_val as f32,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                clamp::clamp_bf16(
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                    min_val as f32,
                    max_val as f32,
                );
                return;
            }
            _ => {}
        }
    }

    clamp_scalar(a, out, len, min_val, max_val);
}

/// Scalar clamp for all Element types
#[inline]
unsafe fn clamp_scalar<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    min_val: f64,
    max_val: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let val = a_slice[i].to_f64();
        let clamped = if val < min_val {
            min_val
        } else if val > max_val {
            max_val
        } else {
            val
        };
        out_slice[i] = T::from_f64(clamped);
    }
}
