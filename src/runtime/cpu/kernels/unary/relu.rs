//! ReLU activation kernel with automatic SIMD dispatch.

use crate::dtype::{DType, Element};

/// ReLU activation: max(0, x) with automatic SIMD dispatch
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::unary;

        match T::DTYPE {
            DType::F32 => {
                unary::relu_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                unary::relu_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                unary::relu_f16(a as *const half::f16, out as *mut half::f16, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                unary::relu_bf16(a as *const half::bf16, out as *mut half::bf16, len);
                return;
            }
            _ => {}
        }
    }

    relu_scalar(a, out, len);
}

/// Scalar ReLU for all Element types
#[inline]
unsafe fn relu_scalar<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        out_slice[i] = if a_slice[i] > zero { a_slice[i] } else { zero };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let a = [-1.0f32, 0.0, 1.0, -2.0];
        let mut out = [0.0f32; 4];

        unsafe {
            relu_kernel(a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [0.0, 0.0, 1.0, 0.0]);
    }
}
