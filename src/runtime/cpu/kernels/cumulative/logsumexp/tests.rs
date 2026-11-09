//! Direct generic fallback and strided masked-limit tests.

use super::{logsumexp_kernel, logsumexp_kernel_scalar, logsumexp_strided_kernel};
use crate::dtype::Element;

fn check<T: Element>() {
    for width in [1, 2, 31, 32, 33, 64, 65, 128] {
        for position in [0, width / 2, width - 1] {
            let mut input = vec![T::from_f64(f64::NEG_INFINITY); width * 3];
            input[width + position] = T::from_f64(f64::NAN);
            input[width * 2 + width / 2] = T::from_f64(2.0);
            let mut out = vec![T::zero(); 3];
            unsafe {
                logsumexp_kernel_scalar(input.as_ptr(), out.as_mut_ptr(), width, 3);
            }
            assert_eq!(out[0].to_f64(), f64::NEG_INFINITY);
            assert!(out[1].to_f64().is_nan());
            assert_eq!(out[2].to_f64(), 2.0);
            let mut strided = vec![T::from_f64(f64::NEG_INFINITY); width * 4];
            strided[position * 2 + 1] = T::from_f64(f64::NAN);
            strided[(width + width / 2) * 2 + 1] = T::from_f64(2.0);
            let mut out = vec![T::zero(); 4];
            unsafe {
                logsumexp_strided_kernel(strided.as_ptr(), out.as_mut_ptr(), width, 2, 2, 2, 2);
            }
            assert_eq!(out[0].to_f64(), f64::NEG_INFINITY);
            assert!(out[1].to_f64().is_nan());
            assert_eq!(out[2].to_f64(), f64::NEG_INFINITY);
            assert_eq!(out[3].to_f64(), 2.0);
        }
    }
}

#[test]
fn generic_scalar_and_strided_original_masked_limits() {
    check::<f32>();
    check::<f64>();
}

#[cfg(feature = "f16")]
#[test]
fn generic_half_scalar_and_strided_original_masked_limits() {
    check::<half::f16>();
    check::<half::bf16>();
}

#[test]
fn empty_reduction_and_empty_batch_preserve_identities() {
    let empty: [f64; 0] = [];
    let mut output = [0.0f64; 2];
    unsafe {
        logsumexp_kernel(empty.as_ptr(), output.as_mut_ptr(), 0, 2);
    }
    assert_eq!(output, [f64::NEG_INFINITY; 2]);
    output = [0.0; 2];
    unsafe {
        logsumexp_strided_kernel(empty.as_ptr(), output.as_mut_ptr(), 0, 1, 2, 2, 2);
    }
    assert_eq!(output, [f64::NEG_INFINITY; 2]);
    let mut untouched = [7.0f64];
    unsafe {
        logsumexp_kernel(empty.as_ptr(), untouched.as_mut_ptr(), 128, 0);
    }
    assert_eq!(untouched, [7.0]);
}
