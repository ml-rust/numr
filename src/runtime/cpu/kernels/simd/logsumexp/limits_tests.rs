//! Direct native scalar, dispatched, and available ISA masked limits.

use crate::dtype::{DType, Element};

fn check<T: Element>(kernel: unsafe fn(*const T, *mut T, usize, usize)) {
    for width in [1, 2, 31, 32, 33, 64, 65, 128] {
        let mut masked = vec![T::from_f64(f64::NEG_INFINITY); width * 3];
        let mut output = vec![T::zero(); 3];
        for position in [0, width / 2, width - 1] {
            masked[width + position] = T::from_f64(f64::NAN);
            masked[width * 2 + width / 2] = T::from_f64(2.0);
            unsafe {
                kernel(masked.as_ptr(), output.as_mut_ptr(), width, 3);
            }
            assert_eq!(output[0].to_f64(), f64::NEG_INFINITY);
            assert!(output[1].to_f64().is_nan());
            assert_eq!(output[2].to_f64(), 2.0);
            masked[width + position] = T::from_f64(f64::NEG_INFINITY);
        }
        for value in [2.0, 1024.0] {
            let input = vec![T::from_f64(value); width];
            let mut result = [T::zero()];
            unsafe {
                kernel(input.as_ptr(), result.as_mut_ptr(), width, 1);
            }
            let expected = value + (width as f64).ln();
            let tolerance = if T::DTYPE == DType::F64 { 1e-12 } else { 1e-4 };
            assert!(
                (result[0].to_f64() - expected).abs() <= tolerance,
                "width={width},value={value},actual={},expected={expected}",
                result[0].to_f64()
            );
        }
    }
}

#[test]
fn native_scalar_and_dispatched_masked_limits() {
    check::<f32>(super::scalar::logsumexp_scalar_f32);
    check::<f64>(super::scalar::logsumexp_scalar_f64);
    check::<f32>(super::dispatch::logsumexp_f32);
    check::<f64>(super::dispatch::logsumexp_f64);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn available_x86_isas_preserve_masked_nan_and_finite_rows() {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        check::<f32>(super::avx2::logsumexp_f32);
        check::<f64>(super::avx2::logsumexp_f64);
    }
    if std::is_x86_feature_detected!("avx512f") {
        check::<f32>(super::avx512::logsumexp_f32);
        if std::is_x86_feature_detected!("avx512dq") {
            check::<f64>(super::avx512::logsumexp_f64);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_preserves_masked_nan_and_finite_rows() {
    check::<f32>(super::aarch64::neon::logsumexp_f32);
    check::<f64>(super::aarch64::neon::logsumexp_f64);
}
