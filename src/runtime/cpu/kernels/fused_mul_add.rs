//! Architecture-neutral single-rounding F32/F64 multiply-add kernels.

/// Compute native-dtype fused multiply-add without rounding the intermediate product.
///
/// # Safety
/// Input pointers must address `len` initialized elements.
/// Output must address `len` writable elements without input overlap.
#[inline]
pub unsafe fn fused_mul_add_scalar_f32(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = (*a.add(i)).mul_add(*b.add(i), *c.add(i));
    }
}

/// Compute native-dtype fused multiply-add without rounding the intermediate product.
///
/// # Safety
/// Input pointers must address `len` initialized elements.
/// Output must address `len` writable elements without input overlap.
#[inline]
pub unsafe fn fused_mul_add_scalar_f64(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    out: *mut f64,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = (*a.add(i)).mul_add(*b.add(i), *c.add(i));
    }
}

/// Compute native-dtype fused multiply-add without rounding the intermediate product.
///
/// # Safety
/// Input pointers must address `len` initialized elements.
/// Output must address `len` writable elements without input overlap.
#[inline]
pub unsafe fn fused_mul_add_scalar_loop_f32(
    a: *const f32,
    scale: f32,
    bias: f32,
    out: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = (*a.add(i)).mul_add(scale, bias);
    }
}

/// Compute native-dtype fused multiply-add without rounding the intermediate product.
///
/// # Safety
/// Input pointers must address `len` initialized elements.
/// Output must address `len` writable elements without input overlap.
#[inline]
pub unsafe fn fused_mul_add_scalar_loop_f64(
    a: *const f64,
    scale: f64,
    bias: f64,
    out: *mut f64,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = (*a.add(i)).mul_add(scale, bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! check_fused_loops {
        ($name:ident, $ty:ty, $tensor:ident, $scalar:ident, $cases:expr) => {
            #[test]
            fn $name() {
                let cases: &[($ty, $ty, $ty, $ty)] = &$cases;
                for &(a, b, c, expected) in cases {
                    for len in [0, 1, 7, 31, 32, 33, 65] {
                        let a = vec![a; len];
                        let b_tensor = vec![b; len];
                        let c_tensor = vec![c; len];
                        let mut out = vec![0.0; len];
                        unsafe {
                            $tensor(a.as_ptr(), b_tensor.as_ptr(), c_tensor.as_ptr(), out.as_mut_ptr(), len);
                        }
                        assert!(
                            out.iter().all(|value| value.to_bits() == expected.to_bits()),
                            "{}: len={len}, a={a:?}, b={b:?}, c={c:?}, expected={expected:?}, out={out:?}",
                            stringify!($name),
                        );
                        unsafe {
                            $scalar(
                                a.as_ptr(), b, c,
                                out.as_mut_ptr(), len,
                            );
                        }
                        assert!(
                            out.iter().all(|value| value.to_bits() == expected.to_bits()),
                            "{}: len={len}, a={a:?}, b={b:?}, c={c:?}, expected={expected:?}, out={out:?}",
                            stringify!($name),
                        );
                    }
                }
            }
        };
    }

    // (1 + 2^-27) * (1 - 2^-27) - 1 = -2^-54 exactly.
    check_fused_loops!(
        fused_mul_add_portable_f64,
        f64,
        fused_mul_add_scalar_f64,
        fused_mul_add_scalar_loop_f64,
        [
            (
                f64::from_bits(0x3ff0000002000000),
                f64::from_bits(0x3feffffffc000000),
                -1.0,
                f64::from_bits(0xbc90000000000000),
            ),
            (f64::MAX, 2.0, -f64::MAX, f64::MAX),
            (f64::from_bits(1), 1.0, 0.0, f64::from_bits(1)),
            (f64::from_bits(1), 0.5, f64::from_bits(1), f64::from_bits(2)),
            (f64::from_bits(1), 0.5, 0.0, 0.0),
            (-f64::from_bits(1), 0.5, -0.0, -0.0),
            (-0.0, 1.0, -0.0, -0.0),
        ]
    );

    // (1 + 2^-13) * (1 - 2^-13) - 1 = -2^-26 exactly.
    check_fused_loops!(
        fused_mul_add_portable_f32,
        f32,
        fused_mul_add_scalar_f32,
        fused_mul_add_scalar_loop_f32,
        [
            (
                f32::from_bits(0x3f800400),
                f32::from_bits(0x3f7ff800),
                -1.0,
                f32::from_bits(0xb2800000),
            ),
            (f32::MAX, 2.0, -f32::MAX, f32::MAX),
            (f32::from_bits(1), 1.0, 0.0, f32::from_bits(1)),
            (f32::from_bits(1), 0.5, f32::from_bits(1), f32::from_bits(2)),
            (f32::from_bits(1), 0.5, 0.0, 0.0),
            (-f32::from_bits(1), 0.5, -0.0, -0.0),
            (-0.0, 1.0, -0.0, -0.0),
        ]
    );
}
