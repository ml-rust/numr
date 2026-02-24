//! SIMD-accelerated fused elementwise operations
//!
//! Provides vectorized implementations of:
//! - fused_mul_add: a * b + c (FMA)
//! - fused_add_mul: (a + b) * c
//! - fused_mul_add_scalar: a * scale + bias (affine transform)
//!
//! These use hardware FMA intrinsics where available for better accuracy
//! and throughput (single rounding instead of two).

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};

/// Minimum length to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

// ============================================================================
// fused_mul_add: a * b + c
// ============================================================================

/// SIMD fused_mul_add for f32: out[i] = a[i] * b[i] + c[i]
///
/// # Safety
/// - `a`, `b`, `c`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn fused_mul_add_f32(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
    len: usize,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_mul_add_scalar_f32(a, b, c, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::fused_mul_add_f32(a, b, c, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::fused_mul_add_f32(a, b, c, out, len),
        _ => fused_mul_add_scalar_f32(a, b, c, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::fused_mul_add_f32(a, b, c, out, len)
        }
        _ => fused_mul_add_scalar_f32(a, b, c, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_mul_add_scalar_f32(a, b, c, out, len);
}

/// SIMD fused_mul_add for f64: out[i] = a[i] * b[i] + c[i]
///
/// # Safety
/// - `a`, `b`, `c`, and `out` must point to `len` elements
#[inline]
pub unsafe fn fused_mul_add_f64(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    out: *mut f64,
    len: usize,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_mul_add_scalar_f64(a, b, c, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::fused_mul_add_f64(a, b, c, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::fused_mul_add_f64(a, b, c, out, len),
        _ => fused_mul_add_scalar_f64(a, b, c, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::fused_mul_add_f64(a, b, c, out, len)
        }
        _ => fused_mul_add_scalar_f64(a, b, c, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_mul_add_scalar_f64(a, b, c, out, len);
}

// ============================================================================
// fused_add_mul: (a + b) * c
// ============================================================================

/// SIMD fused_add_mul for f32: out[i] = (a[i] + b[i]) * c[i]
///
/// # Safety
/// - `a`, `b`, `c`, and `out` must point to `len` elements
#[inline]
pub unsafe fn fused_add_mul_f32(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
    len: usize,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_mul_scalar_f32(a, b, c, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::fused_add_mul_f32(a, b, c, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::fused_add_mul_f32(a, b, c, out, len),
        _ => fused_add_mul_scalar_f32(a, b, c, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::fused_add_mul_f32(a, b, c, out, len)
        }
        _ => fused_add_mul_scalar_f32(a, b, c, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_mul_scalar_f32(a, b, c, out, len);
}

/// SIMD fused_add_mul for f64: out[i] = (a[i] + b[i]) * c[i]
///
/// # Safety
/// - `a`, `b`, `c`, and `out` must point to `len` elements
#[inline]
pub unsafe fn fused_add_mul_f64(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    out: *mut f64,
    len: usize,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_mul_scalar_f64(a, b, c, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::fused_add_mul_f64(a, b, c, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::fused_add_mul_f64(a, b, c, out, len),
        _ => fused_add_mul_scalar_f64(a, b, c, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::fused_add_mul_f64(a, b, c, out, len)
        }
        _ => fused_add_mul_scalar_f64(a, b, c, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_mul_scalar_f64(a, b, c, out, len);
}

// ============================================================================
// fused_mul_add_scalar: a * scale + bias
// ============================================================================

/// SIMD fused_mul_add_scalar for f32: out[i] = a[i] * scale + bias
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn fused_mul_add_scalar_f32_kernel(
    a: *const f32,
    scale: f32,
    bias: f32,
    out: *mut f32,
    len: usize,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_mul_add_scalar_loop_f32(a, scale, bias, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::fused_mul_add_scalar_f32(a, scale, bias, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::fused_mul_add_scalar_f32(a, scale, bias, out, len),
        _ => fused_mul_add_scalar_loop_f32(a, scale, bias, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::fused_mul_add_scalar_f32(a, scale, bias, out, len)
        }
        _ => fused_mul_add_scalar_loop_f32(a, scale, bias, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_mul_add_scalar_loop_f32(a, scale, bias, out, len);
}

/// SIMD fused_mul_add_scalar for f64: out[i] = a[i] * scale + bias
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn fused_mul_add_scalar_f64_kernel(
    a: *const f64,
    scale: f64,
    bias: f64,
    out: *mut f64,
    len: usize,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_mul_add_scalar_loop_f64(a, scale, bias, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::fused_mul_add_scalar_f64(a, scale, bias, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::fused_mul_add_scalar_f64(a, scale, bias, out, len),
        _ => fused_mul_add_scalar_loop_f64(a, scale, bias, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::fused_mul_add_scalar_f64(a, scale, bias, out, len)
        }
        _ => fused_mul_add_scalar_loop_f64(a, scale, bias, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_mul_add_scalar_loop_f64(a, scale, bias, out, len);
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

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

#[inline]
pub unsafe fn fused_add_mul_scalar_f32(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = (*a.add(i) + *b.add(i)) * *c.add(i);
    }
}

#[inline]
pub unsafe fn fused_add_mul_scalar_f64(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    out: *mut f64,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = (*a.add(i) + *b.add(i)) * *c.add(i);
    }
}

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

// ============================================================================
// f16/bf16 block-convert-compute wrappers
// ============================================================================

/// Generate f16/bf16 wrappers for ternary fused ops: `fn(a, b, c, out, len)`
macro_rules! _half_ternary_fused {
    ($fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            a: *const $half_ty,
            b: *const $half_ty,
            c: *const $half_ty,
            out: *mut $half_ty,
            len: usize,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut b_buf = [0.0f32; HALF_BLOCK];
            let mut c_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $to_f32(b.add(offset) as *const u16, b_buf.as_mut_ptr(), chunk);
                $to_f32(c.add(offset) as *const u16, c_buf.as_mut_ptr(), chunk);
                $f32_fn(
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    c_buf.as_ptr(),
                    out_buf.as_mut_ptr(),
                    chunk,
                );
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
}

macro_rules! half_ternary_fused {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_ternary_fused!([<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_ternary_fused!([<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

half_ternary_fused!(fused_mul_add, fused_mul_add_f32);
half_ternary_fused!(fused_add_mul, fused_add_mul_f32);

/// Generate f16/bf16 wrappers for scalar fused ops: `fn(a, scale, bias, out, len)`
macro_rules! _half_scalar_fused {
    ($fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            a: *const $half_ty,
            scale: f32,
            bias: f32,
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
                $f32_fn(a_buf.as_ptr(), scale, bias, out_buf.as_mut_ptr(), chunk);
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
}

macro_rules! half_scalar_fused {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_scalar_fused!([<$name _f32_f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_scalar_fused!([<$name _f32_bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

half_scalar_fused!(fused_mul_add_scalar, fused_mul_add_scalar_f32_kernel);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_mul_add_f32() {
        let len = 128;
        let a: Vec<f32> = (0..len).map(|x| x as f32 * 0.1).collect();
        let b: Vec<f32> = (0..len).map(|x| x as f32 * 0.2 + 1.0).collect();
        let c: Vec<f32> = (0..len).map(|x| x as f32 * 0.05 - 0.5).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            fused_mul_add_f32(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), len);
            fused_mul_add_scalar_f32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                out_ref.as_mut_ptr(),
                len,
            );
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            assert!(
                diff < 1e-5,
                "fused_mul_add mismatch at {i}: {} vs {}",
                out[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_fused_add_mul_f32() {
        let len = 128;
        let a: Vec<f32> = (0..len).map(|x| x as f32 * 0.1).collect();
        let b: Vec<f32> = (0..len).map(|x| x as f32 * 0.2 + 1.0).collect();
        let c: Vec<f32> = (0..len).map(|x| x as f32 * 0.05 + 0.5).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            fused_add_mul_f32(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), len);
            fused_add_mul_scalar_f32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                out_ref.as_mut_ptr(),
                len,
            );
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            assert!(
                diff < 1e-5,
                "fused_add_mul mismatch at {i}: {} vs {}",
                out[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_fused_mul_add_scalar_f32() {
        let len = 128;
        let a: Vec<f32> = (0..len).map(|x| x as f32 * 0.1 - 5.0).collect();
        let scale = 2.5f32;
        let bias = -1.0f32;
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            fused_mul_add_scalar_f32_kernel(a.as_ptr(), scale, bias, out.as_mut_ptr(), len);
            fused_mul_add_scalar_loop_f32(a.as_ptr(), scale, bias, out_ref.as_mut_ptr(), len);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            assert!(
                diff < 1e-5,
                "fused_mul_add_scalar mismatch at {i}: {} vs {}",
                out[i],
                out_ref[i]
            );
        }
    }
}
