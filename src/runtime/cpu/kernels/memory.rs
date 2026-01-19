//! Memory operation kernels (fill, copy, cast, random)

use crate::dtype::Element;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Fill buffer with a constant value
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn fill_kernel<T: Element>(out: *mut T, value: T, len: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    out_slice.fill(value);
}

/// Copy elements from src to dst
///
/// # Safety
/// - `src` and `dst` must be valid pointers to `len` elements
/// - `dst` must not overlap with `src`
#[inline]
pub unsafe fn copy_kernel<T: Element>(src: *const T, dst: *mut T, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

/// Cast tensor data from one dtype to another.
///
/// Converts elements by going through f64 as an intermediate representation,
/// which works for all numeric types via the Element trait.
///
/// # Safety
/// - `src` must be valid pointer to `len` elements of `src_dtype`
/// - `dst` must be valid pointer to `len` elements of `dst_dtype`
/// - `src` and `dst` must not overlap
#[inline]
pub unsafe fn cast_kernel(
    src: *const u8,
    dst: *mut u8,
    len: usize,
    src_dtype: crate::dtype::DType,
    dst_dtype: crate::dtype::DType,
) -> crate::error::Result<()> {
    use crate::dtype::DType;
    use crate::error::Error;

    // Helper macro to cast from a known source type to any destination type
    macro_rules! cast_from {
        ($src_ty:ty, $src_ptr:expr, $dst_ptr:expr, $len:expr, $dst_dtype:expr) => {{
            let src_slice = std::slice::from_raw_parts($src_ptr as *const $src_ty, $len);
            match $dst_dtype {
                DType::F64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut f64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64();
                    }
                }
                DType::F32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut f32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as f32;
                    }
                }
                DType::F16 => {
                    #[cfg(feature = "f16")]
                    {
                        let dst_slice =
                            std::slice::from_raw_parts_mut($dst_ptr as *mut half::f16, $len);
                        for i in 0..$len {
                            dst_slice[i] = half::f16::from_f64(src_slice[i].to_f64());
                        }
                    }
                    #[cfg(not(feature = "f16"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::F16,
                            op: "cast",
                        });
                    }
                }
                DType::BF16 => {
                    #[cfg(feature = "f16")]
                    {
                        let dst_slice =
                            std::slice::from_raw_parts_mut($dst_ptr as *mut half::bf16, $len);
                        for i in 0..$len {
                            dst_slice[i] = half::bf16::from_f64(src_slice[i].to_f64());
                        }
                    }
                    #[cfg(not(feature = "f16"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::BF16,
                            op: "cast",
                        });
                    }
                }
                DType::FP8E4M3 => {
                    #[cfg(feature = "fp8")]
                    {
                        let dst_slice = std::slice::from_raw_parts_mut(
                            $dst_ptr as *mut crate::dtype::FP8E4M3,
                            $len,
                        );
                        for i in 0..$len {
                            dst_slice[i] =
                                crate::dtype::FP8E4M3::from_f32(src_slice[i].to_f64() as f32);
                        }
                    }
                    #[cfg(not(feature = "fp8"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::FP8E4M3,
                            op: "cast",
                        });
                    }
                }
                DType::FP8E5M2 => {
                    #[cfg(feature = "fp8")]
                    {
                        let dst_slice = std::slice::from_raw_parts_mut(
                            $dst_ptr as *mut crate::dtype::FP8E5M2,
                            $len,
                        );
                        for i in 0..$len {
                            dst_slice[i] =
                                crate::dtype::FP8E5M2::from_f32(src_slice[i].to_f64() as f32);
                        }
                    }
                    #[cfg(not(feature = "fp8"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::FP8E5M2,
                            op: "cast",
                        });
                    }
                }
                DType::I64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i64;
                    }
                }
                DType::I32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i32;
                    }
                }
                DType::I16 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i16, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i16;
                    }
                }
                DType::I8 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i8, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i8;
                    }
                }
                DType::U64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u64;
                    }
                }
                DType::U32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u32;
                    }
                }
                DType::U16 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u16, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u16;
                    }
                }
                DType::U8 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u8, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u8;
                    }
                }
                DType::Bool => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u8, $len);
                    for i in 0..$len {
                        dst_slice[i] = if src_slice[i].to_f64() != 0.0 { 1 } else { 0 };
                    }
                }
            }
        }};
    }

    // Dispatch based on source dtype
    match src_dtype {
        DType::F64 => cast_from!(f64, src, dst, len, dst_dtype),
        DType::F32 => cast_from!(f32, src, dst, len, dst_dtype),
        DType::F16 => {
            #[cfg(feature = "f16")]
            {
                cast_from!(half::f16, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "f16"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::F16,
                    op: "cast",
                });
            }
        }
        DType::BF16 => {
            #[cfg(feature = "f16")]
            {
                cast_from!(half::bf16, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "f16"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::BF16,
                    op: "cast",
                });
            }
        }
        DType::FP8E4M3 => {
            #[cfg(feature = "fp8")]
            {
                cast_from!(crate::dtype::FP8E4M3, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "fp8"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::FP8E4M3,
                    op: "cast",
                });
            }
        }
        DType::FP8E5M2 => {
            #[cfg(feature = "fp8")]
            {
                cast_from!(crate::dtype::FP8E5M2, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "fp8"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::FP8E5M2,
                    op: "cast",
                });
            }
        }
        DType::I64 => cast_from!(i64, src, dst, len, dst_dtype),
        DType::I32 => cast_from!(i32, src, dst, len, dst_dtype),
        DType::I16 => cast_from!(i16, src, dst, len, dst_dtype),
        DType::I8 => cast_from!(i8, src, dst, len, dst_dtype),
        DType::U64 => cast_from!(u64, src, dst, len, dst_dtype),
        DType::U32 => cast_from!(u32, src, dst, len, dst_dtype),
        DType::U16 => cast_from!(u16, src, dst, len, dst_dtype),
        DType::U8 => cast_from!(u8, src, dst, len, dst_dtype),
        DType::Bool => {
            // Bool is stored as u8 (0 or 1)
            cast_from!(u8, src, dst, len, dst_dtype)
        }
    }

    Ok(())
}

/// Fill output with uniform random values in [0, 1)
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn rand_uniform_kernel<T: Element>(out: *mut T, len: usize) {
    let mut rng = rand::rng();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = rng.random();
        *elem = T::from_f64(val);
    }
}

/// Fill output with standard normal random values (mean=0, std=1)
///
/// Uses the Box-Muller transform for generating normally distributed values.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn rand_normal_kernel<T: Element>(out: *mut T, len: usize) {
    let mut rng = rand::rng();
    let normal = StandardNormal;
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = normal.sample(&mut rng);
        *elem = T::from_f64(val);
    }
}

/// Fill output with random integers in [low, high)
///
/// Generates uniformly distributed random integers.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `low < high` must be satisfied
#[inline]
pub unsafe fn randint_kernel<T: Element>(out: *mut T, low: i64, high: i64, len: usize) {
    use rand::distr::Uniform;
    use rand::prelude::Distribution;

    let mut rng = rand::rng();
    let dist = Uniform::new(low, high).unwrap();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: i64 = dist.sample(&mut rng);
        *elem = T::from_f64(val as f64);
    }
}

/// Fill output with evenly spaced values in [start, stop)
///
/// Generates values: start + step * i for i in 0..len
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn arange_kernel<T: Element>(out: *mut T, start: f64, step: f64, len: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for (i, elem) in out_slice.iter_mut().enumerate() {
        let val = start + step * (i as f64);
        *elem = T::from_f64(val);
    }
}

/// Fill output with evenly spaced values from start to stop (inclusive)
///
/// Generates values: start + (stop - start) * i / (steps - 1) for i in 0..steps
///
/// # Safety
/// - `out` must be a valid pointer to `steps` elements
/// - `steps` must be >= 2
#[inline]
pub unsafe fn linspace_kernel<T: Element>(out: *mut T, start: f64, stop: f64, steps: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, steps);

    if steps == 1 {
        out_slice[0] = T::from_f64(start);
        return;
    }

    let divisor = (steps - 1) as f64;
    let delta = stop - start;

    for (i, elem) in out_slice.iter_mut().enumerate() {
        let val = start + delta * (i as f64) / divisor;
        *elem = T::from_f64(val);
    }
}

/// Fill output with identity matrix (1s on diagonal, 0s elsewhere)
///
/// # Safety
/// - `out` must be a valid pointer to `n * m` elements
#[inline]
pub unsafe fn eye_kernel<T: Element>(out: *mut T, n: usize, m: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, n * m);

    // Fill with zeros first
    out_slice.fill(T::from_f64(0.0));

    // Set diagonal to 1
    let diag_len = n.min(m);
    for i in 0..diag_len {
        out_slice[i * m + i] = T::from_f64(1.0);
    }
}
