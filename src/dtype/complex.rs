//! Complex number types for FFT and other signal processing operations
//!
//! This module provides Complex64 and Complex128 types that are compatible
//! with bytemuck for zero-copy conversions and implement the Element trait
//! for tensor operations.
//!
//! # Storage Format
//!
//! Complex numbers are stored in interleaved format (re, im, re, im...),
//! matching numpy, FFTW, and cuFFT conventions.
//!
//! # Arithmetic Operations
//!
//! Complex arithmetic follows standard mathematical definitions:
//! - Addition: `(a+bi) + (c+di) = (a+c) + (b+d)i`
//! - Subtraction: `(a+bi) - (c+di) = (a-c) + (b-d)i`
//! - Multiplication: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
//! - Division: `(a+bi)/(c+di) = (a+bi)*conj(c+di)/|c+di|²`
//!
//! # Examples
//!
//! ```ignore
//! use numr::dtype::complex::Complex64;
//!
//! let z = Complex64::new(3.0, 4.0);
//! assert_eq!(z.magnitude(), 5.0);  // |z| = sqrt(3² + 4²) = 5
//!
//! let w = Complex64::new(1.0, 2.0);
//! let product = z * w;  // Complex multiplication
//! let conjugate = z.conj();  // 3 - 4i
//! ```

use bytemuck::{Pod, Zeroable};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// CUDA Compatibility Traits
// ============================================================================

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;
#[cfg(feature = "cuda")]
use cudarc::types::CudaTypeName;

/// Macro to implement complex number type with all operations
///
/// This avoids code duplication between Complex64 and Complex128.
macro_rules! impl_complex {
    (
        $name:ident,
        $float:ty,
        $doc_bits:literal,
        $doc_float_bits:literal,
        $doc_gpu_type:literal
    ) => {
        #[doc = concat!($doc_bits, "-bit complex number with ", $doc_float_bits, " real and imaginary parts")]
        ///
        #[doc = concat!("Memory layout: ", stringify!($name), " is ", stringify!($float), " × 2, interleaved format.")]
        #[doc = concat!("This matches the layout expected by ", $doc_gpu_type, ".")]
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
        pub struct $name {
            /// Real part
            pub re: $float,
            /// Imaginary part
            pub im: $float,
        }

        impl $name {
            /// Zero complex number
            pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

            /// One (real unit)
            pub const ONE: Self = Self { re: 1.0, im: 0.0 };

            /// Imaginary unit i
            pub const I: Self = Self { re: 0.0, im: 1.0 };

            /// Create a new complex number
            #[inline]
            pub const fn new(re: $float, im: $float) -> Self {
                Self { re, im }
            }

            /// Create a complex number from polar form: r * e^(iθ)
            #[inline]
            pub fn from_polar(r: $float, theta: $float) -> Self {
                Self {
                    re: r * theta.cos(),
                    im: r * theta.sin(),
                }
            }

            /// Magnitude (absolute value): |z| = sqrt(re² + im²)
            #[inline]
            pub fn magnitude(self) -> $float {
                (self.re * self.re + self.im * self.im).sqrt()
            }

            /// Squared magnitude: |z|² = re² + im²
            ///
            /// More efficient than `magnitude()` when you only need the squared value.
            #[inline]
            pub fn magnitude_squared(self) -> $float {
                self.re * self.re + self.im * self.im
            }

            /// Phase angle (argument): atan2(im, re)
            ///
            /// Returns the angle in radians from the positive real axis.
            #[inline]
            pub fn phase(self) -> $float {
                self.im.atan2(self.re)
            }

            /// Complex conjugate: conj(a + bi) = a - bi
            #[inline]
            pub fn conj(self) -> Self {
                Self {
                    re: self.re,
                    im: -self.im,
                }
            }

            /// Reciprocal: 1/z = conj(z)/|z|²
            #[inline]
            pub fn recip(self) -> Self {
                let mag_sq = self.magnitude_squared();
                if mag_sq == 0.0 {
                    Self {
                        re: <$float>::INFINITY,
                        im: <$float>::INFINITY,
                    }
                } else {
                    Self {
                        re: self.re / mag_sq,
                        im: -self.im / mag_sq,
                    }
                }
            }

            /// Complex exponential: e^z = e^re * (cos(im) + i*sin(im))
            #[inline]
            pub fn exp(self) -> Self {
                let exp_re = self.re.exp();
                Self {
                    re: exp_re * self.im.cos(),
                    im: exp_re * self.im.sin(),
                }
            }

            /// Natural logarithm: ln(z) = ln(|z|) + i*arg(z)
            #[inline]
            pub fn ln(self) -> Self {
                Self {
                    re: self.magnitude().ln(),
                    im: self.phase(),
                }
            }

            /// Square root using principal branch
            #[inline]
            pub fn sqrt(self) -> Self {
                let mag = self.magnitude();
                if mag == 0.0 {
                    Self::ZERO
                } else {
                    let re = ((mag + self.re) / 2.0).sqrt();
                    let im = self.im.signum() * ((mag - self.re) / 2.0).sqrt();
                    Self { re, im }
                }
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self {
                    re: self.re + rhs.re,
                    im: self.im + rhs.im,
                }
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self {
                    re: self.re - rhs.re,
                    im: self.im - rhs.im,
                }
            }
        }

        impl Mul for $name {
            type Output = Self;

            /// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self {
                    re: self.re * rhs.re - self.im * rhs.im,
                    im: self.re * rhs.im + self.im * rhs.re,
                }
            }
        }

        impl Div for $name {
            type Output = Self;

            /// Complex division: (a+bi)/(c+di) = (a+bi)*conj(c+di)/|c+di|²
            #[inline]
            fn div(self, rhs: Self) -> Self {
                let denom = rhs.magnitude_squared();
                if denom == 0.0 {
                    Self {
                        re: <$float>::NAN,
                        im: <$float>::NAN,
                    }
                } else {
                    Self {
                        re: (self.re * rhs.re + self.im * rhs.im) / denom,
                        im: (self.im * rhs.re - self.re * rhs.im) / denom,
                    }
                }
            }
        }

        impl Neg for $name {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                Self {
                    re: -self.re,
                    im: -self.im,
                }
            }
        }

        impl PartialOrd for $name {
            /// Complex numbers are not naturally ordered.
            /// This compares by magnitude for sorting purposes.
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.magnitude().partial_cmp(&other.magnitude())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.im >= 0.0 {
                    write!(f, "{}+{}i", self.re, self.im)
                } else {
                    write!(f, "{}{}i", self.re, self.im)
                }
            }
        }

        impl From<$float> for $name {
            #[inline]
            fn from(re: $float) -> Self {
                Self { re, im: 0.0 }
            }
        }

        impl From<($float, $float)> for $name {
            #[inline]
            fn from((re, im): ($float, $float)) -> Self {
                Self { re, im }
            }
        }
    };
}

// Generate Complex64 and Complex128 using the macro
impl_complex!(
    Complex64,
    f32,
    "64",
    "f32",
    "CUDA float2 and WebGPU vec2<f32>"
);
impl_complex!(Complex128, f64, "128", "f64", "CUDA double2");

// ============================================================================
// CUDA Trait Implementations
// ============================================================================

/// Complex64 maps to CUDA's float2 (two 32-bit floats in interleaved format)
#[cfg(feature = "cuda")]
impl CudaTypeName for Complex64 {
    const NAME: &'static str = "float2";
}

/// Complex128 maps to CUDA's double2 (two 64-bit doubles in interleaved format)
#[cfg(feature = "cuda")]
impl CudaTypeName for Complex128 {
    const NAME: &'static str = "double2";
}

/// SAFETY: Complex64 is #[repr(C)] with two f32 fields, which matches CUDA float2 layout.
/// The type is Pod and Zeroable, ensuring safe memory representation for GPU transfers.
#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for Complex64 {}

/// SAFETY: Complex128 is #[repr(C)] with two f64 fields, which matches CUDA double2 layout.
/// The type is Pod and Zeroable, ensuring safe memory representation for GPU transfers.
#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for Complex128 {}

// ============================================================================
// Conversion between complex types (cannot be in macro due to cross-type refs)
// ============================================================================

impl From<Complex64> for Complex128 {
    #[inline]
    fn from(c: Complex64) -> Self {
        Self {
            re: c.re as f64,
            im: c.im as f64,
        }
    }
}

impl From<Complex128> for Complex64 {
    #[inline]
    fn from(c: Complex128) -> Self {
        Self {
            re: c.re as f32,
            im: c.im as f32,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Macro to generate tests for both Complex64 and Complex128
    macro_rules! test_complex_type {
        ($mod_name:ident, $type_name:ident, $float:ty, $pi:expr) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn test_basic() {
                    let z = $type_name::new(3.0, 4.0);
                    assert_eq!(z.re, 3.0);
                    assert_eq!(z.im, 4.0);
                    assert_eq!(z.magnitude(), 5.0);
                    assert_eq!(z.magnitude_squared(), 25.0);
                }

                #[test]
                fn test_arithmetic() {
                    let a = $type_name::new(1.0, 2.0);
                    let b = $type_name::new(3.0, 4.0);

                    let sum = a + b;
                    assert_eq!(sum.re, 4.0);
                    assert_eq!(sum.im, 6.0);

                    let diff = a - b;
                    assert_eq!(diff.re, -2.0);
                    assert_eq!(diff.im, -2.0);

                    // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
                    let prod = a * b;
                    assert_eq!(prod.re, -5.0);
                    assert_eq!(prod.im, 10.0);
                }

                #[test]
                fn test_conjugate() {
                    let z = $type_name::new(3.0, 4.0);
                    let conj = z.conj();
                    assert_eq!(conj.re, 3.0);
                    assert_eq!(conj.im, -4.0);

                    // z * conj(z) = |z|²
                    let prod = z * conj;
                    assert!((prod.re - 25.0).abs() < 1e-6);
                    assert!(prod.im.abs() < 1e-6);
                }

                #[test]
                fn test_polar() {
                    let pi: $float = $pi;

                    // e^(i*pi) = -1
                    let z = $type_name::from_polar(1.0, pi);
                    assert!((z.re - (-1.0)).abs() < 1e-5);
                    assert!(z.im.abs() < 1e-5);

                    // e^(i*pi/2) = i
                    let z2 = $type_name::from_polar(1.0, pi / 2.0);
                    assert!(z2.re.abs() < 1e-5);
                    assert!((z2.im - 1.0).abs() < 1e-5);
                }

                #[test]
                fn test_exp() {
                    let pi: $float = $pi;

                    // e^(i*pi) = -1
                    let z = $type_name::new(0.0, pi);
                    let exp_z = z.exp();
                    assert!((exp_z.re - (-1.0)).abs() < 1e-5);
                    assert!(exp_z.im.abs() < 1e-5);
                }

                #[test]
                fn test_division() {
                    let a = $type_name::new(1.0, 0.0);
                    let b = $type_name::new(0.0, 1.0);

                    // 1/i = -i
                    let result = a / b;
                    assert!(result.re.abs() < 1e-6);
                    assert!((result.im - (-1.0)).abs() < 1e-6);
                }

                #[test]
                fn test_negation() {
                    let z = $type_name::new(3.0, 4.0);
                    let neg_z = -z;
                    assert_eq!(neg_z.re, -3.0);
                    assert_eq!(neg_z.im, -4.0);
                }

                #[test]
                fn test_constants() {
                    assert_eq!($type_name::ZERO.re, 0.0);
                    assert_eq!($type_name::ZERO.im, 0.0);
                    assert_eq!($type_name::ONE.re, 1.0);
                    assert_eq!($type_name::ONE.im, 0.0);
                    assert_eq!($type_name::I.re, 0.0);
                    assert_eq!($type_name::I.im, 1.0);
                }
            }
        };
    }

    test_complex_type!(complex64_tests, Complex64, f32, std::f32::consts::PI);
    test_complex_type!(complex128_tests, Complex128, f64, std::f64::consts::PI);

    #[test]
    fn test_complex_conversion() {
        let c64 = Complex64::new(1.5, 2.5);
        let c128: Complex128 = c64.into();
        assert_eq!(c128.re, 1.5);
        assert_eq!(c128.im, 2.5);

        let back: Complex64 = c128.into();
        assert_eq!(back.re, 1.5);
        assert_eq!(back.im, 2.5);
    }

    #[test]
    fn test_complex_pod() {
        // Verify bytemuck traits work for Complex64
        let z = Complex64::new(1.0, 2.0);
        let bytes = bytemuck::bytes_of(&z);
        assert_eq!(bytes.len(), 8);

        // Round-trip through aligned buffer
        let z2: &Complex64 = bytemuck::from_bytes(bytes);
        assert_eq!(*z2, z);

        // Verify Complex128 Pod
        let z128 = Complex128::new(3.0, 4.0);
        let bytes128 = bytemuck::bytes_of(&z128);
        assert_eq!(bytes128.len(), 16);
        let z128_2: &Complex128 = bytemuck::from_bytes(bytes128);
        assert_eq!(*z128_2, z128);
    }

    #[test]
    fn test_complex64_size() {
        assert_eq!(std::mem::size_of::<Complex64>(), 8);
        assert_eq!(std::mem::align_of::<Complex64>(), 4);
    }

    #[test]
    fn test_complex128_size() {
        assert_eq!(std::mem::size_of::<Complex128>(), 16);
        assert_eq!(std::mem::align_of::<Complex128>(), 8);
    }
}
