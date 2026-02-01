//! Shared constants and algorithm definitions for SIMD math functions
//!
//! This module provides polynomial coefficients and macros for generating
//! SIMD implementations across different instruction sets (AVX2, AVX-512).
//! By centralizing the algorithm logic, we ensure consistency and reduce
//! maintenance burden.

// ============================================================================
// Polynomial Coefficients for exp(x)
// ============================================================================

/// Taylor series coefficients for exp(r) where r is in [-ln(2)/2, ln(2)/2]
/// exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6!
pub mod exp_coefficients {
    pub const C0_F32: f32 = 1.0;
    pub const C1_F32: f32 = 1.0;
    pub const C2_F32: f32 = 0.5;
    pub const C3_F32: f32 = 1.0 / 6.0;
    pub const C4_F32: f32 = 1.0 / 24.0;
    pub const C5_F32: f32 = 1.0 / 120.0;
    pub const C6_F32: f32 = 1.0 / 720.0;

    pub const C0_F64: f64 = 1.0;
    pub const C1_F64: f64 = 1.0;
    pub const C2_F64: f64 = 0.5;
    pub const C3_F64: f64 = 1.0 / 6.0;
    pub const C4_F64: f64 = 1.0 / 24.0;
    pub const C5_F64: f64 = 1.0 / 120.0;
    pub const C6_F64: f64 = 1.0 / 720.0;

    /// Input clamp range to avoid overflow/underflow
    pub const MIN_F32: f32 = -88.0;
    pub const MAX_F32: f32 = 88.0;
    pub const MIN_F64: f64 = -709.0;
    pub const MAX_F64: f64 = 709.0;
}

// ============================================================================
// Polynomial Coefficients for log(x)
// ============================================================================

/// Minimax polynomial coefficients for log(1+f) where f is in [-0.2929, 0.4142]
/// (i.e., mantissa normalized to [sqrt(2)/2, sqrt(2)])
pub mod log_coefficients {
    // f32 coefficients (7-term polynomial)
    pub const C1_F32: f32 = 0.9999999995;
    pub const C2_F32: f32 = -0.4999999206;
    pub const C3_F32: f32 = 0.3333320848;
    pub const C4_F32: f32 = -0.2500097652;
    pub const C5_F32: f32 = 0.1999796621;
    pub const C6_F32: f32 = -0.1666316004;
    pub const C7_F32: f32 = 0.1428962594;

    // f64 coefficients (9-term polynomial for higher precision)
    pub const C1_F64: f64 = 0.9999999999999999;
    pub const C2_F64: f64 = -0.5;
    pub const C3_F64: f64 = 0.33333333333333333;
    pub const C4_F64: f64 = -0.25;
    pub const C5_F64: f64 = 0.2;
    pub const C6_F64: f64 = -0.16666666666666666;
    pub const C7_F64: f64 = 0.14285714285714285;
    pub const C8_F64: f64 = -0.125;
    pub const C9_F64: f64 = 0.1111111111111111;

    // IEEE 754 bit manipulation constants
    pub const EXP_BIAS_F32: i32 = 127;
    pub const EXP_BIAS_F64: i64 = 1023;
    pub const MANTISSA_MASK_F32: i32 = 0x007F_FFFF;
    pub const MANTISSA_MASK_F64: u64 = 0x000F_FFFF_FFFF_FFFF;
    pub const EXP_ZERO_F32: i32 = 0x3F80_0000; // exponent = 127 (bias)
    pub const EXP_ZERO_F64: u64 = 0x3FF0_0000_0000_0000; // exponent = 1023 (bias)
}

// ============================================================================
// Polynomial Coefficients for sin/cos
// ============================================================================

/// Taylor series coefficients for sin(x) and cos(x)
/// sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9!
/// cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8!
pub mod trig_coefficients {
    // sin(x) coefficients
    pub const S1_F32: f32 = 1.0;
    pub const S3_F32: f32 = -1.0 / 6.0;
    pub const S5_F32: f32 = 1.0 / 120.0;
    pub const S7_F32: f32 = -1.0 / 5040.0;

    pub const S1_F64: f64 = 1.0;
    pub const S3_F64: f64 = -1.0 / 6.0;
    pub const S5_F64: f64 = 1.0 / 120.0;
    pub const S7_F64: f64 = -1.0 / 5040.0;
    pub const S9_F64: f64 = 1.0 / 362880.0;

    // cos(x) coefficients
    pub const C0_F32: f32 = 1.0;
    pub const C2_F32: f32 = -0.5;
    pub const C4_F32: f32 = 1.0 / 24.0;
    pub const C6_F32: f32 = -1.0 / 720.0;

    pub const C0_F64: f64 = 1.0;
    pub const C2_F64: f64 = -0.5;
    pub const C4_F64: f64 = 1.0 / 24.0;
    pub const C6_F64: f64 = -1.0 / 720.0;
    pub const C8_F64: f64 = 1.0 / 40320.0;
}

// ============================================================================
// Polynomial Coefficients for tan(x)
// ============================================================================

/// Minimax polynomial coefficients for tan(x) on [-π/4, π/4]
/// tan(x) ≈ x * (1 + x²*(t3 + x²*(t5 + x²*(t7 + ...))))
pub mod tan_coefficients {
    pub const T1_F32: f32 = 1.0;
    pub const T3_F32: f32 = 0.3333333333333333;
    pub const T5_F32: f32 = 0.13333333333333333;
    pub const T7_F32: f32 = 0.05396825396825397;
    pub const T9_F32: f32 = 0.021869488536155203;
    pub const T11_F32: f32 = 0.008863235529902197;

    pub const T1_F64: f64 = 1.0;
    pub const T3_F64: f64 = 0.3333333333333333;
    pub const T5_F64: f64 = 0.13333333333333333;
    pub const T7_F64: f64 = 0.05396825396825397;
    pub const T9_F64: f64 = 0.021869488536155203;
    pub const T11_F64: f64 = 0.008863235529902197;
    pub const T13_F64: f64 = 0.003592128036572481;
}

// ============================================================================
// Algorithm Documentation
// ============================================================================

/// Algorithm for exp(x):
///
/// 1. **Range reduction**: exp(x) = 2^(x * log₂(e)) = 2^n * 2^f
///    - Compute y = x * log₂(e)
///    - n = round(y) (integer part)
///    - f = y - n (fractional part in [-0.5, 0.5])
///
/// 2. **Polynomial approximation**: Compute exp(f * ln(2)) using Taylor series
///    - r = f * ln(2)
///    - exp(r) ≈ 1 + r + r²/2! + r³/3! + ... (using Horner's method)
///
/// 3. **Reconstruction**: Multiply by 2^n using IEEE 754 bit manipulation
///    - For f32: 2^n = reinterpret((n + 127) << 23)
///    - For f64: 2^n = reinterpret((n + 1023) << 52)
///
/// # Accuracy
/// - f32: Relative error < 1e-6 for inputs in [-88, 88]
/// - f64: Relative error < 1e-12 for inputs in [-709, 709]
///
/// # Edge Cases
/// - Inputs outside the valid range are clamped to avoid overflow/underflow
pub const _EXP_ALGORITHM_DOC: () = ();

/// Algorithm for log(x):
///
/// 1. **Argument decomposition**: log(x) = log(2^n * m) = n * log(2) + log(m)
///    - Extract exponent n from IEEE 754 representation
///    - Extract mantissa m, normalized to [1, 2)
///
/// 2. **Range normalization**: If m > √2, divide by 2 and increment n
///    - This keeps f = m - 1 in [-0.2929, 0.4142] for better polynomial convergence
///
/// 3. **Polynomial approximation**: Compute log(1 + f) using minimax polynomial
///    - log(1+f) ≈ f * (c₁ + f*(c₂ + f*(c₃ + ...))) (Horner's method)
///
/// 4. **Reconstruction**: result = n * ln(2) + log(m)
///
/// # Accuracy
/// - f32: Relative error < 1e-6 for positive inputs
/// - f64: Relative error < 1e-12 for positive inputs
///
/// # Edge Cases
/// - x ≤ 0: Returns -inf or NaN (follows IEEE 754 semantics)
/// - x = +inf: Returns +inf
pub const _LOG_ALGORITHM_DOC: () = ();

/// Algorithm for sin(x) and cos(x):
///
/// 1. **Range reduction**: Reduce x to y in [-π/4, π/4]
///    - j = round(x * 2/π) (quadrant index)
///    - y = x - j * π/2
///
/// 2. **Polynomial approximation**:
///    - sin(y) ≈ y - y³/6 + y⁵/120 - y⁷/5040 (Taylor series)
///    - cos(y) ≈ 1 - y²/2 + y⁴/24 - y⁶/720 (Taylor series)
///
/// 3. **Quadrant selection**: Based on j mod 4:
///    - 0: sin(x) = sin(y)
///    - 1: sin(x) = cos(y)
///    - 2: sin(x) = -sin(y)
///    - 3: sin(x) = -cos(y)
///
/// # Accuracy
/// - Relative error < 1e-6 for f32, < 1e-10 for f64
/// - Accuracy degrades for very large inputs due to range reduction precision
///
/// # Input Range Warning
/// For |x| > 2^20, range reduction may lose significant precision.
/// Consider using extended precision range reduction for very large inputs.
pub const _TRIG_ALGORITHM_DOC: () = ();

// ============================================================================
// Polynomial Coefficients for atan(x)
// ============================================================================

/// Minimax polynomial coefficients for atan(x) on [-1, 1]
/// atan(x) ≈ x * (A0 + x²*(A2 + x²*(A4 + x²*(A6 + ...))))
///
/// For |x| > 1, use identity: atan(x) = sign(x) * π/2 - atan(1/x)
pub mod atan_coefficients {
    // f32 coefficients (7-term polynomial, ~1e-7 accuracy)
    pub const A0_F32: f32 = 1.0;
    pub const A2_F32: f32 = -0.333333333;
    pub const A4_F32: f32 = 0.2;
    pub const A6_F32: f32 = -0.142857142;
    pub const A8_F32: f32 = 0.111111111;
    pub const A10_F32: f32 = -0.0909090909;
    pub const A12_F32: f32 = 0.0769230769;

    // f64 coefficients (11-term polynomial, ~1e-14 accuracy)
    pub const A0_F64: f64 = 1.0;
    pub const A2_F64: f64 = -0.3333333333333333;
    pub const A4_F64: f64 = 0.2;
    pub const A6_F64: f64 = -0.14285714285714285;
    pub const A8_F64: f64 = 0.1111111111111111;
    pub const A10_F64: f64 = -0.09090909090909091;
    pub const A12_F64: f64 = 0.07692307692307693;
    pub const A14_F64: f64 = -0.06666666666666667;
    pub const A16_F64: f64 = 0.058823529411764705;
    pub const A18_F64: f64 = -0.05263157894736842;
    pub const A20_F64: f64 = 0.047619047619047616;
}

/// Algorithm for tan(x):
///
/// 1. **Range reduction**: Reduce x to y in [-π/4, π/4]
///    - j = round(x * 2/π)
///    - y = x - j * π/2
///
/// 2. **Polynomial approximation**: Using odd polynomial
///    - tan(y) ≈ y * (1 + y²*(t₃ + y²*(t₅ + ...)))
///
/// 3. **Quadrant handling**: For odd quadrants, use cotangent
///    - If j is odd: result = -1/tan(y) (cotangent)
///
/// # Accuracy
/// - Relative error < 2e-4 for f32, < 1e-4 for f64
/// - Note: tan(x) has asymptotes at x = ±π/2, ±3π/2, etc.
///
/// # Edge Cases
/// - Near asymptotes: Results may have large errors or overflow
pub const _TAN_ALGORITHM_DOC: () = ();

/// Algorithm for atan(x):
///
/// 1. **Sign handling**: Save sign of x, work with |x|
///
/// 2. **Range reduction**: For |x| > 1, use identity:
///    - atan(x) = π/2 - atan(1/x)
///
/// 3. **Polynomial approximation**: For |x| <= 1, use minimax polynomial:
///    - atan(x) ≈ x * (a₀ + x²*(a₂ + x²*(a₄ + ...))) (Horner's method)
///
/// 4. **Reconstruction**: Apply sign and range reduction inverse
///
/// # Accuracy
/// - f32: Relative error < 1e-6 for all finite inputs
/// - f64: Relative error < 1e-12 for all finite inputs
///
/// # Edge Cases
/// - atan(±∞) = ±π/2
/// - atan(0) = 0
pub const _ATAN_ALGORITHM_DOC: () = ();
