//! Lane maps of the correctly rounded F64 functions in
//! [`crate::algorithm::correctly_rounded`]. Not a stable API.
//!
//! One `#[inline(always)]` lane body compiles twice on x86_64: inside a
//! `#[target_feature(enable = "fma")]` function, and plain. `f64::mul_add` is a single
//! rounding in both, through the FMA instruction or libm `fma`, so both copies give the
//! host functions' bits. Other targets compile the plain copy only.

use crate::algorithm::correctly_rounded::traced::{
    acos_traced, exp_traced, log_traced, log1p_traced, pow_traced,
};

/// The function a lane map applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrFunction {
    /// [`crate::algorithm::correctly_rounded::exp`].
    Exp,
    /// [`crate::algorithm::correctly_rounded::log`].
    Log,
    /// [`crate::algorithm::correctly_rounded::log1p`].
    Log1p,
    /// [`crate::algorithm::correctly_rounded::acos`].
    Acos,
    /// [`crate::algorithm::correctly_rounded::pow`] with this exponent.
    PowScalar(f64),
}

/// Which compiled copy of the lane body runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneCopy {
    /// The FMA copy when the CPU has FMA, the plain copy otherwise.
    Best,
    /// The plain copy, which reaches `f64::mul_add` through libm `fma` on x86_64.
    Plain,
}

/// True when [`LaneCopy::Best`] runs the FMA copy on this CPU.
pub fn fused_copy_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Write `function(input[i])` to `output[i]` for every index both slices hold.
pub fn map_f64(function: CrFunction, input: &[f64], output: &mut [f64], copy: LaneCopy) {
    #[cfg(target_arch = "x86_64")]
    if copy == LaneCopy::Best && fused_copy_available() {
        // SAFETY: the CPU reports FMA, the only feature the copy enables.
        unsafe { lanes_fused(function, input, output) };
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = copy;
    lanes(function, input, output);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn lanes_fused(function: CrFunction, input: &[f64], output: &mut [f64]) {
    lanes(function, input, output);
}

/// The lane body. The match runs once per call, outside the loop.
///
/// Plain `for` loops keep each kernel body inside the calling copy. An iterator adaptor
/// such as `for_each` compiles to one shared out-of-line function, without FMA.
#[inline(always)]
fn lanes(function: CrFunction, input: &[f64], output: &mut [f64]) {
    match function {
        CrFunction::Exp => {
            for (y, &x) in output.iter_mut().zip(input) {
                *y = exp_traced(x).0;
            }
        }
        CrFunction::Log => {
            for (y, &x) in output.iter_mut().zip(input) {
                *y = log_traced(x).0;
            }
        }
        CrFunction::Log1p => {
            for (y, &x) in output.iter_mut().zip(input) {
                *y = log1p_traced(x).0;
            }
        }
        CrFunction::Acos => {
            for (y, &x) in output.iter_mut().zip(input) {
                *y = acos_traced(x).0;
            }
        }
        CrFunction::PowScalar(e) => {
            for (y, &x) in output.iter_mut().zip(input) {
                *y = pow_traced(x, e).0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::correctly_rounded::{acos, exp, log, log1p, pow};

    const FUNCTIONS: [CrFunction; 7] = [
        CrFunction::Exp,
        CrFunction::Log,
        CrFunction::Log1p,
        CrFunction::Acos,
        CrFunction::PowScalar(4.0 / 3.0),
        CrFunction::PowScalar(-1.5),
        CrFunction::PowScalar(3.0),
    ];

    fn host(function: CrFunction, x: f64) -> f64 {
        match function {
            CrFunction::Exp => exp(x),
            CrFunction::Log => log(x),
            CrFunction::Log1p => log1p(x),
            CrFunction::Acos => acos(x),
            CrFunction::PowScalar(e) => pow(x, e),
        }
    }

    /// Specials, binade edges, inputs near 1 and -1, and fixed-seed random bit patterns.
    fn inputs() -> Vec<f64> {
        let mut v = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff0_0000_0000_0001),
            f64::from_bits(0xfff8_0000_0000_beef),
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            f64::MAX,
            1.0,
            -1.0,
            709.782_712_893_384,
            -745.133_219_101_941_1,
        ];
        for k in 0..64u64 {
            v.push(f64::from_bits(1.0f64.to_bits() + k));
            v.push(f64::from_bits(1.0f64.to_bits() - k));
            v.push(-f64::from_bits(1.0f64.to_bits() - k));
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..1 << 14 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(f64::from_bits(state));
            // Positive inputs with exponents 2^-15 to 2^112 keep log and pow off the specials.
            v.push(f64::from_bits(
                ((0x3f0 + (state & 0x7f)) << 52) | (state >> 12),
            ));
            // Signed inputs with exponents 2^-64 to 2^-1 reach every acos branch.
            v.push(f64::from_bits(
                (state & (1 << 63)) | ((0x3bf + ((state >> 1) & 0x3f)) << 52) | (state >> 12),
            ));
        }
        v
    }

    #[test]
    fn test_both_copies_equal_host_bits() {
        let input = inputs();
        for function in FUNCTIONS {
            for copy in [LaneCopy::Best, LaneCopy::Plain] {
                let mut output = vec![0.0; input.len()];
                map_f64(function, &input, &mut output, copy);
                for (&x, &y) in input.iter().zip(&output) {
                    assert_eq!(
                        y.to_bits(),
                        host(function, x).to_bits(),
                        "{function:?} {copy:?} x={:#018x}",
                        x.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn test_map_stops_at_shorter_slice() {
        let input = [0.0, 1.0, 2.0];
        let mut output = [7.0; 2];
        map_f64(CrFunction::Exp, &input, &mut output, LaneCopy::Best);
        assert_eq!(output, [1.0, exp(1.0)]);
        let mut wide = [7.0; 4];
        map_f64(CrFunction::Exp, &input, &mut wide, LaneCopy::Plain);
        assert_eq!(wide[3], 7.0);
    }
}
