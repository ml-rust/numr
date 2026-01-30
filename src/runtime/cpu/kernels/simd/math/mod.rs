//! Shared SIMD mathematical functions
//!
//! This module provides optimized SIMD implementations of transcendental functions
//! (exp, tanh) that are used across multiple SIMD kernel modules. By centralizing
//! these implementations, we ensure consistency and eliminate code duplication.
//!
//! # Supported Functions
//!
//! | Function | f32 | f64 | Algorithm |
//! |----------|-----|-----|-----------|
//! | exp      | ✓   | ✓   | Range reduction + Taylor series |
//! | tanh     | ✓   | ✓   | Based on exp: (e^2x - 1)/(e^2x + 1) |
//!
//! # Accuracy
//!
//! These approximations prioritize speed over full IEEE precision:
//! - Relative error: < 1e-6 for f32, < 1e-12 for f64
//! - Valid input range: [-88, 88] for f32, [-709, 709] for f64

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_exp_f32_accuracy() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let inputs: Vec<f32> = (-40..=40).map(|x| x as f32 * 0.1).collect();
        let mut outputs = vec![0.0f32; inputs.len()];

        unsafe {
            for (i, &x) in inputs.iter().enumerate() {
                let v = std::arch::x86_64::_mm256_set1_ps(x);
                let result = avx2::exp_f32(v);
                let mut arr = [0.0f32; 8];
                std::arch::x86_64::_mm256_storeu_ps(arr.as_mut_ptr(), result);
                outputs[i] = arr[0];
            }
        }

        for (i, (&input, &output)) in inputs.iter().zip(outputs.iter()).enumerate() {
            let expected = input.exp();
            let rel_err = (output - expected).abs() / expected.abs().max(1e-10);
            assert!(
                rel_err < 1e-5,
                "exp({}) = {} (expected {}), rel_err = {} at index {}",
                input,
                output,
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_tanh_f32_accuracy() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let inputs: Vec<f32> = (-30..=30).map(|x| x as f32 * 0.1).collect();
        let mut outputs = vec![0.0f32; inputs.len()];

        unsafe {
            for (i, &x) in inputs.iter().enumerate() {
                let v = std::arch::x86_64::_mm256_set1_ps(x);
                let result = avx2::tanh_f32(v);
                let mut arr = [0.0f32; 8];
                std::arch::x86_64::_mm256_storeu_ps(arr.as_mut_ptr(), result);
                outputs[i] = arr[0];
            }
        }

        for (&input, &output) in inputs.iter().zip(outputs.iter()) {
            let expected = input.tanh();
            let abs_err = (output - expected).abs();
            assert!(
                abs_err < 1e-5,
                "tanh({}) = {} (expected {}), abs_err = {}",
                input,
                output,
                expected,
                abs_err
            );
        }
    }
}
