//! Advanced PRNG kernels for CPU
//!
//! Implements counter-based and state-of-the-art PRNGs for reproducible parallel generation.
//!
//! # Algorithms
//!
//! - `philox`: Philox4x32-10 (Salmon et al. 2011)
//! - `threefry`: ThreeFry4x64-20 (Salmon et al. 2011)
//! - `pcg64`: PCG64 (O'Neill 2014)
//! - `xoshiro256`: Xoshiro256++ (Blackman & Vigna 2018)

mod pcg64;
mod philox;
mod threefry;
mod xoshiro256;

pub use pcg64::{pcg64_randn_kernel, pcg64_uniform_kernel};
pub use philox::{philox_randn_kernel, philox_uniform_kernel};
pub use threefry::{threefry_randn_kernel, threefry_uniform_kernel};
pub use xoshiro256::{xoshiro256_randn_kernel, xoshiro256_uniform_kernel};

use std::f64::consts::PI;

/// Box-Muller transform: convert two uniform values to two standard normal values
///
/// This is the shared implementation used by all PRNGs for consistency.
#[inline(always)]
pub(crate) fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    // Clamp to avoid log(0) and ensure valid range
    let u1 = u1.clamp(1e-10, 1.0 - 1e-10);

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;

    (r * theta.cos(), r * theta.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_philox_uniform_range() {
        let mut out = vec![0.0f32; 1000];
        unsafe { philox_uniform_kernel(out.as_mut_ptr(), 1000, 42, 0) };
        assert!(out.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_philox_reproducibility() {
        let mut out1 = vec![0.0f64; 100];
        let mut out2 = vec![0.0f64; 100];
        unsafe {
            philox_uniform_kernel(out1.as_mut_ptr(), 100, 42, 0);
            philox_uniform_kernel(out2.as_mut_ptr(), 100, 42, 0);
        }
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_philox_different_counters() {
        let mut out1 = vec![0.0f64; 100];
        let mut out2 = vec![0.0f64; 100];
        unsafe {
            philox_uniform_kernel(out1.as_mut_ptr(), 100, 42, 0);
            philox_uniform_kernel(out2.as_mut_ptr(), 100, 42, 1);
        }
        assert_ne!(out1, out2);
    }

    #[test]
    fn test_philox_randn_statistics() {
        let mut out = vec![0.0f64; 10000];
        unsafe { philox_randn_kernel(out.as_mut_ptr(), 10000, 42, 0) };

        let mean: f64 = out.iter().sum::<f64>() / 10000.0;
        let variance: f64 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 10000.0;

        assert!((mean - 0.0).abs() < 0.05, "mean = {}", mean);
        assert!((variance - 1.0).abs() < 0.1, "variance = {}", variance);
    }

    #[test]
    fn test_threefry_uniform_range() {
        let mut out = vec![0.0f32; 1000];
        unsafe { threefry_uniform_kernel(out.as_mut_ptr(), 1000, 42, 0) };
        assert!(out.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_pcg64_uniform_range() {
        let mut out = vec![0.0f32; 1000];
        unsafe { pcg64_uniform_kernel(out.as_mut_ptr(), 1000, 42, 0) };
        assert!(out.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_xoshiro256_uniform_range() {
        let mut out = vec![0.0f32; 1000];
        unsafe { xoshiro256_uniform_kernel(out.as_mut_ptr(), 1000, 42) };
        assert!(out.iter().all(|&x| x >= 0.0 && x < 1.0));
    }
}
