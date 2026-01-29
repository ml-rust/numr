//! Distribution sampling kernels for CPU
//!
//! Implements probability distribution sampling using the rand_distr crate.
//! All kernels support F32, F64, and optionally F16/BF16 via the Element trait.

use crate::dtype::Element;
use rand::Rng;
use rand_distr::{Beta, Binomial, Distribution, Exp, Gamma, Poisson, StandardNormal};

/// Sample from Bernoulli distribution (binary outcomes)
///
/// Each element is 1 with probability p, 0 otherwise.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `p` must be in [0, 1]
#[inline]
pub unsafe fn bernoulli_kernel<T: Element>(out: *mut T, p: f64, len: usize) {
    let mut rng = rand::rng();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let u: f64 = rng.random();
        let val = if u < p { 1.0 } else { 0.0 };
        *elem = T::from_f64(val);
    }
}

/// Sample from Beta distribution
///
/// Uses the relationship: if X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
/// then X / (X + Y) ~ Beta(α, β).
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `alpha > 0` and `beta > 0`
#[inline]
pub unsafe fn beta_kernel<T: Element>(out: *mut T, alpha: f64, beta: f64, len: usize) {
    let mut rng = rand::rng();
    let dist = Beta::new(alpha, beta).expect("Invalid beta parameters");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = dist.sample(&mut rng);
        *elem = T::from_f64(val);
    }
}

/// Sample from Gamma distribution
///
/// Uses Marsaglia and Tsang's method for shape >= 1,
/// with transformation for shape < 1.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `shape_param > 0` and `scale > 0`
#[inline]
pub unsafe fn gamma_kernel<T: Element>(out: *mut T, shape_param: f64, scale: f64, len: usize) {
    let mut rng = rand::rng();
    let dist = Gamma::new(shape_param, scale).expect("Invalid gamma parameters");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = dist.sample(&mut rng);
        *elem = T::from_f64(val);
    }
}

/// Sample from Exponential distribution
///
/// Uses inverse transform: X = -ln(U) / rate where U ~ Uniform(0, 1).
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `rate > 0`
#[inline]
pub unsafe fn exponential_kernel<T: Element>(out: *mut T, rate: f64, len: usize) {
    let mut rng = rand::rng();
    let dist = Exp::new(rate).expect("Invalid exponential rate");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = dist.sample(&mut rng);
        *elem = T::from_f64(val);
    }
}

/// Sample from Poisson distribution
///
/// For small lambda (< 30): uses direct inversion method.
/// For large lambda: uses normal approximation internally.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `lambda > 0`
#[inline]
pub unsafe fn poisson_kernel<T: Element>(out: *mut T, lambda: f64, len: usize) {
    let mut rng = rand::rng();
    let dist = Poisson::new(lambda).expect("Invalid poisson lambda");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = dist.sample(&mut rng);
        *elem = T::from_f64(val);
    }
}

/// Sample from Binomial distribution
///
/// For small n: direct simulation (sum of Bernoulli trials).
/// For large n: uses BTRD algorithm internally.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `n > 0` and `p` in [0, 1]
#[inline]
pub unsafe fn binomial_kernel<T: Element>(out: *mut T, n: u64, p: f64, len: usize) {
    let mut rng = rand::rng();
    let dist = Binomial::new(n, p).expect("Invalid binomial parameters");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val = dist.sample(&mut rng);
        *elem = T::from_f64(val as f64);
    }
}

/// Sample from Laplace (double exponential) distribution
///
/// Uses inverse transform: X = μ - b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
/// where U ~ Uniform(0, 1).
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `scale > 0`
#[inline]
pub unsafe fn laplace_kernel<T: Element>(out: *mut T, loc: f64, scale: f64, len: usize) {
    let mut rng = rand::rng();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let u: f64 = rng.random::<f64>() - 0.5;
        // Avoid log(0) by clamping
        let abs_u = u.abs().max(1e-300);
        let val = loc - scale * u.signum() * (1.0 - 2.0 * abs_u).ln();
        *elem = T::from_f64(val);
    }
}

/// Sample from Chi-squared distribution
///
/// Implemented as Gamma(df/2, 2) since χ²(k) = Gamma(k/2, 2).
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `df > 0`
#[inline]
pub unsafe fn chi_squared_kernel<T: Element>(out: *mut T, df: f64, len: usize) {
    let mut rng = rand::rng();
    // χ²(df) = Gamma(df/2, 2)
    let dist = Gamma::new(df / 2.0, 2.0).expect("Invalid chi-squared df");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = dist.sample(&mut rng);
        *elem = T::from_f64(val);
    }
}

/// Sample from Student's t distribution
///
/// Uses the relationship: T = Z / sqrt(V/ν) where Z ~ N(0,1) and V ~ χ²(ν).
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `df > 0`
#[inline]
pub unsafe fn student_t_kernel<T: Element>(out: *mut T, df: f64, len: usize) {
    let mut rng = rand::rng();
    let normal = StandardNormal;
    // χ²(df) = Gamma(df/2, 2)
    let chi2 = Gamma::new(df / 2.0, 2.0).expect("Invalid student-t df");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let z: f64 = normal.sample(&mut rng);
        let v: f64 = chi2.sample(&mut rng);
        let val = z / (v / df).sqrt();
        *elem = T::from_f64(val);
    }
}

/// Sample from F distribution
///
/// Uses the relationship: F = (X₁/d₁) / (X₂/d₂)
/// where X₁ ~ χ²(d₁) and X₂ ~ χ²(d₂).
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `df1 > 0` and `df2 > 0`
#[inline]
pub unsafe fn f_distribution_kernel<T: Element>(out: *mut T, df1: f64, df2: f64, len: usize) {
    let mut rng = rand::rng();
    // χ²(df) = Gamma(df/2, 2)
    let chi2_1 = Gamma::new(df1 / 2.0, 2.0).expect("Invalid F df1");
    let chi2_2 = Gamma::new(df2 / 2.0, 2.0).expect("Invalid F df2");
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let x1: f64 = chi2_1.sample(&mut rng);
        let x2: f64 = chi2_2.sample(&mut rng);
        let val = (x1 / df1) / (x2 / df2);
        *elem = T::from_f64(val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernoulli_kernel() {
        let mut out = vec![0.0f32; 1000];
        unsafe { bernoulli_kernel(out.as_mut_ptr(), 0.5, 1000) };

        // All values should be 0 or 1
        assert!(out.iter().all(|&x| x == 0.0 || x == 1.0));

        // Mean should be approximately 0.5
        let mean: f32 = out.iter().sum::<f32>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_beta_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { beta_kernel(out.as_mut_ptr(), 2.0, 5.0, 1000) };

        // All values should be in (0, 1)
        assert!(out.iter().all(|&x| x > 0.0 && x < 1.0));

        // Mean should be approximately α/(α+β) = 2/7 ≈ 0.286
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!((mean - 0.286).abs() < 0.05);
    }

    #[test]
    fn test_gamma_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { gamma_kernel(out.as_mut_ptr(), 2.0, 1.0, 1000) };

        // All values should be positive
        assert!(out.iter().all(|&x| x > 0.0));

        // Mean should be approximately k*θ = 2
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!((mean - 2.0).abs() < 0.3);
    }

    #[test]
    fn test_exponential_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { exponential_kernel(out.as_mut_ptr(), 0.5, 1000) };

        // All values should be non-negative
        assert!(out.iter().all(|&x| x >= 0.0));

        // Mean should be approximately 1/λ = 2
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!((mean - 2.0).abs() < 0.4);
    }

    #[test]
    fn test_poisson_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { poisson_kernel(out.as_mut_ptr(), 5.0, 1000) };

        // All values should be non-negative integers
        assert!(out.iter().all(|&x| x >= 0.0 && x == x.floor()));

        // Mean should be approximately λ = 5
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!((mean - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_binomial_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { binomial_kernel(out.as_mut_ptr(), 10, 0.5, 1000) };

        // All values should be in [0, n]
        assert!(out.iter().all(|&x| x >= 0.0 && x <= 10.0 && x == x.floor()));

        // Mean should be approximately n*p = 5
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!((mean - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_laplace_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { laplace_kernel(out.as_mut_ptr(), 0.0, 1.0, 1000) };

        // Mean should be approximately 0
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_chi_squared_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { chi_squared_kernel(out.as_mut_ptr(), 5.0, 1000) };

        // All values should be positive
        assert!(out.iter().all(|&x| x > 0.0));

        // Mean should be approximately df = 5
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!((mean - 5.0).abs() < 0.6);
    }

    #[test]
    fn test_student_t_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { student_t_kernel(out.as_mut_ptr(), 10.0, 1000) };

        // Mean should be approximately 0 (for df > 1)
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_f_distribution_kernel() {
        let mut out = vec![0.0f64; 1000];
        unsafe { f_distribution_kernel(out.as_mut_ptr(), 5.0, 20.0, 1000) };

        // All values should be positive
        assert!(out.iter().all(|&x| x > 0.0));

        // Mean should be approximately d₂/(d₂-2) = 20/18 ≈ 1.11 for d₂ > 2
        let mean: f64 = out.iter().sum::<f64>() / 1000.0;
        assert!((mean - 1.11).abs() < 0.3);
    }
}
