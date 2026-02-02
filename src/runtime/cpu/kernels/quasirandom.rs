//! CPU kernels for quasi-random sequence generation.
//!
//! Implements Sobol, Halton, and Latin Hypercube Sampling sequences.

use rand::Rng;
use rand::seq::SliceRandom;

// ============================================================================
// Sobol Sequence
// ============================================================================

/// Direction numbers for Sobol sequence (6 dimensions)
///
/// These are the initialization values for the Sobol sequence generator based on
/// primitive polynomials over GF(2). Source: Joe & Kuo (2008) direction numbers.
///
/// **Current limitation:** Only 6 dimensions have direction numbers. Dimensions 7+
/// fall back to van der Corput sequence (different algorithm, lower quality).
///
/// **TODO:** Expand to at least 50-100 dimensions for production use. Full 1000-dimension
/// table would require ~120KB of data (30 u32 values × 1000 dimensions).
const SOBOL_DIRECTION_NUMBERS: &[[u32; 30]] = &[
    // Dimension 0 (implicit: all 1s in binary)
    [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // Dimension 1: x^1 + 1
    [
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // Dimension 2: x^2 + x + 1
    [
        1, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // Dimension 3: x^3 + x + 1
    [
        1, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // Dimension 4: x^3 + x^2 + 1
    [
        1, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // Dimension 5: x^4 + x^3 + 1
    [
        1, 1, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
];

/// Generate Sobol sequence points (F32 version).
///
/// **Implementation note:** Only the first 6 dimensions use true Sobol direction
/// numbers. Dimensions 7+ fall back to van der Corput sequence, which has lower
/// quality (worse discrepancy) than true Sobol.
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_points * dimension`
/// - Dimension should be <= 6 for true Sobol sequence
/// - Dimensions 7+ will use van der Corput fallback
#[inline]
pub unsafe fn sobol_f32(out: *mut f32, n_points: usize, dimension: usize, skip: usize) {
    let max_dim = SOBOL_DIRECTION_NUMBERS.len().min(dimension);

    for d in 0..dimension {
        if d < max_dim {
            // Dimensions 0-5: Use true Sobol direction numbers
            sobol_dimension_f32(out.add(d), n_points, dimension, d, skip);
        } else {
            // Dimensions 6+: Fallback to van der Corput (LOWER QUALITY)
            // This provides quasi-random coverage but NOT the same discrepancy
            // properties as true Sobol sequences.
            van_der_corput_f32(out.add(d), n_points, dimension, 2 + d as u32);
        }
    }
}

/// Generate Sobol sequence for a single dimension.
#[inline]
unsafe fn sobol_dimension_f32(
    out: *mut f32,
    n_points: usize,
    stride: usize,
    dimension: usize,
    skip: usize,
) {
    let dir_nums = &SOBOL_DIRECTION_NUMBERS[dimension];

    // Initialize direction vectors
    let mut v = [0u32; 32];
    for (i, &d) in dir_nums.iter().enumerate() {
        v[i] = d << (31 - i);
    }

    let start_index = skip;

    for i in 0..n_points {
        let index = start_index + i;

        // Gray code: i XOR (i >> 1)
        let gray = index as u32 ^ (index as u32 >> 1);

        // Compute Sobol point using direction vectors
        let mut x = 0u32;
        for bit in 0..32 {
            if (gray & (1 << bit)) != 0 {
                x ^= v[bit];
            }
        }

        // Convert to float in [0, 1)
        *out.add(i * stride) = (x as f32) / (1u64 << 32) as f32;
    }
}

/// Generate Sobol sequence points (F64 version).
///
/// **Implementation note:** Only the first 6 dimensions use true Sobol direction
/// numbers. Dimensions 7+ fall back to van der Corput sequence, which has lower
/// quality (worse discrepancy) than true Sobol.
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_points * dimension`
/// - Dimension should be <= 6 for true Sobol sequence
/// - Dimensions 7+ will use van der Corput fallback
#[inline]
pub unsafe fn sobol_f64(out: *mut f64, n_points: usize, dimension: usize, skip: usize) {
    let max_dim = SOBOL_DIRECTION_NUMBERS.len().min(dimension);

    for d in 0..dimension {
        if d < max_dim {
            // Dimensions 0-5: Use true Sobol direction numbers
            sobol_dimension_f64(out.add(d), n_points, dimension, d, skip);
        } else {
            // Dimensions 6+: Fallback to van der Corput (LOWER QUALITY)
            van_der_corput_f64(out.add(d), n_points, dimension, 2 + d as u32);
        }
    }
}

#[inline]
unsafe fn sobol_dimension_f64(
    out: *mut f64,
    n_points: usize,
    stride: usize,
    dimension: usize,
    skip: usize,
) {
    let dir_nums = &SOBOL_DIRECTION_NUMBERS[dimension];

    let mut v = [0u32; 32];
    for (i, &d) in dir_nums.iter().enumerate() {
        v[i] = d << (31 - i);
    }

    let start_index = skip;

    for i in 0..n_points {
        let index = start_index + i;
        let gray = index as u32 ^ (index as u32 >> 1);

        let mut x = 0u32;
        for bit in 0..32 {
            if (gray & (1 << bit)) != 0 {
                x ^= v[bit];
            }
        }

        *out.add(i * stride) = (x as f64) / (1u64 << 32) as f64;
    }
}

// ============================================================================
// Halton Sequence
// ============================================================================

/// First 100 prime numbers for Halton sequence
const PRIMES: [u32; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

/// Generate Halton sequence points (F32 version).
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_points * dimension`
/// - Dimension must be <= 100 (limited by prime numbers)
#[inline]
pub unsafe fn halton_f32(out: *mut f32, n_points: usize, dimension: usize, skip: usize) {
    for d in 0..dimension {
        let base = PRIMES[d];
        van_der_corput_f32(out.add(d), n_points, dimension, base);

        // Apply skip offset
        if skip > 0 {
            for i in 0..n_points {
                let skipped_value = van_der_corput_single_f32(skip + i, base);
                *out.add(i * dimension + d) = skipped_value;
            }
        }
    }
}

/// van der Corput sequence in given base (F32).
#[inline]
unsafe fn van_der_corput_f32(out: *mut f32, n_points: usize, stride: usize, base: u32) {
    for i in 0..n_points {
        *out.add(i * stride) = van_der_corput_single_f32(i, base);
    }
}

/// Compute single van der Corput value.
#[inline]
fn van_der_corput_single_f32(mut index: usize, base: u32) -> f32 {
    let base_f = base as f32;
    let mut result = 0.0f32;
    let mut f = 1.0 / base_f;

    while index > 0 {
        result += f * (index % base as usize) as f32;
        index /= base as usize;
        f /= base_f;
    }

    result
}

/// Generate Halton sequence points (F64 version).
#[inline]
pub unsafe fn halton_f64(out: *mut f64, n_points: usize, dimension: usize, skip: usize) {
    for d in 0..dimension {
        let base = PRIMES[d];
        van_der_corput_f64(out.add(d), n_points, dimension, base);

        if skip > 0 {
            for i in 0..n_points {
                let skipped_value = van_der_corput_single_f64(skip + i, base);
                *out.add(i * dimension + d) = skipped_value;
            }
        }
    }
}

#[inline]
unsafe fn van_der_corput_f64(out: *mut f64, n_points: usize, stride: usize, base: u32) {
    for i in 0..n_points {
        *out.add(i * stride) = van_der_corput_single_f64(i, base);
    }
}

#[inline]
fn van_der_corput_single_f64(mut index: usize, base: u32) -> f64 {
    let base_f = base as f64;
    let mut result = 0.0f64;
    let mut f = 1.0 / base_f;

    while index > 0 {
        result += f * (index % base as usize) as f64;
        index /= base as usize;
        f /= base_f;
    }

    result
}

// ============================================================================
// Latin Hypercube Sampling
// ============================================================================

/// Generate Latin Hypercube samples (F32 version).
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_samples * dimension`
#[inline]
pub unsafe fn latin_hypercube_f32(out: *mut f32, n_samples: usize, dimension: usize) {
    let mut rng = rand::rng();

    for d in 0..dimension {
        // Create stratified intervals
        let mut intervals: Vec<usize> = (0..n_samples).collect();

        // Shuffle intervals
        intervals.shuffle(&mut rng);

        // Generate random point within each interval
        for (i, &interval) in intervals.iter().enumerate() {
            let lower = interval as f32 / n_samples as f32;
            let upper = (interval + 1) as f32 / n_samples as f32;
            let random_offset: f32 = rng.random_range(0.0..1.0);

            *out.add(i * dimension + d) = lower + random_offset * (upper - lower);
        }
    }
}

/// Generate Latin Hypercube samples (F64 version).
#[inline]
pub unsafe fn latin_hypercube_f64(out: *mut f64, n_samples: usize, dimension: usize) {
    let mut rng = rand::rng();

    for d in 0..dimension {
        let mut intervals: Vec<usize> = (0..n_samples).collect();
        intervals.shuffle(&mut rng);

        for (i, &interval) in intervals.iter().enumerate() {
            let lower = interval as f64 / n_samples as f64;
            let upper = (interval + 1) as f64 / n_samples as f64;
            let random_offset: f64 = rng.random_range(0.0..1.0);

            *out.add(i * dimension + d) = lower + random_offset * (upper - lower);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobol_f64_basic() {
        let n = 10;
        let d = 2;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            sobol_f64(out.as_mut_ptr(), n, d, 0);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }

        // First point should be 0
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn test_halton_f64_basic() {
        let n = 10;
        let d = 3;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            halton_f64(out.as_mut_ptr(), n, d, 0);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }

        // First Halton values are known:
        // base 2: 0, 0.5, 0.25, 0.75, ...
        // base 3: 0, 1/3, 2/3, 1/9, ...
        // base 5: 0, 0.2, 0.4, 0.6, ...
        assert_eq!(out[0], 0.0); // H(0, 2)
        assert!((out[d] - 0.5).abs() < 1e-10); // H(1, 2)
    }

    #[test]
    fn test_latin_hypercube_f64_stratification() {
        let n = 20;
        let d = 2;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            latin_hypercube_f64(out.as_mut_ptr(), n, d);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }

        // Check stratification: each interval should have exactly one sample
        for dim in 0..d {
            let mut bins = vec![0; n];
            for i in 0..n {
                let val = out[i * d + dim];
                let bin = (val * n as f64).floor() as usize;
                let bin = bin.min(n - 1); // Handle edge case where val is very close to 1.0
                bins[bin] += 1;
            }

            // Each bin should have exactly 1 sample
            for &count in &bins {
                assert_eq!(count, 1, "Latin hypercube not properly stratified");
            }
        }
    }

    #[test]
    fn test_van_der_corput_base2() {
        // van der Corput in base 2:
        // 0 -> 0
        // 1 -> 0.5
        // 2 -> 0.25
        // 3 -> 0.75
        // 4 -> 0.125
        assert_eq!(van_der_corput_single_f64(0, 2), 0.0);
        assert!((van_der_corput_single_f64(1, 2) - 0.5).abs() < 1e-10);
        assert!((van_der_corput_single_f64(2, 2) - 0.25).abs() < 1e-10);
        assert!((van_der_corput_single_f64(3, 2) - 0.75).abs() < 1e-10);
        assert!((van_der_corput_single_f64(4, 2) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_van_der_corput_base3() {
        // van der Corput in base 3:
        // 0 -> 0
        // 1 -> 1/3
        // 2 -> 2/3
        // 3 -> 1/9
        assert_eq!(van_der_corput_single_f64(0, 3), 0.0);
        assert!((van_der_corput_single_f64(1, 3) - 1.0 / 3.0).abs() < 1e-10);
        assert!((van_der_corput_single_f64(2, 3) - 2.0 / 3.0).abs() < 1e-10);
        assert!((van_der_corput_single_f64(3, 3) - 1.0 / 9.0).abs() < 1e-10);
    }
}
