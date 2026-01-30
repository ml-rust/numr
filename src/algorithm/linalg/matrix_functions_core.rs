//! Core numerical algorithms for matrix functions
//!
//! This module provides the shared mathematical implementations used by all backends
//! (CPU, CUDA, WebGPU). These are pure numerical algorithms operating on f64 arrays.
//!
//! # Design Rationale
//!
//! By centralizing these algorithms, we ensure:
//! 1. **DRY compliance** - algorithms defined once, used everywhere
//! 2. **Backend parity** - identical math across all backends
//! 3. **Testability** - algorithms can be unit tested in isolation
//!
//! # Why f64-Only?
//!
//! Matrix function algorithms (Parlett recurrence, Denman-Beavers) are inherently
//! sequential and don't benefit from GPU parallelization. They operate on small
//! quasi-triangular matrices (typically n×n where n is matrix dimension).
//!
//! The workflow is:
//! 1. **Schur decomposition** (on GPU) - O(n³), benefits from parallelization
//! 2. **Matrix function on T** (this module) - O(n²), sequential
//! 3. **Reconstruction Z @ f(T) @ Z^T** (on GPU) - O(n³), benefits from parallelization
//!
//! This hybrid approach gives the best performance: expensive operations on GPU,
//! sequential operations on CPU.
//!
//! # Usage from Backends
//!
//! ```ignore
//! // 1. Perform Schur decomposition on GPU
//! let schur = client.schur_decompose(&a)?;
//!
//! // 2. Transfer to CPU for matrix function computation
//! let t_data: Vec<f64> = schur.t.to_vec();
//!
//! // 3. Use shared algorithm
//! let exp_t = matrix_functions_core::exp_quasi_triangular_f64(&t_data, n);
//!
//! // 4. Transfer back and reconstruct on GPU
//! let result = Z @ exp(T) @ Z^T;  // GPU matmul
//! ```

use crate::error::{Error, Result};

// ============================================================================
// Convergence Constants
// ============================================================================

/// Maximum square root iterations in inverse scaling and squaring (log)
const LOG_MAX_SQRT_ITERATIONS: usize = 20;

/// Maximum terms in Taylor series for log(I + X)
const LOG_TAYLOR_MAX_TERMS: usize = 20;

/// Maximum iterations for Denman-Beavers sqrt iteration
const SQRT_DENMAN_BEAVERS_MAX_ITER: usize = 20;

/// Convergence threshold for inverse scaling and squaring (relative to identity)
const LOG_CONVERGENCE_THRESHOLD: f64 = 0.5;

/// Compute exp(T) for quasi-triangular T using Parlett recurrence (f64)
///
/// T is in real Schur form (upper quasi-triangular):
/// - 1×1 diagonal blocks for real eigenvalues
/// - 2×2 diagonal blocks for complex conjugate eigenvalue pairs
///
/// Returns the matrix exponential exp(T) as a flat array.
pub fn exp_quasi_triangular_f64(t: &[f64], n: usize) -> Vec<f64> {
    let mut f = vec![0.0; n * n];
    let eps = f64::EPSILON;

    // Phase 1: Compute diagonal blocks
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1) * n + i].abs() > eps {
            // 2×2 block: complex conjugate eigenvalues
            let a = t[i * n + i];
            let b = t[i * n + (i + 1)];
            let c = t[(i + 1) * n + i];
            let (f11, f12, f21, f22) = exp_2x2_block_f64(a, b, c);
            f[i * n + i] = f11;
            f[i * n + (i + 1)] = f12;
            f[(i + 1) * n + i] = f21;
            f[(i + 1) * n + (i + 1)] = f22;
            i += 2;
        } else {
            // 1×1 block: real eigenvalue
            f[i * n + i] = t[i * n + i].exp();
            i += 1;
        }
    }

    // Phase 2: Fill superdiagonals using Parlett recurrence
    for d in 1..n {
        for i in 0..(n - d) {
            let j = i + d;

            // Skip if this is part of a 2×2 block
            if i + 1 < n && t[(i + 1) * n + i].abs() > eps && d == 1 {
                continue;
            }
            if j > 0 && t[j * n + (j - 1)].abs() > eps && d == 1 {
                continue;
            }

            let t_ii = t[i * n + i];
            let t_jj = t[j * n + j];
            let t_ij = t[i * n + j];

            // Compute contribution from intermediate elements
            let mut sum = 0.0;
            for k in (i + 1)..j {
                sum += f[i * n + k] * t[k * n + j];
                sum -= t[i * n + k] * f[k * n + j];
            }

            let f_ii = f[i * n + i];
            let f_jj = f[j * n + j];

            let diff = t_ii - t_jj;
            if diff.abs() > eps {
                f[i * n + j] = ((f_ii - f_jj) * t_ij + sum) / diff;
            } else {
                f[i * n + j] = f_ii * t_ij + sum;
            }
        }
    }

    f
}

/// Compute exp of a 2×2 block [a, b; c, a'] with potentially complex eigenvalues
fn exp_2x2_block_f64(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let bc = b * c;
    let exp_a = a.exp();

    if bc < 0.0 {
        // Complex eigenvalues: a ± i*ω where ω = sqrt(-bc)
        let omega = (-bc).sqrt();
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let f11 = exp_a * cos_omega;
        let f12 = exp_a * sin_omega * b / omega;
        let f21 = exp_a * sin_omega * c / omega;
        (f11, f12, f21, f11)
    } else if bc > 0.0 {
        // Real eigenvalues: a ± δ where δ = sqrt(bc)
        let delta = bc.sqrt();
        let sinh_delta = delta.sinh();
        let cosh_delta = delta.cosh();
        let f11 = exp_a * cosh_delta;
        let f12 = exp_a * sinh_delta * b / delta;
        let f21 = exp_a * sinh_delta * c / delta;
        (f11, f12, f21, f11)
    } else {
        // bc = 0, degenerate case
        (exp_a, exp_a * b, exp_a * c, exp_a)
    }
}

/// Compute log(T) for quasi-triangular T using inverse scaling and squaring (f64)
///
/// Algorithm:
/// 1. Take repeated square roots until T^{1/2^k} is close to identity
/// 2. Compute log(T^{1/2^k}) using Padé approximation
/// 3. Scale back: log(T) = 2^k * log(T^{1/2^k})
pub fn log_quasi_triangular_f64(t: &[f64], n: usize, eps: f64) -> Vec<f64> {
    let mut work = t.to_vec();
    let mut k = 0;

    // Take square roots until matrix is close to identity
    while k < LOG_MAX_SQRT_ITERATIONS {
        let mut norm_diff = 0.0;
        for i in 0..n {
            for j in 0..n {
                let target = if i == j { 1.0 } else { 0.0 };
                let diff = work[i * n + j] - target;
                norm_diff += diff * diff;
            }
        }
        norm_diff = norm_diff.sqrt();

        if norm_diff < LOG_CONVERGENCE_THRESHOLD {
            break;
        }

        work = sqrt_quasi_triangular_f64(&work, n, eps);
        k += 1;
    }

    // Compute log(work) for work close to I using Taylor series
    let mut x = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { 1.0 } else { 0.0 };
            x[i * n + j] = work[i * n + j] - target;
        }
    }

    let log_work = log_near_identity_f64(&x, n, eps);

    // Scale back: log(T) = 2^k * log_work
    let scale = (1u64 << k) as f64;
    log_work.iter().map(|&v| v * scale).collect()
}

/// Compute sqrt(T) for quasi-triangular T using Denman-Beavers iteration (f64)
pub fn sqrt_quasi_triangular_f64(t: &[f64], n: usize, eps: f64) -> Vec<f64> {
    let mut y = t.to_vec();
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }

    for _iter in 0..SQRT_DENMAN_BEAVERS_MAX_ITER {
        let y_inv = match invert_matrix_f64(&y, n, eps) {
            Some(inv) => inv,
            None => break,
        };
        let z_inv = match invert_matrix_f64(&z, n, eps) {
            Some(inv) => inv,
            None => break,
        };

        let mut y_new = vec![0.0; n * n];
        let mut z_new = vec![0.0; n * n];

        for i in 0..(n * n) {
            y_new[i] = (y[i] + z_inv[i]) / 2.0;
            z_new[i] = (z[i] + y_inv[i]) / 2.0;
        }

        let mut diff = 0.0;
        for i in 0..(n * n) {
            diff += (y_new[i] - y[i]).abs();
        }

        y = y_new;
        z = z_new;

        if diff < eps * (n * n) as f64 {
            break;
        }
    }

    y
}

/// Compute log(I + X) for small X using Taylor series (f64)
fn log_near_identity_f64(x: &[f64], n: usize, eps: f64) -> Vec<f64> {
    let mut result = x.to_vec();
    let mut x_power = x.to_vec();

    for k in 2..=LOG_TAYLOR_MAX_TERMS {
        x_power = matmul_square_f64(&x_power, x, n);
        let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
        let coeff = sign / (k as f64);

        let mut max_term: f64 = 0.0;
        for i in 0..(n * n) {
            let term = x_power[i] * coeff;
            result[i] += term;
            max_term = max_term.max(term.abs());
        }

        if max_term < eps {
            break;
        }
    }

    result
}

/// Apply general function to quasi-triangular matrix using Parlett recurrence (f64)
///
/// Computes f(T) where f is a scalar function applied to eigenvalues.
pub fn funm_quasi_triangular_f64<F>(t: &[f64], n: usize, f: &F) -> Result<Vec<f64>>
where
    F: Fn(f64) -> f64,
{
    let mut result = vec![0.0; n * n];
    let eps = f64::EPSILON;

    // Phase 1: Compute diagonal blocks
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1) * n + i].abs() > eps {
            // 2×2 block
            let a = t[i * n + i];
            let b = t[i * n + (i + 1)];
            let c = t[(i + 1) * n + i];
            let d = t[(i + 1) * n + (i + 1)];

            let (f11, f12, f21, f22) = funm_2x2_block_f64(a, b, c, d, f)?;
            result[i * n + i] = f11;
            result[i * n + (i + 1)] = f12;
            result[(i + 1) * n + i] = f21;
            result[(i + 1) * n + (i + 1)] = f22;
            i += 2;
        } else {
            // 1×1 block
            let val = t[i * n + i];
            let f_val = f(val);
            if f_val.is_nan() || f_val.is_infinite() {
                return Err(Error::InvalidArgument {
                    arg: "f",
                    reason: format!("function returned NaN or Inf for eigenvalue {}", val),
                });
            }
            result[i * n + i] = f_val;
            i += 1;
        }
    }

    // Phase 2: Fill superdiagonals using Parlett recurrence
    for diag in 1..n {
        for i in 0..(n - diag) {
            let j = i + diag;

            // Skip if part of a 2×2 block
            if i + 1 < n && t[(i + 1) * n + i].abs() > eps && diag == 1 {
                continue;
            }
            if j > 0 && t[j * n + (j - 1)].abs() > eps && diag == 1 {
                continue;
            }

            let t_ii = t[i * n + i];
            let t_jj = t[j * n + j];
            let t_ij = t[i * n + j];

            let f_ii = result[i * n + i];
            let f_jj = result[j * n + j];

            let mut sum = 0.0;
            for k in (i + 1)..j {
                sum += result[i * n + k] * t[k * n + j];
                sum -= t[i * n + k] * result[k * n + j];
            }

            let diff = t_ii - t_jj;
            let f_ij = if diff.abs() > eps {
                (f_ii - f_jj) * t_ij / diff + sum / diff
            } else {
                f_ii * t_ij + sum
            };

            result[i * n + j] = f_ij;
        }
    }

    Ok(result)
}

/// Apply function to 2×2 block (f64)
fn funm_2x2_block_f64<F>(a: f64, b: f64, c: f64, d: f64, f: &F) -> Result<(f64, f64, f64, f64)>
where
    F: Fn(f64) -> f64,
{
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace / 4.0 - det;

    if disc >= 0.0 {
        // Real eigenvalues
        let sqrt_disc = disc.sqrt();
        let lambda1 = trace / 2.0 + sqrt_disc;
        let lambda2 = trace / 2.0 - sqrt_disc;

        let f1 = f(lambda1);
        let f2 = f(lambda2);

        if f1.is_nan() || f1.is_infinite() || f2.is_nan() || f2.is_infinite() {
            return Err(Error::InvalidArgument {
                arg: "f",
                reason: "function returned NaN or Inf for eigenvalue".to_string(),
            });
        }

        if (lambda1 - lambda2).abs() > f64::EPSILON {
            let coeff1 = (f1 - f2) / (lambda1 - lambda2);
            let coeff0 = f1 - coeff1 * lambda1;
            Ok((
                coeff0 + coeff1 * a,
                coeff1 * b,
                coeff1 * c,
                coeff0 + coeff1 * d,
            ))
        } else {
            // Repeated eigenvalue
            Ok((
                f1,
                f1 * b / (a - lambda1 + 1.0),
                f1 * c / (a - lambda1 + 1.0),
                f1,
            ))
        }
    } else {
        // Complex eigenvalues: λ = α ± iβ
        let alpha = trace / 2.0;
        let beta = (-disc).sqrt();

        let f_alpha = f(alpha);
        if f_alpha.is_nan() || f_alpha.is_infinite() {
            return Err(Error::InvalidArgument {
                arg: "f",
                reason: "function returned NaN or Inf for eigenvalue".to_string(),
            });
        }

        // Approximate derivative using finite differences
        let h = beta.abs().max(1e-8);
        let f_plus = f(alpha + h);
        let f_minus = f(alpha - h);
        let df_approx = (f_plus - f_minus) / (2.0 * h);

        let f11 = f_alpha + df_approx * (a - alpha);
        let f12 = df_approx * b;
        let f21 = df_approx * c;
        let f22 = f_alpha + df_approx * (d - alpha);

        Ok((f11, f12, f21, f22))
    }
}

/// Matrix multiplication for square matrices (f64)
pub fn matmul_square_f64(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Matrix inversion using LU decomposition with partial pivoting (f64)
pub fn invert_matrix_f64(a: &[f64], n: usize, eps: f64) -> Option<Vec<f64>> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    // LU factorization with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < eps {
            return None; // Singular matrix
        }

        // Swap rows
        if max_row != k {
            perm.swap(k, max_row);
            for j in 0..n {
                let tmp = lu[k * n + j];
                lu[k * n + j] = lu[max_row * n + j];
                lu[max_row * n + j] = tmp;
            }
        }

        // Elimination
        let pivot = lu[k * n + k];
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    // Solve for each column of the inverse
    let mut inv = vec![0.0; n * n];

    for col in 0..n {
        // Create column of identity (permuted)
        let mut b = vec![0.0; n];
        b[perm[col]] = 1.0;

        // Forward substitution (Ly = Pb)
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= lu[i * n + j] * y[j];
            }
            y[i] = sum;
        }

        // Back substitution (Ux = y)
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= lu[i * n + j] * x[j];
            }
            x[i] = sum / lu[i * n + i];
        }

        // Store in inverse matrix
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }

    Some(inv)
}

/// Reconstruct result from Schur decomposition: R = Z @ F @ Z^T (f64)
pub fn reconstruct_from_schur_f64(z: &[f64], f: &[f64], n: usize) -> Vec<f64> {
    // First compute temp = Z @ F
    let temp = matmul_square_f64(z, f, n);

    // Then compute result = temp @ Z^T
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += temp[i * n + k] * z[j * n + k]; // z[j,k] = z^T[k,j]
            }
            result[i * n + j] = sum;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_2x2_block_identity() {
        // exp([0,0;0,0]) = I
        let (f11, f12, f21, f22) = exp_2x2_block_f64(0.0, 0.0, 0.0);
        assert!((f11 - 1.0).abs() < 1e-10);
        assert!(f12.abs() < 1e-10);
        assert!(f21.abs() < 1e-10);
        assert!((f22 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_square_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = matmul_square_f64(&a, &b, 2);
        assert!((c[0] - 2.0).abs() < 1e-10);
        assert!((c[1] - 3.0).abs() < 1e-10);
        assert!((c[2] - 4.0).abs() < 1e-10);
        assert!((c[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_matrix_2x2() {
        // [[1,2],[3,4]] has inverse [[-2,1],[1.5,-0.5]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let inv = invert_matrix_f64(&a, 2, 1e-10).unwrap();
        assert!((inv[0] - (-2.0)).abs() < 1e-10);
        assert!((inv[1] - 1.0).abs() < 1e-10);
        assert!((inv[2] - 1.5).abs() < 1e-10);
        assert!((inv[3] - (-0.5)).abs() < 1e-10);
    }
}
