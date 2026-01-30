//! QZ Decomposition (Generalized Schur decomposition)
//!
//! Implements the implicit double-shift QZ algorithm following LAPACK's approach.
//! This allows computation entirely in real arithmetic, even for complex eigenvalues.

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::GeneralizedSchurDecomposition;
use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// QZ decomposition for matrix pencil (A, B)
pub fn qz_decompose_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<GeneralizedSchurDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device),
            z: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device),
            s: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device),
            t: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0, 0], device),
            eigenvalues_real: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0], device),
            eigenvalues_imag: Tensor::<CpuRuntime>::from_slice(&[] as &[T], &[0], device),
        });
    }

    // Copy inputs to working matrices
    let mut s_data: Vec<T> = a.to_vec();
    let mut t_data: Vec<T> = b.to_vec();

    // Initialize Q and Z as identity matrices
    let mut q_data: Vec<T> = vec![T::zero(); n * n];
    let mut z_data: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        q_data[i * n + i] = T::one();
        z_data[i * n + i] = T::one();
    }

    // Step 1: Reduce to Hessenberg-triangular form
    hessenberg_triangular_reduction::<T>(&mut s_data, &mut t_data, &mut q_data, &mut z_data, n);

    // Step 2: Double-shift QZ iteration
    qz_double_shift_iteration::<T>(&mut s_data, &mut t_data, &mut q_data, &mut z_data, n);

    // Step 3: Extract generalized eigenvalues
    let (eigenvalues_real, eigenvalues_imag) =
        extract_generalized_eigenvalues::<T>(&s_data, &t_data, n);

    Ok(GeneralizedSchurDecomposition {
        q: Tensor::<CpuRuntime>::from_slice(&q_data, &[n, n], device),
        z: Tensor::<CpuRuntime>::from_slice(&z_data, &[n, n], device),
        s: Tensor::<CpuRuntime>::from_slice(&s_data, &[n, n], device),
        t: Tensor::<CpuRuntime>::from_slice(&t_data, &[n, n], device),
        eigenvalues_real: Tensor::<CpuRuntime>::from_slice(&eigenvalues_real, &[n], device),
        eigenvalues_imag: Tensor::<CpuRuntime>::from_slice(&eigenvalues_imag, &[n], device),
    })
}

/// Reduce (A, B) to Hessenberg-triangular form using Givens rotations
fn hessenberg_triangular_reduction<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
) {
    // First, reduce B to upper triangular using QR
    for k in 0..(n - 1) {
        for i in (k + 1)..n {
            let a_val = r[k * n + k].to_f64();
            let b_val = r[i * n + k].to_f64();

            if b_val.abs() < T::epsilon_val() {
                continue;
            }

            // Compute Givens rotation
            let rr = (a_val * a_val + b_val * b_val).sqrt();
            let c = a_val / rr;
            let s = -b_val / rr;

            // Apply to B (rows k and i)
            for j in 0..n {
                let t1 = r[k * n + j].to_f64();
                let t2 = r[i * n + j].to_f64();
                r[k * n + j] = T::from_f64(c * t1 - s * t2);
                r[i * n + j] = T::from_f64(s * t1 + c * t2);
            }

            // Apply to A (rows k and i)
            for j in 0..n {
                let t1 = h[k * n + j].to_f64();
                let t2 = h[i * n + j].to_f64();
                h[k * n + j] = T::from_f64(c * t1 - s * t2);
                h[i * n + j] = T::from_f64(s * t1 + c * t2);
            }

            // Accumulate into Q
            for j in 0..n {
                let t1 = q[j * n + k].to_f64();
                let t2 = q[j * n + i].to_f64();
                q[j * n + k] = T::from_f64(c * t1 - s * t2);
                q[j * n + i] = T::from_f64(s * t1 + c * t2);
            }
        }
    }

    // Then, reduce A to Hessenberg while keeping B triangular
    for k in 0..(n - 2) {
        for i in (k + 2)..n {
            let a_val = h[(k + 1) * n + k].to_f64();
            let b_val = h[i * n + k].to_f64();

            if b_val.abs() < T::epsilon_val() {
                continue;
            }

            // Givens rotation to zero h[i, k]
            let rr = (a_val * a_val + b_val * b_val).sqrt();
            let c = a_val / rr;
            let s = -b_val / rr;

            // Apply to A (rows k+1 and i)
            for j in 0..n {
                let t1 = h[(k + 1) * n + j].to_f64();
                let t2 = h[i * n + j].to_f64();
                h[(k + 1) * n + j] = T::from_f64(c * t1 - s * t2);
                h[i * n + j] = T::from_f64(s * t1 + c * t2);
            }

            // Apply to B (rows k+1 and i)
            for j in 0..n {
                let t1 = r[(k + 1) * n + j].to_f64();
                let t2 = r[i * n + j].to_f64();
                r[(k + 1) * n + j] = T::from_f64(c * t1 - s * t2);
                r[i * n + j] = T::from_f64(s * t1 + c * t2);
            }

            // Accumulate into Q
            for j in 0..n {
                let t1 = q[j * n + (k + 1)].to_f64();
                let t2 = q[j * n + i].to_f64();
                q[j * n + (k + 1)] = T::from_f64(c * t1 - s * t2);
                q[j * n + i] = T::from_f64(s * t1 + c * t2);
            }

            // Now zero the fill-in in B using column operations
            let b1 = r[(k + 1) * n + (k + 1)].to_f64();
            let b2 = r[(k + 1) * n + i].to_f64();

            if b2.abs() > T::epsilon_val() {
                let rr2 = (b1 * b1 + b2 * b2).sqrt();
                let c2 = b1 / rr2;
                let s2 = b2 / rr2;

                // Apply to B (cols k+1 and i)
                for row in 0..n {
                    let t1 = r[row * n + (k + 1)].to_f64();
                    let t2 = r[row * n + i].to_f64();
                    r[row * n + (k + 1)] = T::from_f64(c2 * t1 + s2 * t2);
                    r[row * n + i] = T::from_f64(-s2 * t1 + c2 * t2);
                }

                // Apply to A (cols k+1 and i)
                for row in 0..n {
                    let t1 = h[row * n + (k + 1)].to_f64();
                    let t2 = h[row * n + i].to_f64();
                    h[row * n + (k + 1)] = T::from_f64(c2 * t1 + s2 * t2);
                    h[row * n + i] = T::from_f64(-s2 * t1 + c2 * t2);
                }

                // Accumulate into Z
                for row in 0..n {
                    let t1 = z[row * n + (k + 1)].to_f64();
                    let t2 = z[row * n + i].to_f64();
                    z[row * n + (k + 1)] = T::from_f64(c2 * t1 + s2 * t2);
                    z[row * n + i] = T::from_f64(-s2 * t1 + c2 * t2);
                }
            }
        }
    }
}

/// Double-shift QZ iteration (Francis's implicit algorithm)
///
/// This algorithm performs two QZ steps implicitly in a single iteration,
/// allowing computation entirely in real arithmetic even for complex eigenvalues.
/// The double shift is computed from the trailing 2×2 block of the pencil.
fn qz_double_shift_iteration<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
) {
    if n < 2 {
        return;
    }

    let max_iter = 30 * n;
    let eps = T::epsilon_val();

    // Active submatrix indices [ilo, ihi)
    let mut ihi = n;

    for _iter in 0..max_iter {
        // Deflation: check for converged eigenvalues at the bottom
        while ihi > 1 {
            let i = ihi - 1;
            let h_ii = h[(i - 1) * n + (i - 1)].to_f64().abs();
            let h_ip1 = h[i * n + i].to_f64().abs();
            let threshold = eps * (h_ii + h_ip1).max(1.0);

            if h[i * n + (i - 1)].to_f64().abs() <= threshold {
                h[i * n + (i - 1)] = T::zero();
                ihi -= 1;
            } else {
                break;
            }
        }

        if ihi <= 1 {
            break;
        }

        // Find ilo: the start of the active unreduced block
        let mut ilo = 0;
        for i in (1..ihi).rev() {
            let h_ii = h[(i - 1) * n + (i - 1)].to_f64().abs();
            let h_ip1 = h[i * n + i].to_f64().abs();
            let threshold = eps * (h_ii + h_ip1).max(1.0);

            if h[i * n + (i - 1)].to_f64().abs() <= threshold {
                h[i * n + (i - 1)] = T::zero();
                ilo = i;
                break;
            }
        }

        // If block size is 1, we're done with this eigenvalue
        if ihi - ilo <= 1 {
            ihi = ilo;
            continue;
        }

        // If block size is 2, handle directly
        if ihi - ilo == 2 {
            // 2×2 block converged - leave as is (may contain complex conjugate pair)
            ihi = ilo;
            continue;
        }

        // Perform implicit double-shift QZ step on active block [ilo, ihi)
        implicit_double_shift_qz_step::<T>(h, r, q, z, n, ilo, ihi);
    }

    // Final cleanup of small subdiagonals
    for i in 1..n {
        let h_ii = h[(i - 1) * n + (i - 1)].to_f64().abs();
        let h_ip1 = h[i * n + i].to_f64().abs();
        let threshold = eps * (h_ii + h_ip1).max(1.0);

        if h[i * n + (i - 1)].to_f64().abs() <= threshold {
            h[i * n + (i - 1)] = T::zero();
        }
    }
}

/// Implicit double-shift QZ step on the active block [ilo, ihi)
///
/// Uses Francis's bulge-chasing algorithm with implicit double shift.
/// The two shifts are the eigenvalues of the trailing 2×2 block of H * inv(R).
fn implicit_double_shift_qz_step<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
    ilo: usize,
    ihi: usize,
) {
    let eps = T::epsilon_val();

    // Compute the implicit double shift from trailing 2×2 block
    // We work with M = H * inv(R) implicitly
    let m = ihi - 1;

    // Get elements from trailing 2×2 block of H and R
    let h_mm = h[(m - 1) * n + (m - 1)].to_f64();
    let h_m1m = h[m * n + (m - 1)].to_f64();
    let h_mm1 = h[(m - 1) * n + m].to_f64();
    let h_m1m1 = h[m * n + m].to_f64();

    let r_mm = r[(m - 1) * n + (m - 1)].to_f64();
    let r_mm1 = r[(m - 1) * n + m].to_f64();
    let r_m1m1 = r[m * n + m].to_f64();

    // Compute trace and determinant of trailing 2×2 of H*inv(R)
    // For generalized problem, eigenvalues satisfy det(H - λR) = 0
    let (s1, s2) = if r_mm.abs() > eps && r_m1m1.abs() > eps {
        // M = H * inv(R) for 2×2 block
        // [h_mm, h_mm1]   [r_mm, r_mm1]^-1
        // [h_m1m, h_m1m1] [0,    r_m1m1]
        let inv_r_mm = 1.0 / r_mm;
        let inv_r_m1m1 = 1.0 / r_m1m1;

        let m00 = h_mm * inv_r_mm;
        let m01 = (h_mm1 - h_mm * r_mm1 * inv_r_mm) * inv_r_m1m1;
        let m10 = h_m1m * inv_r_mm;
        let m11 = (h_m1m1 - h_m1m * r_mm1 * inv_r_mm) * inv_r_m1m1;

        let trace = m00 + m11;
        let det = m00 * m11 - m01 * m10;
        (trace, det)
    } else {
        // Fallback: use H elements directly
        let trace = h_mm + h_m1m1;
        let det = h_mm * h_m1m1 - h_mm1 * h_m1m;
        (trace, det)
    };

    // First column of (H - s1*R)(H - s2*R) = H^2 - s*H*R + p*R^2
    // where s = s1 + s2 = trace, p = s1*s2 = det
    // We compute first column implicitly using H and R at column ilo
    let h00 = h[ilo * n + ilo].to_f64();
    let h10 = h[(ilo + 1) * n + ilo].to_f64();
    let h20 = if ilo + 2 < n {
        h[(ilo + 2) * n + ilo].to_f64()
    } else {
        0.0
    };
    let h01 = h[ilo * n + (ilo + 1)].to_f64();
    let h11 = h[(ilo + 1) * n + (ilo + 1)].to_f64();

    let r00 = r[ilo * n + ilo].to_f64();
    let r01 = r[ilo * n + (ilo + 1)].to_f64();
    let r11 = r[(ilo + 1) * n + (ilo + 1)].to_f64();

    // Compute first column of M^2 - s*M + p*I implicitly
    // v = first column of (H - s1*R)(H - s2*R)
    let v0 = h00 * h00 + h01 * h10 - s1 * h00 * r00 + s2 * r00 * r00;
    let v1 = h10 * (h00 + h11 - s1 * r00 - s1 * r11 + s1 * r01 * h10 / h00.max(eps));
    let v2 = h10 * h20;

    // Householder to introduce bulge
    let v_norm = (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
    if v_norm < eps {
        return;
    }

    let beta = if v0 >= 0.0 { -v_norm } else { v_norm };
    let v0_h = v0 - beta;
    let tau = -v0_h / beta;

    // Normalize Householder vector
    let h_norm = (v0_h * v0_h + v1 * v1 + v2 * v2).sqrt();
    if h_norm < eps {
        return;
    }
    let u0 = v0_h / h_norm;
    let u1 = v1 / h_norm;
    let u2 = v2 / h_norm;

    // Apply initial Householder from the left to H and R
    let p_end = (ilo + 3).min(ihi);
    apply_householder_left(h, n, ilo, p_end, 0, n, u0, u1, u2, tau);
    apply_householder_left(r, n, ilo, p_end, ilo, n, u0, u1, u2, tau);

    // Accumulate into Q
    apply_householder_right(q, n, 0, n, ilo, p_end, u0, u1, u2, tau);

    // Chase the bulge down
    for k in ilo..(ihi - 2) {
        // Determine size of current Householder (3 or 2 at the end)
        let p_size = if k + 3 < ihi { 3 } else { 2 };

        // Restore R to upper triangular with column rotation
        for i in (k + 1)..(k + p_size).min(ihi) {
            let r1 = r[k * n + k].to_f64();
            let r2 = r[i * n + k].to_f64();

            if r2.abs() < eps {
                continue;
            }

            let rr = (r1 * r1 + r2 * r2).sqrt();
            let c = r1 / rr;
            let s = r2 / rr;

            // Column rotation on R
            for row in 0..ihi {
                let t1 = r[row * n + k].to_f64();
                let t2 = r[row * n + i].to_f64();
                r[row * n + k] = T::from_f64(c * t1 + s * t2);
                r[row * n + i] = T::from_f64(-s * t1 + c * t2);
            }

            // Same on H
            for row in 0..ihi {
                let t1 = h[row * n + k].to_f64();
                let t2 = h[row * n + i].to_f64();
                h[row * n + k] = T::from_f64(c * t1 + s * t2);
                h[row * n + i] = T::from_f64(-s * t1 + c * t2);
            }

            // Accumulate into Z
            for row in 0..n {
                let t1 = z[row * n + k].to_f64();
                let t2 = z[row * n + i].to_f64();
                z[row * n + k] = T::from_f64(c * t1 + s * t2);
                z[row * n + i] = T::from_f64(-s * t1 + c * t2);
            }
        }

        // Zero out elements below subdiagonal in column k of H
        if k + 2 < ihi {
            let w0 = h[(k + 1) * n + k].to_f64();
            let w1 = h[(k + 2) * n + k].to_f64();
            let w2 = if k + 3 < ihi {
                h[(k + 3) * n + k].to_f64()
            } else {
                0.0
            };

            let w_size = if k + 3 < ihi { 3 } else { 2 };
            let w_norm = if w_size == 3 {
                (w0 * w0 + w1 * w1 + w2 * w2).sqrt()
            } else {
                (w0 * w0 + w1 * w1).sqrt()
            };

            if w_norm > eps {
                let beta_w = if w0 >= 0.0 { -w_norm } else { w_norm };
                let w0_h = w0 - beta_w;
                let tau_w = -w0_h / beta_w;

                let h_norm_w = if w_size == 3 {
                    (w0_h * w0_h + w1 * w1 + w2 * w2).sqrt()
                } else {
                    (w0_h * w0_h + w1 * w1).sqrt()
                };

                if h_norm_w > eps {
                    let uu0 = w0_h / h_norm_w;
                    let uu1 = w1 / h_norm_w;
                    let uu2 = if w_size == 3 { w2 / h_norm_w } else { 0.0 };

                    let p_start = k + 1;
                    let p_end_w = (k + 1 + w_size).min(ihi);

                    apply_householder_left(h, n, p_start, p_end_w, k, n, uu0, uu1, uu2, tau_w);
                    apply_householder_left(r, n, p_start, p_end_w, k + 1, n, uu0, uu1, uu2, tau_w);
                    apply_householder_right(q, n, 0, n, p_start, p_end_w, uu0, uu1, uu2, tau_w);
                }
            }
        }
    }
}

/// Apply Householder reflection from the left: H[rows, cols] = (I - tau * u * u^T) * H[rows, cols]
fn apply_householder_left<T: Element + LinalgElement>(
    a: &mut [T],
    lda: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    u0: f64,
    u1: f64,
    u2: f64,
    tau: f64,
) {
    let size = row_end - row_start;
    if size < 2 {
        return;
    }

    for j in col_start..col_end {
        let mut dot = u0 * a[row_start * lda + j].to_f64();
        if size >= 2 {
            dot += u1 * a[(row_start + 1) * lda + j].to_f64();
        }
        if size >= 3 {
            dot += u2 * a[(row_start + 2) * lda + j].to_f64();
        }

        let factor = tau * dot;
        a[row_start * lda + j] = T::from_f64(a[row_start * lda + j].to_f64() - factor * u0);
        if size >= 2 {
            a[(row_start + 1) * lda + j] =
                T::from_f64(a[(row_start + 1) * lda + j].to_f64() - factor * u1);
        }
        if size >= 3 {
            a[(row_start + 2) * lda + j] =
                T::from_f64(a[(row_start + 2) * lda + j].to_f64() - factor * u2);
        }
    }
}

/// Apply Householder reflection from the right: A[rows, cols] = A[rows, cols] * (I - tau * u * u^T)
fn apply_householder_right<T: Element + LinalgElement>(
    a: &mut [T],
    lda: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    u0: f64,
    u1: f64,
    u2: f64,
    tau: f64,
) {
    let size = col_end - col_start;
    if size < 2 {
        return;
    }

    for i in row_start..row_end {
        let mut dot = u0 * a[i * lda + col_start].to_f64();
        if size >= 2 {
            dot += u1 * a[i * lda + (col_start + 1)].to_f64();
        }
        if size >= 3 {
            dot += u2 * a[i * lda + (col_start + 2)].to_f64();
        }

        let factor = tau * dot;
        a[i * lda + col_start] = T::from_f64(a[i * lda + col_start].to_f64() - factor * u0);
        if size >= 2 {
            a[i * lda + (col_start + 1)] =
                T::from_f64(a[i * lda + (col_start + 1)].to_f64() - factor * u1);
        }
        if size >= 3 {
            a[i * lda + (col_start + 2)] =
                T::from_f64(a[i * lda + (col_start + 2)].to_f64() - factor * u2);
        }
    }
}

/// Extract generalized eigenvalues from QZ decomposition
fn extract_generalized_eigenvalues<T: Element + LinalgElement>(
    s: &[T],
    t: &[T],
    n: usize,
) -> (Vec<T>, Vec<T>) {
    let mut real_parts = Vec::with_capacity(n);
    let mut imag_parts = Vec::with_capacity(n);
    let eps = T::epsilon_val();

    let mut i = 0;
    while i < n {
        // Check for 2×2 block (complex eigenvalues)
        if i + 1 < n && s[(i + 1) * n + i].to_f64().abs() > eps {
            // 2×2 block - compute complex eigenvalues
            let a = s[i * n + i].to_f64();
            let b = s[i * n + (i + 1)].to_f64();
            let c = s[(i + 1) * n + i].to_f64();
            let d = s[(i + 1) * n + (i + 1)].to_f64();

            let t_ii = t[i * n + i].to_f64();
            let t_ip1 = t[(i + 1) * n + (i + 1)].to_f64();

            // Eigenvalues of 2×2 block
            let trace = a + d;
            let det = a * d - b * c;
            let disc = trace * trace - 4.0 * det;

            if disc < 0.0 {
                let scale = (t_ii * t_ip1).abs().sqrt().max(1.0);
                let re = trace / 2.0 / scale;
                let im = (-disc).sqrt() / 2.0 / scale;
                real_parts.push(T::from_f64(re));
                imag_parts.push(T::from_f64(im));
                real_parts.push(T::from_f64(re));
                imag_parts.push(T::from_f64(-im));
            } else {
                let sqrt_disc = disc.sqrt();
                real_parts.push(T::from_f64((trace + sqrt_disc) / 2.0 / t_ii.abs().max(1.0)));
                imag_parts.push(T::zero());
                real_parts.push(T::from_f64(
                    (trace - sqrt_disc) / 2.0 / t_ip1.abs().max(1.0),
                ));
                imag_parts.push(T::zero());
            }
            i += 2;
        } else {
            // 1×1 block - real eigenvalue
            let s_ii = s[i * n + i].to_f64();
            let t_ii = t[i * n + i].to_f64();

            let lambda = if t_ii.abs() > eps {
                s_ii / t_ii
            } else if s_ii >= 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            real_parts.push(T::from_f64(lambda));
            imag_parts.push(T::zero());
            i += 1;
        }
    }

    (real_parts, imag_parts)
}
