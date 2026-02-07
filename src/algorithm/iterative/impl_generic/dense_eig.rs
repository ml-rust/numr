//! Dense eigensolvers for small matrices produced by Krylov methods
//!
//! These operate on CPU-side `Vec<Vec<f64>>` matrices (typically 5-50 dimension)
//! produced by Lanczos tridiagonalization or Arnoldi Hessenberg reduction.
//! They are NOT tensor operations — the small dense problem is always solved on CPU.

use super::super::types::WhichEigenvalues;

/// Convergence tolerance for Jacobi/QR iterations on dense matrices
const DENSE_EIG_TOL: f64 = 1e-14;

// ============================================================================
// Symmetric tridiagonal eigensolver (for Lanczos)
// ============================================================================

/// Solve symmetric tridiagonal eigenvalue problem using Jacobi rotations.
///
/// Input: diagonal entries `alphas` and off-diagonal entries `betas` of
/// the tridiagonal matrix T from Lanczos.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i][j]` is
/// the j-th component of the i-th eigenvector.
pub fn tridiagonal_eig(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = alphas.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![alphas[0]], vec![vec![1.0]]);
    }

    // Build full tridiagonal matrix (small, CPU-side)
    let mut t = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        t[i][i] = alphas[i];
        if i + 1 < n && i < betas.len() {
            t[i][i + 1] = betas[i];
            t[i + 1][i] = betas[i];
        }
    }

    // Eigen-decomposition via Jacobi rotations (simple, robust for small matrices)
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_sweeps = 100 * n * n;
    for _ in 0..max_sweeps {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if t[i][j].abs() > max_val {
                    max_val = t[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < DENSE_EIG_TOL {
            break;
        }

        // Compute Jacobi rotation angle
        let theta = if (t[p][p] - t[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * t[p][q] / (t[p][p] - t[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to T: T' = G^T T G
        let mut new_t = t.clone();
        for i in 0..n {
            if i != p && i != q {
                new_t[i][p] = c * t[i][p] + s * t[i][q];
                new_t[p][i] = new_t[i][p];
                new_t[i][q] = -s * t[i][p] + c * t[i][q];
                new_t[q][i] = new_t[i][q];
            }
        }
        new_t[p][p] = c * c * t[p][p] + 2.0 * c * s * t[p][q] + s * s * t[q][q];
        new_t[q][q] = s * s * t[p][p] - 2.0 * c * s * t[p][q] + c * c * t[q][q];
        new_t[p][q] = 0.0;
        new_t[q][p] = 0.0;
        t = new_t;

        // Accumulate eigenvectors: V' = V * G
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| t[i][i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| v[j][i]).collect()).collect();

    (eigenvalues, eigenvectors)
}

/// Select eigenvalue indices according to the `which` criterion (real eigenvalues).
pub fn select_eigenvalues(eigenvalues: &[f64], k: usize, which: &WhichEigenvalues) -> Vec<usize> {
    let n = eigenvalues.len();
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();

    match which {
        WhichEigenvalues::LargestMagnitude => {
            indices.sort_by(|&a, &b| {
                eigenvalues[b]
                    .abs()
                    .partial_cmp(&eigenvalues[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::SmallestMagnitude => {
            indices.sort_by(|&a, &b| {
                eigenvalues[a]
                    .abs()
                    .partial_cmp(&eigenvalues[b].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::LargestReal => {
            indices.sort_by(|&a, &b| {
                eigenvalues[b]
                    .partial_cmp(&eigenvalues[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::SmallestReal => {
            indices.sort_by(|&a, &b| {
                eigenvalues[a]
                    .partial_cmp(&eigenvalues[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::ClosestTo(sigma) => {
            indices.sort_by(|&a, &b| {
                let da = (eigenvalues[a] - sigma).abs();
                let db = (eigenvalues[b] - sigma).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    indices.truncate(k);
    indices
}

// ============================================================================
// General Hessenberg eigensolver (for Arnoldi)
// ============================================================================

/// Compute eigenvalues of an upper Hessenberg matrix using QR iteration
/// with Wilkinson shifts.
///
/// Returns `(real_parts, imag_parts, eigenvectors_of_H)`.
pub fn hessenberg_eig(h: &[Vec<f64>], m: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    if m == 0 {
        return (vec![], vec![], vec![]);
    }
    if m == 1 {
        return (vec![h[0][0]], vec![0.0], vec![vec![1.0]]);
    }

    let mut a = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            if i < h.len() && j < h[i].len() {
                a[i][j] = h[i][j];
            }
        }
    }

    // Schur vectors
    let mut q = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        q[i][i] = 1.0;
    }

    let max_iters = 200 * m;
    let mut p = m;

    let mut eig_real = vec![0.0f64; m];
    let mut eig_imag = vec![0.0f64; m];

    for _ in 0..max_iters {
        if p <= 1 {
            if p == 1 {
                eig_real[0] = a[0][0];
            }
            break;
        }

        // Check for deflation
        let mut deflated = false;
        for i in (1..p).rev() {
            let threshold = DENSE_EIG_TOL * (a[i - 1][i - 1].abs() + a[i][i].abs()).max(1e-20);
            if a[i][i - 1].abs() < threshold {
                a[i][i - 1] = 0.0;
                if i == p - 1 {
                    eig_real[p - 1] = a[p - 1][p - 1];
                    eig_imag[p - 1] = 0.0;
                    p -= 1;
                    deflated = true;
                    break;
                }
            }
        }
        if deflated {
            continue;
        }

        // Check for 2x2 block at bottom
        if p >= 2 {
            let i = p - 2;
            let tr = a[i][i] + a[i + 1][i + 1];
            let det = a[i][i] * a[i + 1][i + 1] - a[i][i + 1] * a[i + 1][i];
            let disc = tr * tr - 4.0 * det;

            if disc < 0.0 && (p < 3 || a[i][i - 1].abs() < DENSE_EIG_TOL) {
                eig_real[p - 2] = tr / 2.0;
                eig_real[p - 1] = tr / 2.0;
                eig_imag[p - 2] = (-disc).sqrt() / 2.0;
                eig_imag[p - 1] = -(-disc).sqrt() / 2.0;
                p -= 2;
                continue;
            }
        }

        // Wilkinson shift
        let shift = wilkinson_shift(&a, p);

        // QR step: A - shift*I = Q*R, then A = R*Q + shift*I
        for i in 0..p {
            a[i][i] -= shift;
        }

        // Givens QR
        for i in 0..p - 1 {
            let (c, s, _r) = super::super::helpers::givens_rotation(a[i][i], a[i + 1][i]);

            for j in 0..m {
                let t1 = a[i][j];
                let t2 = a[i + 1][j];
                a[i][j] = c * t1 + s * t2;
                a[i + 1][j] = -s * t1 + c * t2;
            }

            for j in 0..p.min(i + 3) {
                let t1 = a[j][i];
                let t2 = a[j][i + 1];
                a[j][i] = c * t1 + s * t2;
                a[j][i + 1] = -s * t1 + c * t2;
            }

            for j in 0..m {
                let t1 = q[j][i];
                let t2 = q[j][i + 1];
                q[j][i] = c * t1 + s * t2;
                q[j][i + 1] = -s * t1 + c * t2;
            }
        }

        for i in 0..p {
            a[i][i] += shift;
        }
    }

    // Extract remaining eigenvalues
    for i in 0..p {
        eig_real[i] = a[i][i];
        if i + 1 < p && a[i + 1][i].abs() > DENSE_EIG_TOL {
            let tr = a[i][i] + a[i + 1][i + 1];
            let det = a[i][i] * a[i + 1][i + 1] - a[i][i + 1] * a[i + 1][i];
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                eig_real[i] = tr / 2.0;
                eig_real[i + 1] = tr / 2.0;
                eig_imag[i] = (-disc).sqrt() / 2.0;
                eig_imag[i + 1] = -(-disc).sqrt() / 2.0;
            }
        }
    }

    let eigenvectors: Vec<Vec<f64>> = (0..m).map(|i| (0..m).map(|j| q[j][i]).collect()).collect();

    (eig_real, eig_imag, eigenvectors)
}

/// Wilkinson shift: eigenvalue of bottom-right 2x2 block closest to a[p-1][p-1].
fn wilkinson_shift(a: &[Vec<f64>], p: usize) -> f64 {
    if p < 2 {
        return a[p - 1][p - 1];
    }
    let i = p - 2;
    let tr = a[i][i] + a[i + 1][i + 1];
    let det = a[i][i] * a[i + 1][i + 1] - a[i][i + 1] * a[i + 1][i];
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let s1 = (tr + disc.sqrt()) / 2.0;
        let s2 = (tr - disc.sqrt()) / 2.0;
        if (s1 - a[p - 1][p - 1]).abs() < (s2 - a[p - 1][p - 1]).abs() {
            s1
        } else {
            s2
        }
    } else {
        a[p - 1][p - 1]
    }
}

/// Select eigenvalue indices for complex eigenvalues (Arnoldi).
pub fn select_eigenvalues_complex(
    real: &[f64],
    imag: &[f64],
    k: usize,
    which: &WhichEigenvalues,
) -> Vec<usize> {
    let n = real.len();
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();

    match which {
        WhichEigenvalues::LargestMagnitude => {
            indices.sort_by(|&a, &b| {
                let ma = real[a] * real[a] + imag[a] * imag[a];
                let mb = real[b] * real[b] + imag[b] * imag[b];
                mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::SmallestMagnitude => {
            indices.sort_by(|&a, &b| {
                let ma = real[a] * real[a] + imag[a] * imag[a];
                let mb = real[b] * real[b] + imag[b] * imag[b];
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::LargestReal => {
            indices.sort_by(|&a, &b| {
                real[b]
                    .partial_cmp(&real[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::SmallestReal => {
            indices.sort_by(|&a, &b| {
                real[a]
                    .partial_cmp(&real[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::ClosestTo(sigma) => {
            indices.sort_by(|&a, &b| {
                let da = (real[a] - sigma) * (real[a] - sigma) + imag[a] * imag[a];
                let db = (real[b] - sigma) * (real[b] - sigma) + imag[b] * imag[b];
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    indices.truncate(k);
    indices
}
