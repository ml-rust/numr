//! Generic GMRES implementation
//!
//! Right-preconditioned GMRES with restarts using Arnoldi iteration
//! and Givens rotations for numerical stability.
//!
//! All operations use tensor primitives - no GPU↔CPU transfers.

use crate::algorithm::iterative::{GmresOptions, GmresResult, PreconditionerType};
use crate::algorithm::sparse_linalg::{IluDecomposition, IluOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

/// Generic GMRES implementation
///
/// Implements right-preconditioned GMRES with Arnoldi iteration and Givens rotations.
/// All operations are performed via tensor primitives to ensure no GPU↔CPU transfers.
///
/// # Type Parameters
///
/// * `R` - Runtime (CPU, CUDA, WebGPU)
/// * `C` - Client type implementing required operations
pub fn gmres_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: GmresOptions,
) -> Result<GmresResult<R>>
where
    R: Runtime,
    R::Client: SparseOps<R>,
    C: SparseLinAlgAlgorithms<R>
        + SparseOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>,
{
    let n = crate::algorithm::iterative::validate_iterative_inputs(a.shape, b, x0)?;
    let device = b.device();
    let dtype = b.dtype();

    // Validate dtype is floating point
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType { dtype, op: "gmres" });
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::zeros(&[n], dtype, device),
    };

    // Compute preconditioner if requested
    let precond = match options.preconditioner {
        PreconditionerType::None => None,
        PreconditionerType::Ilu0 => {
            let ilu = client.ilu0(a, IluOptions::default())?;
            Some(ilu)
        }
        PreconditionerType::Ic0 => {
            // IC0 returns IcDecomposition, but we need ILU interface
            // For IC0, M = L * L^T, so M^-1 * v = L^-T * (L^-1 * v)
            return Err(Error::Internal(
                "IC0 preconditioner not yet supported for GMRES - use ILU0".to_string(),
            ));
        }
    };

    // Compute initial residual norm ||b||
    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        // b is essentially zero, x = 0 is the solution
        return Ok(GmresResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    let m = options.restart;
    let mut total_iterations = 0;

    // Outer restart loop
    for _restart in 0..(options.max_iter / m + 1) {
        // r = b - A @ x
        let ax = a.spmv(&x)?;
        let r = client.sub(b, &ax)?;

        // beta = ||r||
        let beta = vector_norm(client, &r)?;

        // Check convergence
        if beta < options.atol || beta / b_norm < options.rtol {
            return Ok(GmresResult {
                solution: x,
                iterations: total_iterations,
                residual_norm: beta,
                converged: true,
            });
        }

        // v[0] = r / beta
        let v0 = client.mul_scalar(&r, 1.0 / beta)?;

        // Krylov basis vectors V = [v0, v1, ..., vm]
        let mut v_basis: Vec<Tensor<R>> = vec![v0];

        // Preconditioned basis vectors Z = [z0, z1, ..., zm] where z_j = M^-1 @ v_j
        // We store these to avoid recomputing during solution update
        let mut z_basis: Vec<Tensor<R>> = Vec::with_capacity(m);

        // Hessenberg matrix H (m+1 x m) stored as columns
        // H[i][j] = h_{i,j} where i is row, j is column
        // We store the upper Hessenberg entries
        let mut h_matrix: Vec<Vec<f64>> = Vec::with_capacity(m);

        // Givens rotation coefficients
        let mut cs: Vec<f64> = Vec::with_capacity(m);
        let mut sn: Vec<f64> = Vec::with_capacity(m);

        // Right-hand side of least squares: g = beta * e_1
        let mut g: Vec<f64> = vec![beta];

        let mut j = 0;
        while j < m && total_iterations < options.max_iter {
            total_iterations += 1;

            // z_j = M^-1 @ v[j], w = A @ z_j
            let vj = &v_basis[j];
            let z = apply_preconditioner(client, &precond, vj)?;
            let w = a.spmv(&z)?;
            z_basis.push(z); // Store for later use in solution update

            // Arnoldi orthogonalization (modified Gram-Schmidt)
            let mut h_col: Vec<f64> = Vec::with_capacity(j + 2);
            let mut w_current = w;

            for i in 0..=j {
                // h_{i,j} = <w, v_i>
                let h_ij = vector_dot(client, &w_current, &v_basis[i])?;
                h_col.push(h_ij);

                // w = w - h_{i,j} * v_i
                let scaled_vi = client.mul_scalar(&v_basis[i], h_ij)?;
                w_current = client.sub(&w_current, &scaled_vi)?;
            }

            // h_{j+1,j} = ||w||
            let h_jp1_j = vector_norm(client, &w_current)?;
            h_col.push(h_jp1_j);

            // Apply previous Givens rotations to the new column
            for i in 0..j {
                let temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1];
                h_col[i + 1] = -sn[i] * h_col[i] + cs[i] * h_col[i + 1];
                h_col[i] = temp;
            }

            // Compute new Givens rotation
            let (c, s, r) = givens_rotation(h_col[j], h_col[j + 1]);
            cs.push(c);
            sn.push(s);

            // Apply to H column
            h_col[j] = r;
            h_col[j + 1] = 0.0;

            // Apply to g
            let g_old_j = g[j];
            g.push(-s * g_old_j);
            g[j] = c * g_old_j;

            h_matrix.push(h_col);

            // Check convergence (|g[j+1]| is the residual norm)
            let res_norm = g[j + 1].abs();
            if res_norm < options.atol || res_norm / b_norm < options.rtol {
                // Solve upper triangular system H @ y = g
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);

                // x = x + Z @ y (using stored preconditioned basis)
                x = update_solution(client, &x, &z_basis, &y)?;

                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: res_norm,
                    converged: true,
                });
            }

            // Check for lucky breakdown
            if h_jp1_j < 1e-14 {
                // Solve and update solution
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);
                x = update_solution(client, &x, &z_basis, &y)?;

                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: g[j + 1].abs(),
                    converged: true,
                });
            }

            // v[j+1] = w / h_{j+1,j}
            let v_jp1 = client.mul_scalar(&w_current, 1.0 / h_jp1_j)?;
            v_basis.push(v_jp1);

            j += 1;
        }

        // End of restart cycle - update solution
        if !h_matrix.is_empty() {
            let y = solve_upper_triangular(&h_matrix, &g[..j]);
            x = update_solution(client, &x, &z_basis, &y)?;
        }
    }

    // Compute final residual
    let ax = a.spmv(&x)?;
    let r = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r)?;

    Ok(GmresResult {
        solution: x,
        iterations: total_iterations,
        residual_norm: final_residual,
        converged: false,
    })
}

/// Compute vector L2 norm: ||v|| = sqrt(sum(v^2))
///
/// Uses optimized `item()` for scalar extraction (single element copy, no Vec allocation).
pub(super) fn vector_norm<R, C>(client: &C, v: &Tensor<R>) -> Result<f64>
where
    R: Runtime,
    C: BinaryOps<R> + UnaryOps<R> + ReduceOps<R>,
{
    // v^2
    let v_sq = client.mul(v, v)?;
    // sum(v^2) over all dimensions → scalar
    let ndim = v_sq.ndim();
    let dims: Vec<usize> = (0..ndim).collect();
    let sum_sq = client.sum(&v_sq, &dims, false)?;
    // sqrt(sum(v^2))
    let norm_tensor = client.sqrt(&sum_sq)?;

    // Extract scalar using optimized item() (no Vec allocation)
    match norm_tensor.dtype() {
        DType::F32 => Ok(norm_tensor.item::<f32>()? as f64),
        DType::F64 => Ok(norm_tensor.item::<f64>()?),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "vector_norm",
        }),
    }
}

/// Compute vector dot product: <u, v> = sum(u * v)
///
/// Uses optimized `item()` for scalar extraction (single element copy, no Vec allocation).
pub(super) fn vector_dot<R, C>(client: &C, u: &Tensor<R>, v: &Tensor<R>) -> Result<f64>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R>,
{
    // u * v
    let uv = client.mul(u, v)?;
    // sum(u * v) over all dimensions → scalar
    let ndim = uv.ndim();
    let dims: Vec<usize> = (0..ndim).collect();
    let dot_tensor = client.sum(&uv, &dims, false)?;

    // Extract scalar using optimized item() (no Vec allocation)
    match dot_tensor.dtype() {
        DType::F32 => Ok(dot_tensor.item::<f32>()? as f64),
        DType::F64 => Ok(dot_tensor.item::<f64>()?),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "vector_dot",
        }),
    }
}

/// Apply ILU(0) preconditioner: z = M^-1 @ v = U^-1 @ (L^-1 @ v)
pub(super) fn apply_preconditioner<R, C>(
    client: &C,
    precond: &Option<IluDecomposition<R>>,
    v: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SparseLinAlgAlgorithms<R>,
{
    match precond {
        None => Ok(v.clone()),
        Some(ilu) => {
            // L^-1 @ v (forward substitution, unit diagonal)
            let y = client.sparse_solve_triangular(&ilu.l, v, true, true)?;
            // U^-1 @ y (backward substitution)
            client.sparse_solve_triangular(&ilu.u, &y, false, false)
        }
    }
}

/// Compute Givens rotation coefficients
///
/// Given a and b, computes c, s, r such that:
/// [c  s] [a]   [r]
/// [-s c] [b] = [0]
///
/// where c^2 + s^2 = 1 and r = sqrt(a^2 + b^2) >= 0
///
/// This implementation ensures r >= 0 for numerical robustness,
/// following the convention used in LAPACK's DLARTG.
#[inline]
fn givens_rotation(a: f64, b: f64) -> (f64, f64, f64) {
    if b.abs() < 1e-15 {
        // b ≈ 0, no rotation needed
        // Ensure r >= 0 by adjusting sign
        if a >= 0.0 {
            (1.0, 0.0, a)
        } else {
            (-1.0, 0.0, -a)
        }
    } else if a.abs() < 1e-15 {
        // a ≈ 0, 90-degree rotation
        // r = |b| >= 0
        (0.0, b.signum(), b.abs())
    } else {
        // General case: use hypot for numerical stability
        let r = a.hypot(b); // Always >= 0
        let c = a / r;
        let s = b / r;
        (c, s, r)
    }
}

/// Solve upper triangular system R @ y = g via back substitution
///
/// h_matrix[j] contains column j of R (upper triangular)
fn solve_upper_triangular(h_matrix: &[Vec<f64>], g: &[f64]) -> Vec<f64> {
    let m = g.len();
    let mut y = vec![0.0; m];

    for i in (0..m).rev() {
        let mut sum = g[i];
        for j in (i + 1)..m {
            sum -= h_matrix[j][i] * y[j];
        }
        if h_matrix[i][i].abs() > 1e-15 {
            y[i] = sum / h_matrix[i][i];
        }
    }

    y
}

/// Update solution: x = x + Z @ y where Z contains preconditioned basis vectors
fn update_solution<R, C>(
    client: &C,
    x: &Tensor<R>,
    z_basis: &[Tensor<R>],
    y: &[f64],
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + ScalarOps<R>,
{
    let m = y.len();
    let mut delta = Tensor::<R>::zeros(x.shape(), x.dtype(), x.device());

    // delta = sum_j y[j] * z[j] (z vectors already preconditioned)
    for j in 0..m {
        if y[j].abs() > 1e-15 {
            let scaled_z = client.mul_scalar(&z_basis[j], y[j])?;
            delta = client.add(&delta, &scaled_z)?;
        }
    }

    // x = x + delta
    client.add(x, &delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_givens_rotation() {
        // Test case 1: a = 3, b = 4 -> r = 5
        let (c, s, r) = givens_rotation(3.0, 4.0);
        assert!((c * c + s * s - 1.0).abs() < 1e-10, "c^2 + s^2 = 1");
        assert!((r - 5.0).abs() < 1e-10, "r = 5");
        assert!(r >= 0.0, "r must be non-negative");
        assert!((c * 3.0 + s * 4.0 - 5.0).abs() < 1e-10, "rotation works");
        assert!((-s * 3.0 + c * 4.0).abs() < 1e-10, "zeroes out b");

        // Test case 2: b = 0, a > 0
        let (c, s, r) = givens_rotation(5.0, 0.0);
        assert_eq!(c, 1.0);
        assert_eq!(s, 0.0);
        assert_eq!(r, 5.0);
        assert!(r >= 0.0);

        // Test case 3: b = 0, a < 0 (must ensure r >= 0)
        let (c, s, r) = givens_rotation(-5.0, 0.0);
        assert_eq!(c, -1.0);
        assert_eq!(s, 0.0);
        assert_eq!(r, 5.0);
        assert!(r >= 0.0, "r must be non-negative even for negative a");
        // Verify: c*a + s*b = -1*(-5) + 0*0 = 5 = r ✓
        assert!((c * (-5.0) + s * 0.0 - r).abs() < 1e-10);

        // Test case 4: a = 0, b > 0
        let (c, s, r) = givens_rotation(0.0, 3.0);
        assert_eq!(c, 0.0);
        assert_eq!(s, 1.0);
        assert_eq!(r, 3.0);
        assert!(r >= 0.0);

        // Test case 5: a = 0, b < 0
        let (c, s, r) = givens_rotation(0.0, -3.0);
        assert_eq!(c, 0.0);
        assert_eq!(s, -1.0);
        assert_eq!(r, 3.0);
        assert!(r >= 0.0, "r must be non-negative for negative b");

        // Test case 6: negative a and b
        let (c, s, r) = givens_rotation(-3.0, -4.0);
        assert!((c * c + s * s - 1.0).abs() < 1e-10, "c^2 + s^2 = 1");
        assert!((r - 5.0).abs() < 1e-10, "r = 5");
        assert!(r >= 0.0, "r must be non-negative");
        // Verify rotation: c*a + s*b = r, -s*a + c*b = 0
        assert!(
            (c * (-3.0) + s * (-4.0) - r).abs() < 1e-10,
            "rotation gives r"
        );
        assert!((-s * (-3.0) + c * (-4.0)).abs() < 1e-10, "zeroes out b");
    }

    #[test]
    fn test_solve_upper_triangular() {
        // R = [[2, 1], [0, 3]], g = [3, 6]
        // Solution: y[1] = 6/3 = 2, y[0] = (3 - 1*2)/2 = 0.5
        let h_matrix = vec![vec![2.0, 0.0], vec![1.0, 3.0]];
        let g = vec![3.0, 6.0];
        let y = solve_upper_triangular(&h_matrix, &g);
        assert!((y[0] - 0.5).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
    }
}
