//! Triangular solves (Lx = b, Ux = b) for CPU, one matrix or a batch.
//!
//! A batch runs matrix by matrix. Each matrix uses the substitution order
//! of the single-matrix solve, so a batched result equals solving each
//! matrix on its own.

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    TriangularSolveShape, linalg_demote, linalg_promote, validate_linalg_dtype,
    validate_triangular_solve,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::{RuntimeClient, ensure_contiguous};
use crate::tensor::Tensor;

/// Which triangle a solve reads.
#[derive(Clone, Copy)]
enum Triangle {
    Lower { unit_diagonal: bool },
    Upper,
}

/// Forward substitution for lower triangular systems, single or batched.
pub fn solve_triangular_lower_impl(
    client: &CpuClient,
    l: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    unit_diagonal: bool,
) -> Result<Tensor<CpuRuntime>> {
    solve_triangular_impl(
        client,
        l,
        b,
        Triangle::Lower { unit_diagonal },
        "solve_triangular_lower",
    )
}

/// Backward substitution for upper triangular systems, single or batched.
pub fn solve_triangular_upper_impl(
    client: &CpuClient,
    u: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    solve_triangular_impl(client, u, b, Triangle::Upper, "solve_triangular_upper")
}

fn solve_triangular_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    triangle: Triangle,
    op: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let shape = validate_triangular_solve(op, a.shape(), b.shape())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (b, _) = linalg_promote(client, b)?;
    let a = ensure_contiguous(&*a)?;
    let b = ensure_contiguous(&*b)?;

    let result = match a.dtype() {
        DType::F32 => solve_typed::<f32>(client, &a, &b, &shape, triangle),
        DType::F64 => solve_typed::<f64>(client, &a, &b, &shape, triangle),
        dtype => Err(Error::UnsupportedDType { dtype, op }),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn solve_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    shape: &TriangularSolveShape,
    triangle: Triangle,
) -> Result<Tensor<CpuRuntime>> {
    let (n, m) = (shape.n, shape.nrhs);
    let a_data: Vec<T> = a.try_to_vec()?;
    let b_data: Vec<T> = b.try_to_vec()?;
    let mut x: Vec<T> = vec![T::zero(); shape.out_numel()];

    let (mat, rhs) = (n * n, n * m);
    for bi in 0..shape.batch {
        let a_mat = &a_data[bi * mat..(bi + 1) * mat];
        let b_mat = &b_data[bi * rhs..(bi + 1) * rhs];
        let x_mat = &mut x[bi * rhs..(bi + 1) * rhs];
        match triangle {
            Triangle::Lower { unit_diagonal } => {
                forward_substitute(a_mat, b_mat, x_mat, n, m, unit_diagonal)
            }
            Triangle::Upper => backward_substitute(a_mat, b_mat, x_mat, n, m),
        }
    }

    Tensor::<CpuRuntime>::from_slice(&x, &shape.out_shape, client.device())
}

/// Solve one `n x n` lower system against `m` columns, row-major.
fn forward_substitute<T: Element + LinalgElement>(
    l: &[T],
    b: &[T],
    x: &mut [T],
    n: usize,
    m: usize,
    unit_diagonal: bool,
) {
    for rhs in 0..m {
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                sum = sum + l[i * n + j] * x[j * m + rhs];
            }
            let result = b[i * m + rhs] - sum;
            x[i * m + rhs] = if unit_diagonal {
                result
            } else {
                result / l[i * n + i]
            };
        }
    }
}

/// Solve one `n x n` upper system against `m` columns, row-major.
fn backward_substitute<T: Element + LinalgElement>(
    u: &[T],
    b: &[T],
    x: &mut [T],
    n: usize,
    m: usize,
) {
    for rhs in 0..m {
        for i in (0..n).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..n {
                sum = sum + u[i * n + j] * x[j * m + rhs];
            }
            x[i * m + rhs] = (b[i * m + rhs] - sum) / u[i * n + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::create_client;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;
    use crate::algorithm::linalg::triangular_test_data::{
        CASES, Case, gdn_like_unit_lower, host_solve, identity, max_abs_diff, to_f32, to_f64,
        triangular_batch, values,
    };

    /// Solve `case` batched, then each matrix alone. Return both results.
    fn batched_and_separate(
        client: &CpuClient,
        case: &Case,
        a: &[f32],
        b: &[f32],
        lower: bool,
        unit: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let device = client.device();
        let solve = |a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>| {
            if lower {
                client.solve_triangular_lower(a, b, unit).unwrap()
            } else {
                client.solve_triangular_upper(a, b).unwrap()
            }
        };
        let at = Tensor::<CpuRuntime>::from_slice(a, &case.a_shape(), device).unwrap();
        let bt = Tensor::<CpuRuntime>::from_slice(b, &case.b_shape(), device).unwrap();
        let batched = solve(&at, &bt);
        assert_eq!(batched.shape(), case.b_shape().as_slice());

        let (n, m) = (case.n, case.m());
        let b_one: Vec<usize> = case.b_shape()[case.batch_dims.len()..].to_vec();
        let mut separate = Vec::with_capacity(case.batch() * n * m);
        for bi in 0..case.batch() {
            let a_i = &a[bi * n * n..(bi + 1) * n * n];
            let b_i = &b[bi * n * m..(bi + 1) * n * m];
            let a_i = Tensor::<CpuRuntime>::from_slice(a_i, &[n, n], device).unwrap();
            let b_i = Tensor::<CpuRuntime>::from_slice(b_i, &b_one, device).unwrap();
            separate.extend(solve(&a_i, &b_i).to_vec::<f32>());
        }
        (batched.to_vec::<f32>(), separate)
    }

    fn check_all_cases(lower: bool, unit: bool) {
        let client = create_client();
        for (ci, case) in CASES.iter().enumerate() {
            let seed = 0x7a1e_0000 + ci as u64;
            let a = triangular_batch(seed, case.batch(), case.n, lower, unit);
            let b = values(seed ^ 0xb, case.batch() * case.n * case.m());
            let (a32, b32) = (to_f32(&a), to_f32(&b));
            let (batched, separate) = batched_and_separate(&client, case, &a32, &b32, lower, unit);
            // Same substitution order per matrix: results are bit-identical.
            assert_eq!(batched, separate, "case {ci} lower={lower} unit={unit}");

            let want = host_solve(
                &to_f64(&a32),
                &to_f64(&b32),
                case.batch(),
                case.n,
                case.m(),
                lower,
                unit,
            );
            let diff = max_abs_diff(&to_f64(&batched), &want);
            assert!(diff < 1e-4, "case {ci} lower={lower} unit={unit}: {diff}");
        }
    }

    #[test]
    fn batched_lower_matches_separate() {
        check_all_cases(true, false);
    }

    #[test]
    fn batched_lower_unit_matches_separate() {
        check_all_cases(true, true);
    }

    #[test]
    fn batched_upper_matches_separate() {
        check_all_cases(false, false);
    }

    #[test]
    fn batched_f64_matches_host() {
        let client = create_client();
        let device = client.device();
        let case = &CASES[4];
        for lower in [true, false] {
            let a = triangular_batch(11, case.batch(), case.n, lower, false);
            let b = values(12, case.batch() * case.n * case.m());
            let at = Tensor::<CpuRuntime>::from_slice(&a, &case.a_shape(), device).unwrap();
            let bt = Tensor::<CpuRuntime>::from_slice(&b, &case.b_shape(), device).unwrap();
            let x = if lower {
                client.solve_triangular_lower(&at, &bt, false).unwrap()
            } else {
                client.solve_triangular_upper(&at, &bt).unwrap()
            };
            let want = host_solve(&a, &b, case.batch(), case.n, case.m(), lower, false);
            let diff = max_abs_diff(&x.to_vec::<f64>(), &want);
            assert!(diff < 1e-12, "lower={lower}: {diff}");
        }
    }

    #[test]
    fn batched_accepts_strided_input() {
        let client = create_client();
        let device = client.device();
        let (batch, n) = (3, 4);
        let a = to_f32(&triangular_batch(21, batch, n, false, false));
        let b = to_f32(&values(22, batch * n * 2));
        let u = Tensor::<CpuRuntime>::from_slice(&a, &[batch, n, n], device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&b, &[batch, n, 2], device).unwrap();
        // The transpose of an upper matrix is lower, as a non-contiguous view.
        let l_view = u.transpose(1, 2).unwrap();
        let l_copy = l_view.contiguous().unwrap();
        let from_view = client.solve_triangular_lower(&l_view, &b, false).unwrap();
        let from_copy = client.solve_triangular_lower(&l_copy, &b, false).unwrap();
        assert_eq!(from_view.to_vec::<f32>(), from_copy.to_vec::<f32>());
    }

    #[test]
    fn batched_shape_mismatch_errors() {
        let client = create_client();
        let device = client.device();
        let l = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 32], &[2, 4, 4], device).unwrap();
        for b_shape in [&[3, 4, 2][..], &[2, 5, 2], &[2, 4], &[1, 2, 4, 2]] {
            let numel: usize = b_shape.iter().product();
            let b =
                Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; numel], b_shape, device).unwrap();
            let lower = client.solve_triangular_lower(&l, &b, false).unwrap_err();
            let upper = client.solve_triangular_upper(&l, &b).unwrap_err();
            for err in [lower, upper] {
                let text = err.to_string();
                assert!(text.contains(&format!("{b_shape:?}")), "{text}");
                assert!(text.contains("[2, 4, 4]"), "{text}");
            }
        }
    }

    #[test]
    fn single_matrix_row_mismatch_errors() {
        let client = create_client();
        let device = client.device();
        let l = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 9], &[3, 3], device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], device).unwrap();
        assert!(matches!(
            client.solve_triangular_lower(&l, &b, false),
            Err(Error::ShapeMismatch { .. })
        ));
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[4, 2], device).unwrap();
        assert!(matches!(
            client.solve_triangular_upper(&l, &b),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    /// F32 inverse of `I + A` for a strictly lower `A` with entries near 0.8.
    ///
    /// Substitution is backward stable, and this recurrence contracts: each
    /// solved entry is about -0.8 times the running column sum, which shrinks
    /// by about 0.2 per row. Rounding error stays near the f32 unit roundoff
    /// (6e-8) times a small constant. The bound 1e-5 leaves wide margin.
    #[test]
    fn f32_gdn_like_inverse_matches_f64() {
        let client = create_client();
        let device = client.device();
        let n = 64;
        let l32 = to_f32(&gdn_like_unit_lower(0x0d17, n));
        let eye32 = to_f32(&identity(n));
        let l64 = to_f64(&l32);
        let eye64 = identity(n);

        let l = Tensor::<CpuRuntime>::from_slice(&l32, &[1, n, n], device).unwrap();
        let eye = Tensor::<CpuRuntime>::from_slice(&eye32, &[1, n, n], device).unwrap();
        let t32 = client.solve_triangular_lower(&l, &eye, true).unwrap();

        let l = Tensor::<CpuRuntime>::from_slice(&l64, &[1, n, n], device).unwrap();
        let eye = Tensor::<CpuRuntime>::from_slice(&eye64, &[1, n, n], device).unwrap();
        let t64 = client.solve_triangular_lower(&l, &eye, true).unwrap();

        let diff = max_abs_diff(&to_f64(&t32.to_vec::<f32>()), &t64.to_vec::<f64>());
        assert!(diff < 1e-5, "max abs diff {diff}");
    }
}
