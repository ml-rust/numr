//! Minimum-norm least squares via SVD, shared by every backend.
//!
//! For `A = U diag(s) Vt`, the solution is `X = Vt^T (s_inv ⊙ (U^T B))`.
//! `s_inv` inverts each singular value above `rcond * max(s)` and zeroes the
//! rest. This is the solution LAPACK `gelsd` returns. It minimizes
//! `||A X - B||`. Among all minimizers it has the smallest norm.

use super::super::decompositions::LstsqResult;
use super::super::helpers::{linalg_demote, linalg_promote};
use super::super::lstsq_shape::{LstsqShape, validate_lstsq};
use super::super::traits::LinearAlgebraAlgorithms;
use super::svd_cutoff::{resolve_rcond, svd_cutoff};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOps, CompareOps, ConditionalOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps,
    TypeConversionOps, UnaryOps,
};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Ops the least-squares solver composes.
///
/// Every client that implements the listed traits implements this one.
pub trait LstsqOps<R: Runtime>:
    LinearAlgebraAlgorithms<R>
    + BinaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + MatmulOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
{
}

impl<R: Runtime, C> LstsqOps<R> for C where
    C: LinearAlgebraAlgorithms<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + MatmulOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
{
}

/// Minimum-norm least-squares solve of `A X = B` with a relative SVD cutoff.
///
/// Shapes follow [`validate_lstsq`]. `rcond` follows [`resolve_rcond`]. F16,
/// BF16 and FP8 compute in F32 and return in the input dtype. Each matrix of
/// a batch runs its own SVD on its own device. No data leaves the device.
///
/// # Errors
///
/// - `DTypeMismatch` when `a` and `b` differ in dtype.
/// - `UnsupportedDType` for a non-float dtype, or a float dtype the backend
///   does not compute.
/// - `InvalidArgument` or `ShapeMismatch` from [`validate_lstsq`].
/// - `InvalidArgument` from [`resolve_rcond`].
pub fn lstsq_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    rcond: Option<f64>,
) -> Result<LstsqResult<R>>
where
    R: Runtime<DType = DType>,
    C: LstsqOps<R>,
{
    let dtype = a.dtype();
    if b.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: b.dtype(),
        });
    }
    if !dtype.is_float() || !R::computes_float_dtype(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "lstsq" });
    }
    let shape = validate_lstsq(a.shape(), b.shape())?;
    let rcond = resolve_rcond(rcond, shape.m, shape.n, dtype)?;
    if shape.batch == 0 || shape.m == 0 || shape.n == 0 {
        return empty_result(&shape, dtype, a.device());
    }

    let (a_work, original_dtype) = linalg_promote(client, a)?;
    let (b_work, _) = linalg_promote(client, b)?;
    let (batch, m, n, k) = (shape.batch, shape.m, shape.n, shape.nrhs);
    let a3 = a_work.contiguous()?.reshape(&[batch, m, n])?;
    let b3 = b_work.contiguous()?.reshape(&[batch, m, k])?;

    let mut solutions = Vec::with_capacity(batch);
    let mut singular_values = Vec::with_capacity(batch);
    let mut ranks = Vec::with_capacity(batch);
    for i in 0..batch {
        let a_i = a3.narrow(0, i, 1)?.contiguous()?.reshape(&[m, n])?;
        let b_i = b3.narrow(0, i, 1)?.contiguous()?.reshape(&[m, k])?;
        let one = solve_one(client, &a_i, &b_i, rcond)?;
        if let Some(x) = one.solution {
            solutions.push(x);
        }
        singular_values.push(one.singular_values);
        ranks.push(one.rank);
    }

    let solution = if k == 0 {
        Tensor::<R>::zeros(&shape.solution_shape(), a_work.dtype(), a.device())?
    } else {
        join(client, &solutions)?.reshape(&shape.solution_shape())?
    };
    let singular_values =
        join(client, &singular_values)?.reshape(&shape.singular_values_shape())?;
    let rank = TypeConversionOps::cast(client, &join(client, &ranks)?, DType::I64)?
        .reshape(&shape.rank_shape())?;

    Ok(LstsqResult {
        solution: linalg_demote(client, solution, original_dtype)?,
        rank,
        singular_values: linalg_demote(client, singular_values, original_dtype)?,
    })
}

/// Result of one `[m, n]` solve, each part with a leading batch dim of 1.
struct SingleSolve<R: Runtime> {
    /// `[1, n, k]`, absent when `k == 0`.
    solution: Option<Tensor<R>>,
    /// `[1, min(m, n)]`.
    singular_values: Tensor<R>,
    /// `[1]`, in the working dtype.
    rank: Tensor<R>,
}

/// Solve one contiguous `[m, n]` matrix against a contiguous `[m, k]` `b`.
///
/// Both have `m > 0` and `n > 0`. The SVD runs on `a / max|a|`, as `gelsd`
/// scales its input. The SVD then sees entries of order 1 whatever the scale
/// of `a`. The solution divides by the same scale, and the singular values
/// multiply by it. A zero or NaN `max|a|` scales by 1.
///
/// A NaN or infinity anywhere in `a` turns the whole solution into NaN. The
/// SVD alone does not guarantee that, so a zero-scaled sum of `a` joins the
/// result.
fn solve_one<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>, rcond: f64) -> Result<SingleSolve<R>>
where
    R: Runtime<DType = DType>,
    C: LstsqOps<R>,
{
    let (m, n, k) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    let p = m.min(n);
    let (dtype, device) = (a.dtype(), a.device());

    let a_max = ReduceOps::max(client, &UnaryOps::abs(client, a)?, &[0, 1], true)?;
    let zero = Tensor::<R>::zeros(&[], dtype, device)?;
    let one = Tensor::<R>::ones(&[], dtype, device)?;
    let positive = CompareOps::gt(client, &a_max, &zero)?;
    let scale = ConditionalOps::where_cond(client, &positive, &a_max, &one)?;
    let a_scaled = BinaryOps::div(client, a, &scale)?;

    let svd = LinearAlgebraAlgorithms::svd_decompose(client, &a_scaled)?;
    let (s_inv, rank) = svd_cutoff(client, &svd.s, rcond)?;
    let scale_row = scale.reshape(&[1, 1])?;
    let s = svd.s.contiguous()?.reshape(&[1, p])?;
    let singular_values = BinaryOps::mul(client, &s, &scale_row)?;

    let solution = if k == 0 {
        None
    } else {
        let ut = svd.u.transpose(0, 1)?.contiguous()?;
        let utb = MatmulOps::matmul(client, &ut, b)?;
        let weighted = BinaryOps::mul(client, &utb, &s_inv.reshape(&[p, 1])?)?;
        let v = svd.vt.transpose(0, 1)?.contiguous()?;
        let x = MatmulOps::matmul(client, &v, &weighted)?;
        let x = BinaryOps::div(client, &x, &scale_row)?;
        let a_zeroed = ScalarOps::mul_scalar(client, a, 0.0)?;
        let nan_probe = ReduceOps::sum(client, &a_zeroed, &[0, 1], true)?;
        let x = BinaryOps::add(client, &x, &nan_probe)?;
        Some(x.reshape(&[1, n, k])?)
    };

    Ok(SingleSolve {
        solution,
        singular_values,
        rank: rank.reshape(&[1])?,
    })
}

/// Concatenate per-matrix results along their leading batch dim of 1.
fn join<R, C>(client: &C, parts: &[Tensor<R>]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ShapeOps<R>,
{
    match parts {
        [] => Err(Error::Internal(
            "lstsq: no per-matrix results to join".to_string(),
        )),
        [one] => Ok(one.clone()),
        many => {
            let refs: Vec<&Tensor<R>> = many.iter().collect();
            ShapeOps::cat(client, &refs, 0)
        }
    }
}

/// Zero solution, zero singular values and rank 0 for an empty problem.
fn empty_result<R>(shape: &LstsqShape, dtype: DType, device: &R::Device) -> Result<LstsqResult<R>>
where
    R: Runtime<DType = DType>,
{
    Ok(LstsqResult {
        solution: Tensor::<R>::zeros(&shape.solution_shape(), dtype, device)?,
        rank: Tensor::<R>::zeros(&shape.rank_shape(), DType::I64, device)?,
        singular_values: Tensor::<R>::zeros(&shape.singular_values_shape(), dtype, device)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    type T = Tensor<CpuRuntime>;

    fn client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        (CpuClient::new(device.clone()), device)
    }

    fn t64(data: &[f64], shape: &[usize], d: &CpuDevice) -> T {
        T::from_slice(data, shape, d).unwrap()
    }

    fn t32(data: &[f64], shape: &[usize], d: &CpuDevice) -> T {
        let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        T::from_slice(&v, shape, d).unwrap()
    }

    /// Deterministic values in `[-1, 1)`.
    fn lcg(seed: u64, len: usize) -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                2.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
            })
            .collect()
    }

    /// Random `[m, n]` matrix with `3` added to its leading diagonal.
    fn well_conditioned(seed: u64, m: usize, n: usize) -> Vec<f64> {
        let mut a = lcg(seed, m * n);
        for i in 0..m.min(n) {
            a[i * n + i] += 3.0;
        }
        a
    }

    fn solve(c: &CpuClient, a: &T, b: &T, rcond: Option<f64>) -> LstsqResult<CpuRuntime> {
        lstsq_impl(c, a, b, rcond).unwrap()
    }

    fn read(t: &T) -> Vec<f64> {
        match t.dtype() {
            DType::F64 => t.to_vec::<f64>(),
            DType::F32 => t.to_vec::<f32>().into_iter().map(f64::from).collect(),
            other => panic!("read: unexpected dtype {other:?}"),
        }
    }

    fn rank_of(r: &LstsqResult<CpuRuntime>) -> Vec<i64> {
        assert_eq!(r.rank.dtype(), DType::I64);
        r.rank.to_vec::<i64>()
    }

    fn assert_close(got: &[f64], want: &[f64], rtol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= rtol * (1.0 + w.abs()),
                "{what} at {i}: {g} vs {w}"
            );
        }
    }

    fn matmul(c: &CpuClient, a: &T, b: &T) -> T {
        MatmulOps::matmul(c, a, b).unwrap()
    }

    fn transpose(a: &T) -> T {
        a.transpose(0, 1).unwrap().contiguous().unwrap()
    }

    #[test]
    fn matches_pinverse_times_b() {
        let (c, d) = client();
        for (seed, (m, n)) in [(3, 5), (5, 3), (4, 4)].into_iter().enumerate() {
            let a = well_conditioned(seed as u64 + 1, m, n);
            let b = lcg(seed as u64 + 100, m * 2);
            let makers: [(fn(&[f64], &[usize], &CpuDevice) -> T, f64); 2] =
                [(t64, 1e-10), (t32, 1e-4)];
            for (make, rtol) in makers {
                let at = make(&a, &[m, n], &d);
                let bt = make(&b, &[m, 2], &d);
                let x = solve(&c, &at, &bt, None);
                assert_eq!(x.solution.shape(), &[n, 2]);
                assert_eq!(rank_of(&x), vec![m.min(n) as i64]);
                let pinv = LinearAlgebraAlgorithms::pinverse(&c, &at, None).unwrap();
                let want = matmul(&c, &pinv, &bt);
                assert_close(&read(&x.solution), &read(&want), rtol, &format!("{m}x{n}"));
            }
        }
    }

    #[test]
    fn overdetermined_matches_normal_equations() {
        let (c, d) = client();
        let a = t64(&well_conditioned(7, 5, 3), &[5, 3], &d);
        let b = t64(&lcg(8, 10), &[5, 2], &d);
        let x = solve(&c, &a, &b, None).solution;
        let at = transpose(&a);
        let ata_inv = LinearAlgebraAlgorithms::inverse(&c, &matmul(&c, &at, &a)).unwrap();
        let want = matmul(&c, &ata_inv, &matmul(&c, &at, &b));
        assert_close(&read(&x), &read(&want), 1e-10, "normal equations");
    }

    #[test]
    fn underdetermined_is_exact_and_minimum_norm() {
        let (c, d) = client();
        let a = t64(&well_conditioned(9, 3, 5), &[3, 5], &d);
        let b = t64(&lcg(10, 3), &[3], &d);
        let x = solve(&c, &a, &b, None).solution;
        assert_eq!(x.shape(), &[5]);
        let x_col = x.reshape(&[5, 1]).unwrap();
        let ax = matmul(&c, &a, &x_col);
        assert_close(&read(&ax), &read(&b), 1e-10, "A x = b");
        let vt = LinearAlgebraAlgorithms::svd_decompose(&c, &a).unwrap().vt;
        let proj = matmul(&c, &transpose(&vt), &matmul(&c, &vt, &x_col));
        assert_close(&read(&proj), &read(&x), 1e-10, "x in row space");
    }

    #[test]
    fn exact_square_and_overdetermined_cases() {
        let (c, d) = client();
        let a = t32(&[2.0, 1.0, 1.0, 2.0], &[2, 2], &d);
        let b = t32(&[3.0, 3.0], &[2], &d);
        assert_close(
            &read(&solve(&c, &a, &b, None).solution),
            &[1.0, 1.0],
            1e-4,
            "2x2",
        );

        let a = t32(&[1.0, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], &d);
        let b = t32(&[1.0, 2.0, 3.0], &[3], &d);
        assert_close(
            &read(&solve(&c, &a, &b, None).solution),
            &[0.0, 1.0],
            1e-4,
            "line fit",
        );

        let b = t32(&[1.0, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2], &d);
        let x = solve(&c, &a, &b, None).solution;
        assert_eq!(x.shape(), &[2, 2]);
        assert_close(&read(&x), &[0.0, 0.0, 1.0, 2.0], 1e-4, "multi rhs");
    }

    #[test]
    fn rank_deficient_exact_cases() {
        let (c, d) = client();
        let cases: [(&[f64], [usize; 2], &[f64], &[f64], i64); 4] = [
            (&[1.0, 1.0, 1.0, 1.0], [2, 2], &[2.0, 2.0], &[1.0, 1.0], 1),
            (&[0.0; 6], [3, 2], &[1.0, 2.0, 3.0], &[0.0, 0.0], 0),
            (&[1.0, 0.0, 0.0, 0.0], [2, 2], &[3.0, 4.0], &[3.0, 0.0], 1),
            (
                &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                [3, 2],
                &[2.0, 4.0, 6.0],
                &[1.0, 1.0],
                1,
            ),
        ];
        for (a, shape, b, want, rank) in cases {
            let at = t64(a, &shape, &d);
            let bt = t64(b, &[shape[0]], &d);
            let x = solve(&c, &at, &bt, None);
            let got = read(&x.solution);
            assert!(got.iter().all(|v| v.is_finite()), "{a:?}: {got:?}");
            assert_close(&got, want, 1e-10, &format!("{a:?}"));
            assert_eq!(rank_of(&x), vec![rank], "{a:?}");
        }
    }

    #[test]
    fn tiny_singular_values_are_zeroed() {
        let (c, d) = client();
        let a = t64(&[1.0, 0.0, 0.0, 1e-20], &[2, 2], &d);
        let b = t64(&[1.0, 1.0], &[2], &d);
        let x = solve(&c, &a, &b, None);
        assert_close(&read(&x.solution), &[1.0, 0.0], 1e-12, "diag(1, 1e-20)");
        assert_eq!(rank_of(&x), vec![1]);
    }

    #[test]
    fn tiny_scale_matrix_solves() {
        let (c, d) = client();
        let a = well_conditioned(14, 4, 3);
        let b = lcg(15, 4);
        let reference = solve(&c, &t64(&a, &[4, 3], &d), &t64(&b, &[4], &d), None);
        let want_x = read(&reference.solution);
        let want_s = read(&reference.singular_values);

        // A = a * 1e-30 gives X = x * 1e30 and S = s * 1e-30.
        let tiny: Vec<f64> = a.iter().map(|x| x * 1e-30).collect();
        let x = solve(&c, &t64(&tiny, &[4, 3], &d), &t64(&b, &[4], &d), None);
        assert_eq!(rank_of(&x), vec![3]);
        let got_x: Vec<f64> = read(&x.solution).iter().map(|x| x * 1e-30).collect();
        assert_close(&got_x, &want_x, 1e-10, "F64 scaled 1e-30");
        let got_s: Vec<f64> = read(&x.singular_values).iter().map(|s| s * 1e30).collect();
        assert_close(&got_s, &want_s, 1e-10, "F64 singular values");

        let tiny32: Vec<f64> = a.iter().map(|x| x * 1e-10).collect();
        let x = solve(&c, &t32(&tiny32, &[4, 3], &d), &t32(&b, &[4], &d), None);
        assert_eq!(rank_of(&x), vec![3]);
        let got_x: Vec<f64> = read(&x.solution).iter().map(|x| x * 1e-10).collect();
        assert_close(&got_x, &want_x, 1e-4, "F32 scaled 1e-10");
    }

    #[test]
    fn rank_one_random_is_finite_and_matches_pinverse() {
        let (c, d) = client();
        let u = lcg(11, 5);
        let v = lcg(12, 4);
        let a: Vec<f64> = (0..20).map(|i| u[i / 4] * v[i % 4]).collect();
        let at = t64(&a, &[5, 4], &d);
        let bt = t64(&lcg(13, 10), &[5, 2], &d);
        let x = solve(&c, &at, &bt, Some(1e-10));
        assert_eq!(rank_of(&x), vec![1]);
        let got = read(&x.solution);
        assert!(got.iter().all(|v| v.is_finite()));
        let pinv = LinearAlgebraAlgorithms::pinverse(&c, &at, Some(1e-10)).unwrap();
        assert_close(&got, &read(&matmul(&c, &pinv, &bt)), 1e-8, "rank 1");
    }

    #[test]
    fn rcond_override_changes_rank() {
        let (c, d) = client();
        let a = t64(&[1.0, 0.0, 0.0, 1e-3], &[2, 2], &d);
        let b = t64(&[1.0, 1.0], &[2], &d);
        let full = solve(&c, &a, &b, None);
        assert_eq!(rank_of(&full), vec![2]);
        assert_close(
            &read(&full.solution),
            &[1.0, 1000.0],
            1e-10,
            "default rcond",
        );
        let cut = solve(&c, &a, &b, Some(1e-2));
        assert_eq!(rank_of(&cut), vec![1]);
        assert_close(&read(&cut.solution), &[1.0, 0.0], 1e-12, "rcond 1e-2");
        assert_close(
            &read(&cut.singular_values),
            &[1.0, 1e-3],
            1e-12,
            "singular values",
        );
    }

    #[test]
    fn batched_matches_per_matrix() {
        let (c, d) = client();
        let (m, n, k) = (4, 2, 3);
        let a = lcg(21, 6 * m * n);
        let b = lcg(22, 6 * m * k);
        let x = solve(
            &c,
            &t64(&a, &[2, 3, m, n], &d),
            &t64(&b, &[2, 3, m, k], &d),
            None,
        );
        assert_eq!(x.solution.shape(), &[2, 3, n, k]);
        assert_eq!(x.singular_values.shape(), &[2, 3, n]);
        assert_eq!(x.rank.shape(), &[2, 3]);
        let (got_x, got_s) = (read(&x.solution), read(&x.singular_values));
        for i in 0..6 {
            let a_i = t64(&a[i * m * n..(i + 1) * m * n], &[m, n], &d);
            let b_i = t64(&b[i * m * k..(i + 1) * m * k], &[m, k], &d);
            let one = solve(&c, &a_i, &b_i, None);
            let what = format!("matrix {i}");
            assert_close(
                &got_x[i * n * k..(i + 1) * n * k],
                &read(&one.solution),
                1e-12,
                &what,
            );
            assert_close(
                &got_s[i * n..(i + 1) * n],
                &read(&one.singular_values),
                1e-12,
                &what,
            );
        }
        assert_eq!(rank_of(&x), vec![2; 6]);
    }

    #[test]
    fn batch_of_one_matches_unbatched() {
        let (c, d) = client();
        let a = well_conditioned(31, 4, 2);
        let b = lcg(32, 4);
        let batched = solve(&c, &t64(&a, &[1, 4, 2], &d), &t64(&b, &[1, 4, 1], &d), None);
        assert_eq!(batched.solution.shape(), &[1, 2, 1]);
        assert_eq!(batched.rank.shape(), &[1]);
        let single = solve(&c, &t64(&a, &[4, 2], &d), &t64(&b, &[4, 1], &d), None);
        assert_close(
            &read(&batched.solution),
            &read(&single.solution),
            1e-12,
            "batch 1",
        );
    }

    #[test]
    fn vector_and_column_rhs_agree() {
        let (c, d) = client();
        let a = t64(&well_conditioned(41, 5, 3), &[5, 3], &d);
        let b = lcg(42, 5);
        let vec_x = solve(&c, &a, &t64(&b, &[5], &d), None).solution;
        let col_x = solve(&c, &a, &t64(&b, &[5, 1], &d), None).solution;
        assert_eq!(vec_x.shape(), &[3]);
        assert_eq!(col_x.shape(), &[3, 1]);
        assert_close(&read(&vec_x), &read(&col_x), 1e-12, "vector vs column");
    }

    #[test]
    fn empty_inputs_give_zeros() {
        let (c, d) = client();
        let x = solve(&c, &t64(&[], &[0, 3], &d), &t64(&[], &[0], &d), None);
        assert_eq!(x.solution.shape(), &[3]);
        assert_eq!(read(&x.solution), vec![0.0; 3]);
        assert_eq!(x.singular_values.shape(), &[0]);
        assert_eq!(rank_of(&x), vec![0]);

        let x = solve(
            &c,
            &t64(&[], &[3, 0], &d),
            &t64(&lcg(51, 6), &[3, 2], &d),
            None,
        );
        assert_eq!(x.solution.shape(), &[0, 2]);

        let x = solve(
            &c,
            &t64(&[], &[0, 4, 3], &d),
            &t64(&[], &[0, 4, 1], &d),
            None,
        );
        assert_eq!(x.solution.shape(), &[0, 3, 1]);
        assert_eq!(x.rank.shape(), &[0]);

        let a = t64(&well_conditioned(52, 3, 2), &[3, 2], &d);
        let x = solve(&c, &a, &t64(&[], &[3, 0], &d), None);
        assert_eq!(x.solution.shape(), &[2, 0]);
        assert_eq!(rank_of(&x), vec![2]);
    }

    #[test]
    fn invalid_inputs_rejected() {
        let (c, d) = client();
        let a64 = t64(&[1.0; 6], &[3, 2], &d);
        let b32 = t32(&[1.0; 3], &[3], &d);
        assert!(matches!(
            lstsq_impl(&c, &a64, &b32, None),
            Err(Error::DTypeMismatch { .. })
        ));
        let ai = T::from_slice(&[1i32; 4], &[2, 2], &d).unwrap();
        let bi = T::from_slice(&[1i32; 2], &[2], &d).unwrap();
        assert!(matches!(
            lstsq_impl(&c, &ai, &bi, None),
            Err(Error::UnsupportedDType { .. })
        ));
        let b_rows = t64(&[1.0; 4], &[4], &d);
        assert!(matches!(
            lstsq_impl(&c, &a64, &b_rows, None),
            Err(Error::ShapeMismatch { .. })
        ));
        let a_batch = t64(&[1.0; 12], &[2, 3, 2], &d);
        let b_batch = t64(&[1.0; 9], &[3, 3, 1], &d);
        assert!(matches!(
            lstsq_impl(&c, &a_batch, &b_batch, None),
            Err(Error::ShapeMismatch { .. })
        ));
        let b_vec = t64(&[1.0; 3], &[3], &d);
        assert!(matches!(
            lstsq_impl(&c, &a64, &b_vec, Some(-1.0)),
            Err(Error::InvalidArgument { arg: "rcond", .. })
        ));
    }

    #[test]
    fn nan_input_propagates() {
        let (c, d) = client();
        let a = t64(&[1.0, f64::NAN, 0.0, 1.0], &[2, 2], &d);
        let b = t64(&[1.0, 1.0], &[2], &d);
        let x = read(&solve(&c, &a, &b, None).solution);
        assert!(x.iter().all(|v| v.is_nan()), "{x:?}");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn half_inputs_compute_in_f32() {
        let (c, d) = client();
        let a = t32(&well_conditioned(61, 4, 3), &[4, 3], &d);
        let b = t32(&lcg(62, 8), &[4, 2], &d);
        for (half, rtol) in [(DType::F16, 1e-2), (DType::BF16, 3e-2)] {
            let a_h = TypeConversionOps::cast(&c, &a, half).unwrap();
            let b_h = TypeConversionOps::cast(&c, &b, half).unwrap();
            let x = solve(&c, &a_h, &b_h, None);
            assert_eq!(x.solution.dtype(), half);
            assert_eq!(x.singular_values.dtype(), half);
            assert_eq!(rank_of(&x), vec![3]);
            let a_f = TypeConversionOps::cast(&c, &a_h, DType::F32).unwrap();
            let b_f = TypeConversionOps::cast(&c, &b_h, DType::F32).unwrap();
            let want = solve(&c, &a_f, &b_f, None).solution;
            let got = TypeConversionOps::cast(&c, &x.solution, DType::F32).unwrap();
            assert_close(&read(&got), &read(&want), rtol, &format!("{half:?}"));
        }
    }
}
