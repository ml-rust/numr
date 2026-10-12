//! Triangular solves (Lx = b, Ux = b) for CUDA, one matrix or a batch.
//!
//! Every shape runs as one kernel launch: one thread per (matrix, column).

use super::super::super::CudaRuntime;
use super::super::super::client::CudaClient;
use super::super::super::kernels::{
    TriangularBatch, TriangularSubstitution, launch_triangular_solve_batched,
};
use crate::algorithm::linalg::{validate_linalg_dtype, validate_triangular_solve};
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, RuntimeClient, ensure_contiguous};
use crate::tensor::Tensor;

/// Solve lower triangular systems Lx = b.
///
/// `l: [n, n]` with `b: [n]` or `[n, m]`, or `l: [..batch, n, n]` with
/// `b: [..batch, n, m]`. The result has the shape of `b`.
pub fn solve_triangular_lower_impl(
    client: &CudaClient,
    l: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    unit_diagonal: bool,
) -> Result<Tensor<CudaRuntime>> {
    solve_triangular_impl(
        client,
        l,
        b,
        TriangularSubstitution::Forward { unit_diagonal },
        "solve_triangular_lower",
    )
}

/// Solve upper triangular systems Ux = b.
///
/// Takes the same shapes as [`solve_triangular_lower_impl`].
pub fn solve_triangular_upper_impl(
    client: &CudaClient,
    u: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    solve_triangular_impl(
        client,
        u,
        b,
        TriangularSubstitution::Backward,
        "solve_triangular_upper",
    )
}

fn solve_triangular_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    substitution: TriangularSubstitution,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let shape = validate_triangular_solve(op, a.shape(), b.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if shape.out_numel() == 0 {
        return Tensor::<CudaRuntime>::zeros(&shape.out_shape, dtype, device);
    }

    let a = ensure_contiguous(a)?;
    let b = ensure_contiguous(b)?;

    let x_guard = AllocGuard::new(
        client.allocator(),
        shape.out_numel() * dtype.size_in_bytes(),
    )?;

    // The kernel runs on the client stream; frees of `a`, `b` and the
    // output are stream-ordered behind it, so no host synchronization is
    // needed here.
    unsafe {
        launch_triangular_solve_batched(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            b.ptr(),
            x_guard.ptr(),
            TriangularBatch {
                batch: shape.batch,
                n: shape.n,
                nrhs: shape.nrhs,
            },
            substitution,
        )?;
    }

    let x_ptr = x_guard.release();
    Ok(unsafe { CudaClient::tensor_from_raw(x_ptr, &shape.out_shape, dtype, device) })
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::create_client;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;
    use crate::algorithm::linalg::triangular_test_data::{
        CASES, gdn_like_unit_lower, host_solve, identity, max_abs_diff, to_f32, to_f64,
        triangular_batch, values,
    };
    use crate::dtype::DType;
    use crate::ops::TypeConversionOps;

    fn solve(
        client: &CudaClient,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        lower: bool,
        unit: bool,
    ) -> Tensor<CudaRuntime> {
        if lower {
            client.solve_triangular_lower(a, b, unit).unwrap()
        } else {
            client.solve_triangular_upper(a, b).unwrap()
        }
    }

    fn check_all_cases(lower: bool, unit: bool) {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();
        for (ci, case) in CASES.iter().enumerate() {
            let seed = 0x7a1e_0000 + ci as u64;
            let (n, m) = (case.n, case.m());
            let a32 = to_f32(&triangular_batch(seed, case.batch(), n, lower, unit));
            let b32 = to_f32(&values(seed ^ 0xb, case.batch() * n * m));
            let at = Tensor::<CudaRuntime>::from_slice(&a32, &case.a_shape(), device).unwrap();
            let bt = Tensor::<CudaRuntime>::from_slice(&b32, &case.b_shape(), device).unwrap();
            let batched = solve(&client, &at, &bt, lower, unit);
            assert_eq!(batched.shape(), case.b_shape().as_slice());
            let batched = batched.to_vec::<f32>();

            let b_one: Vec<usize> = case.b_shape()[case.batch_dims.len()..].to_vec();
            let mut separate = Vec::with_capacity(batched.len());
            for bi in 0..case.batch() {
                let a_i = &a32[bi * n * n..(bi + 1) * n * n];
                let b_i = &b32[bi * n * m..(bi + 1) * n * m];
                let a_i = Tensor::<CudaRuntime>::from_slice(a_i, &[n, n], device).unwrap();
                let b_i = Tensor::<CudaRuntime>::from_slice(b_i, &b_one, device).unwrap();
                separate.extend(solve(&client, &a_i, &b_i, lower, unit).to_vec::<f32>());
            }
            // One thread runs one column with fixed arithmetic: bit-identical.
            assert_eq!(batched, separate, "case {ci} lower={lower} unit={unit}");

            let want = host_solve(
                &to_f64(&a32),
                &to_f64(&b32),
                case.batch(),
                n,
                m,
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
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();
        let case = &CASES[4];
        for lower in [true, false] {
            let a = triangular_batch(11, case.batch(), case.n, lower, false);
            let b = values(12, case.batch() * case.n * case.m());
            let at = Tensor::<CudaRuntime>::from_slice(&a, &case.a_shape(), device).unwrap();
            let bt = Tensor::<CudaRuntime>::from_slice(&b, &case.b_shape(), device).unwrap();
            let x = solve(&client, &at, &bt, lower, false);
            let want = host_solve(&a, &b, case.batch(), case.n, case.m(), lower, false);
            let diff = max_abs_diff(&x.to_vec::<f64>(), &want);
            assert!(diff < 1e-12, "lower={lower}: {diff}");
        }
    }

    #[test]
    fn batched_accepts_strided_input() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();
        let (batch, n) = (3, 4);
        let a = to_f32(&triangular_batch(21, batch, n, false, false));
        let b = to_f32(&values(22, batch * n * 2));
        let u = Tensor::<CudaRuntime>::from_slice(&a, &[batch, n, n], device).unwrap();
        let b = Tensor::<CudaRuntime>::from_slice(&b, &[batch, n, 2], device).unwrap();
        let l_view = u.transpose(1, 2).unwrap();
        let l_copy = l_view.contiguous().unwrap();
        let from_view = client.solve_triangular_lower(&l_view, &b, false).unwrap();
        let from_copy = client.solve_triangular_lower(&l_copy, &b, false).unwrap();
        assert_eq!(from_view.to_vec::<f32>(), from_copy.to_vec::<f32>());
    }

    #[test]
    fn batched_shape_mismatch_errors() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();
        let l = Tensor::<CudaRuntime>::from_slice(&[1.0f32; 32], &[2, 4, 4], device).unwrap();
        for b_shape in [&[3, 4, 2][..], &[2, 5, 2], &[2, 4], &[1, 2, 4, 2]] {
            let numel: usize = b_shape.iter().product();
            let b =
                Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; numel], b_shape, device).unwrap();
            let lower = client.solve_triangular_lower(&l, &b, false).unwrap_err();
            let upper = client.solve_triangular_upper(&l, &b).unwrap_err();
            for err in [lower, upper] {
                let text = err.to_string();
                assert!(text.contains(&format!("{b_shape:?}")), "{text}");
                assert!(text.contains("[2, 4, 4]"), "{text}");
            }
        }
    }

    /// F32 inverse of `I + A`, strictly lower `A` near 0.8, against F64.
    ///
    /// Same data and bound as the CPU test: substitution is backward stable
    /// and the recurrence contracts, so 1e-5 leaves wide margin.
    #[test]
    fn f32_gdn_like_inverse_matches_f64() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();
        let n = 64;
        let l32 = to_f32(&gdn_like_unit_lower(0x0d17, n));
        let eye32 = to_f32(&identity(n));
        let l = Tensor::<CudaRuntime>::from_slice(&l32, &[1, n, n], device).unwrap();
        let eye = Tensor::<CudaRuntime>::from_slice(&eye32, &[1, n, n], device).unwrap();
        let t32 = client.solve_triangular_lower(&l, &eye, true).unwrap();

        let l = client.cast(&l, DType::F64).unwrap();
        let eye = client.cast(&eye, DType::F64).unwrap();
        let t64 = client.solve_triangular_lower(&l, &eye, true).unwrap();

        let diff = max_abs_diff(&to_f64(&t32.to_vec::<f32>()), &t64.to_vec::<f64>());
        assert!(diff < 1e-5, "max abs diff {diff}");
    }
}
