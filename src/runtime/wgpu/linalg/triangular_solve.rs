//! Triangular system solvers for WebGPU backend, one matrix or a batch.
//!
//! Every shape runs as one dispatch: one invocation per (matrix, column).
//!
//! # Supported Data Types
//!
//! **F32 only.** WGSL shaders do not natively support F64 operations.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{validate_linalg_dtype, validate_triangular_solve};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, RuntimeClient, ensure_contiguous};
use crate::tensor::Tensor;

/// Solve lower triangular systems Lx = b.
///
/// `l: [n, n]` with `b: [n]` or `[n, m]`, or `l: [..batch, n, n]` with
/// `b: [..batch, n, m]`. The result has the shape of `b`.
pub fn solve_triangular_lower(
    client: &WgpuClient,
    l: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    unit_diagonal: bool,
) -> Result<Tensor<WgpuRuntime>> {
    solve_triangular(
        client,
        l,
        b,
        Some(unit_diagonal),
        "solve_triangular_lower",
        "WGPU solve_triangular_lower (only F32 supported)",
    )
}

/// Solve upper triangular systems Ux = b.
///
/// Takes the same shapes as [`solve_triangular_lower`].
pub fn solve_triangular_upper(
    client: &WgpuClient,
    u: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    solve_triangular(
        client,
        u,
        b,
        None,
        "solve_triangular_upper",
        "WGPU solve_triangular_upper (only F32 supported)",
    )
}

/// `lower_unit` is `Some(unit_diagonal)` for a lower solve, `None` for upper.
fn solve_triangular(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    lower_unit: Option<bool>,
    op: &'static str,
    dtype_op: &'static str,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let dtype = a.dtype();
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: dtype_op,
        });
    }
    let shape = validate_triangular_solve(op, a.shape(), b.shape())?;
    let device = client.device();

    if shape.out_numel() == 0 {
        return Tensor::<WgpuRuntime>::zeros(&shape.out_shape, dtype, device);
    }

    let to_u32 = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| Error::BackendLimitation {
            backend: "wgpu",
            operation: "triangular_solve_batched",
            reason: format!("{what} {value} exceeds the shader's 32-bit indexing"),
        })
    };
    // The shader indexes a and b with u32.
    to_u32(shape.batch * shape.n * shape.n, "matrix elements")?;
    to_u32(shape.out_numel(), "right-hand-side elements")?;
    let threads = shape.batch * shape.nrhs;
    let params: [u32; 4] = [
        to_u32(shape.n, "n")?,
        to_u32(shape.nrhs, "nrhs")?,
        to_u32(shape.batch, "batch")?,
        u32::from(lower_unit.unwrap_or(false)),
    ];

    let a = ensure_contiguous(a)?;
    let b = ensure_contiguous(b)?;
    let a_buffer =
        get_buffer(a.ptr()).ok_or_else(|| Error::Internal("Failed to get A buffer".to_string()))?;
    let b_buffer =
        get_buffer(b.ptr()).ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    let x_guard = AllocGuard::new(
        client.allocator(),
        shape.out_numel() * dtype.size_in_bytes(),
    )?;
    let x_buffer = get_buffer(x_guard.ptr())
        .ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

    let params_buffer = client.create_uniform_buffer("triangular_batched_params", 16);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_triangular_solve_batched(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &b_buffer,
        &x_buffer,
        &params_buffer,
        threads,
        lower_unit.is_some(),
        dtype,
    )?;

    client.synchronize();

    Ok(unsafe { WgpuClient::tensor_from_raw(x_guard.release(), &shape.out_shape, dtype, device) })
}

#[cfg(test)]
mod tests {
    use super::super::test_support::create_client;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;
    use crate::algorithm::linalg::triangular_test_data::{
        CASES, gdn_like_unit_lower, host_solve, identity, max_abs_diff, to_f32, to_f64,
        triangular_batch, values,
    };
    use crate::runtime::wgpu::is_wgpu_available;

    fn solve(
        client: &WgpuClient,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        lower: bool,
        unit: bool,
    ) -> Tensor<WgpuRuntime> {
        if lower {
            client.solve_triangular_lower(a, b, unit).unwrap()
        } else {
            client.solve_triangular_upper(a, b).unwrap()
        }
    }

    fn check_all_cases(lower: bool, unit: bool) {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }
        let client = create_client();
        let device = client.device();
        for (ci, case) in CASES.iter().enumerate() {
            let seed = 0x7a1e_0000 + ci as u64;
            let (n, m) = (case.n, case.m());
            let a32 = to_f32(&triangular_batch(seed, case.batch(), n, lower, unit));
            let b32 = to_f32(&values(seed ^ 0xb, case.batch() * n * m));
            let at = Tensor::<WgpuRuntime>::from_slice(&a32, &case.a_shape(), device).unwrap();
            let bt = Tensor::<WgpuRuntime>::from_slice(&b32, &case.b_shape(), device).unwrap();
            let batched = solve(&client, &at, &bt, lower, unit);
            assert_eq!(batched.shape(), case.b_shape().as_slice());
            let batched = batched.to_vec::<f32>();

            let b_one: Vec<usize> = case.b_shape()[case.batch_dims.len()..].to_vec();
            let mut separate = Vec::with_capacity(batched.len());
            for bi in 0..case.batch() {
                let a_i = &a32[bi * n * n..(bi + 1) * n * n];
                let b_i = &b32[bi * n * m..(bi + 1) * n * m];
                let a_i = Tensor::<WgpuRuntime>::from_slice(a_i, &[n, n], device).unwrap();
                let b_i = Tensor::<WgpuRuntime>::from_slice(b_i, &b_one, device).unwrap();
                separate.extend(solve(&client, &a_i, &b_i, lower, unit).to_vec::<f32>());
            }
            // One invocation runs one column with fixed arithmetic: bit-identical.
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
    fn batched_accepts_strided_input() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }
        let client = create_client();
        let device = client.device();
        let (batch, n) = (3, 4);
        let a = to_f32(&triangular_batch(21, batch, n, false, false));
        let b = to_f32(&values(22, batch * n * 2));
        let u = Tensor::<WgpuRuntime>::from_slice(&a, &[batch, n, n], device).unwrap();
        let b = Tensor::<WgpuRuntime>::from_slice(&b, &[batch, n, 2], device).unwrap();
        let l_view = u.transpose(1, 2).unwrap();
        let l_copy = l_view.contiguous().unwrap();
        let from_view = client.solve_triangular_lower(&l_view, &b, false).unwrap();
        let from_copy = client.solve_triangular_lower(&l_copy, &b, false).unwrap();
        assert_eq!(from_view.to_vec::<f32>(), from_copy.to_vec::<f32>());
    }

    #[test]
    fn batched_shape_mismatch_errors() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }
        let client = create_client();
        let device = client.device();
        let l = Tensor::<WgpuRuntime>::from_slice(&[1.0f32; 32], &[2, 4, 4], device).unwrap();
        for b_shape in [&[3, 4, 2][..], &[2, 5, 2], &[2, 4], &[1, 2, 4, 2]] {
            let numel: usize = b_shape.iter().product();
            let b =
                Tensor::<WgpuRuntime>::from_slice(&vec![1.0f32; numel], b_shape, device).unwrap();
            let lower = client.solve_triangular_lower(&l, &b, false).unwrap_err();
            let upper = client.solve_triangular_upper(&l, &b).unwrap_err();
            for err in [lower, upper] {
                let text = err.to_string();
                assert!(text.contains(&format!("{b_shape:?}")), "{text}");
                assert!(text.contains("[2, 4, 4]"), "{text}");
            }
        }
    }

    /// F32 inverse of `I + A`, strictly lower `A` near 0.8, against an f64
    /// host solve of the same f32 data. WebGPU has no F64, so the host
    /// computes the reference. Same bound as the CPU test.
    #[test]
    fn f32_gdn_like_inverse_matches_f64() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }
        let client = create_client();
        let device = client.device();
        let n = 64;
        let l32 = to_f32(&gdn_like_unit_lower(0x0d17, n));
        let eye32 = to_f32(&identity(n));
        let l = Tensor::<WgpuRuntime>::from_slice(&l32, &[1, n, n], device).unwrap();
        let eye = Tensor::<WgpuRuntime>::from_slice(&eye32, &[1, n, n], device).unwrap();
        let t32 = client.solve_triangular_lower(&l, &eye, true).unwrap();

        let want = host_solve(&to_f64(&l32), &identity(n), 1, n, n, true, true);
        let diff = max_abs_diff(&to_f64(&t32.to_vec::<f32>()), &want);
        assert!(diff < 1e-5, "max abs diff {diff}");
    }
}
