//! Batched triangular-solve launchers: one launch for every matrix and column.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_names,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Shape of one batched triangular launch.
#[derive(Debug, Clone, Copy)]
pub struct TriangularBatch {
    /// Number of matrices.
    pub batch: usize,
    /// Order of each matrix.
    pub n: usize,
    /// Right-hand-side columns per matrix.
    pub nrhs: usize,
}

/// Which substitution a launch runs.
#[derive(Debug, Clone, Copy)]
pub enum TriangularSubstitution {
    /// Lower triangular, forward. With `unit_diagonal`, the diagonal is 1.
    Forward {
        /// Treat the diagonal as ones and never read it.
        unit_diagonal: bool,
    },
    /// Upper triangular, backward.
    Backward,
}

fn to_u32(value: usize, what: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidArgument {
        arg: what,
        reason: format!("{value} exceeds the 32-bit range of the triangular-solve kernel"),
    })
}

/// Launch a batched substitution over `[batch, n, n]` against `[batch, n, nrhs]`.
///
/// One thread solves one column of one matrix, so the whole batch runs in
/// a single launch.
///
/// # Safety
///
/// - `a_ptr` must point to `batch * n * n` contiguous elements of `dtype`.
/// - `b_ptr` must point to `batch * n * nrhs` contiguous elements of `dtype`.
/// - `x_ptr` must have space for `batch * n * nrhs` elements and must not
///   overlap `a_ptr` or `b_ptr`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_triangular_solve_batched(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    shape: TriangularBatch,
    substitution: TriangularSubstitution,
) -> Result<()> {
    let threads = shape
        .batch
        .checked_mul(shape.nrhs)
        .ok_or_else(|| Error::InvalidArgument {
            arg: "batch",
            reason: format!(
                "{} matrices times {} columns overflows usize",
                shape.batch, shape.nrhs
            ),
        })?;
    if threads == 0 || shape.n == 0 {
        return Ok(());
    }
    let n = to_u32(shape.n, "n")?;
    let nrhs = to_u32(shape.nrhs, "nrhs")?;
    let batch = to_u32(shape.batch, "batch")?;

    let op = match substitution {
        TriangularSubstitution::Forward { .. } => "forward_sub_batched",
        TriangularSubstitution::Backward => "backward_sub_batched",
    };
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };
    let func_name = format!("{op}_{suffix}");

    let module = get_or_load_module(context, device_index, kernel_names::LINALG_SOLVERS_MODULE)?;
    let func = get_kernel_function(&module, &func_name)?;
    let cfg = launch_config(elementwise_launch_config(threads)?, (BLOCK_SIZE, 1, 1), 0);
    let unit_diag_i32: Option<i32> = match substitution {
        TriangularSubstitution::Forward { unit_diagonal } => Some(i32::from(unit_diagonal)),
        TriangularSubstitution::Backward => None,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&a_ptr);
    builder.arg(&b_ptr);
    builder.arg(&x_ptr);
    builder.arg(&n);
    builder.arg(&nrhs);
    builder.arg(&batch);
    if let Some(unit_diag) = unit_diag_i32.as_ref() {
        builder.arg(unit_diag);
    }

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA {func_name} kernel launch failed: {:?}", e)))?;

    Ok(())
}
