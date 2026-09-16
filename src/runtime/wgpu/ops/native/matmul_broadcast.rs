//! Batch-dimension normalization for WebGPU batched matmul kernels.
//!
//! The batched kernels take a single batch count and read both operands with the
//! same batch stride, so they only handle operands that already share the output's
//! batch shape. Broadcast batch dims, a mismatched batch rank, or more than one
//! batch dim all need the operands expanded to the output's batch shape and
//! flattened to 3D first.
//!
//! The expansion stays on device: a zero-stride broadcast view followed by a
//! device-side copy.

use super::super::super::WgpuRuntime;
use crate::error::Result;
use crate::ops::matmul::matmul_mkn;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Rank-1 operands as the matrices `matmul_output_shape` treats them as: a
/// `[k]` left operand becomes `[1, k]`, a `[k]` right operand `[k, 1]`.
///
/// The WebGPU kernels take matrices only, and their output shape is the same
/// either way, so the promoted operands run through the ordinary paths.
/// Returns `None` when neither operand is rank-1.
pub(crate) fn promote_rank1_operands(
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Option<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)>> {
    if a.shape().len() != 1 && b.shape().len() != 1 {
        return Ok(None);
    }
    let a2 = if a.shape().len() == 1 {
        ensure_contiguous(a)?.reshape(&[1, a.shape()[0]])?
    } else {
        a.clone()
    };
    let b2 = if b.shape().len() == 1 {
        ensure_contiguous(b)?.reshape(&[b.shape()[0], 1])?
    } else {
        b.clone()
    };
    Ok(Some((a2, b2)))
}

/// Operands flattened to `[batch, m, k]` and `[batch, k, n]`, when the batched
/// kernels cannot take them as they are. Operands are rank 2 or more; a rank-1
/// operand goes through [`promote_rank1_operands`] first.
///
/// Returns `None` when both operands already match the output's batch shape, so
/// shapes the kernels already handle keep their existing cost.
pub(crate) fn flatten_batched_operands(
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    out_shape: &[usize],
) -> Result<Option<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)>> {
    if out_shape.len() <= 2 {
        return Ok(None);
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    let out_batch = &out_shape[..out_shape.len() - 2];
    let (m, k, n) = matmul_mkn(a_shape, b_shape);

    let a_target: Vec<usize> = out_batch.iter().copied().chain([m, k]).collect();
    let b_target: Vec<usize> = out_batch.iter().copied().chain([k, n]).collect();

    // A single batch dim shared by both operands is exactly what the kernels take.
    if out_shape.len() == 3 && a_shape == a_target.as_slice() && b_shape == b_target.as_slice() {
        return Ok(None);
    }

    let batch: usize = out_batch.iter().product();
    let a3 = a
        .broadcast_to(&a_target)?
        .contiguous()?
        .reshape(&[batch, m, k])?;
    let b3 = b
        .broadcast_to(&b_target)?
        .contiguous()?
        .reshape(&[batch, k, n])?;

    Ok(Some((a3, b3)))
}
