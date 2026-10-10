//! scatter_reduce for WebGPU.
//!
//! The launcher sorts the source positions by destination and folds each
//! destination's run in increasing source position, the order the CPU
//! reference uses. See runtime/wgpu/shaders/index/scatter_reduce.rs.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScatterReduceOp;
use crate::ops::common::validate_scatter_extents;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{alloc_output, get_tensor_buffer};
use crate::runtime::wgpu::shaders::{
    ScatterIndexType, ScatterReduceBuffers, launch_scatter_reduce,
};
use crate::tensor::Tensor;

pub(super) fn scatter_reduce(
    client: &WgpuClient,
    dst: &Tensor<WgpuRuntime>,
    dim: usize,
    index: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
    op: ScatterReduceOp,
    include_self: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = dst.dtype();

    // WebGPU covers the three dtypes WGSL has.
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "scatter_reduce",
        });
    }

    let shape = dst.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_type = match index.dtype() {
        DType::I32 => ScatterIndexType::I32,
        DType::I64 => ScatterIndexType::I64,
        _ => {
            return Err(Error::InvalidArgument {
                arg: "index",
                reason: format!(
                    "scatter_reduce index must be I32 or I64, got {:?}",
                    index.dtype()
                ),
            });
        }
    };

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Element-wise semantics (matching CPU/CUDA/PyTorch): `index[e]` gives the
    // destination coordinate along `dim` for source element `e`.
    if index.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index.shape().to_vec(),
        });
    }
    if index.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index.shape().to_vec(),
        });
    }

    // Off the scatter axis, every source coordinate must address the
    // destination.
    validate_scatter_extents(shape, src.shape(), dim)?;

    let out = alloc_output(client, shape, dtype)?;
    // An empty destination has no slot any index could address, and no buffer
    // to bind.
    if dst.numel() == 0 {
        return Ok(out);
    }

    let dst = ensure_contiguous(dst)?;
    let dst_buf = get_tensor_buffer(&dst)?;
    let out_buf = get_tensor_buffer(&out)?;

    // An empty source is the zero-byte allocation, with no buffer to bind.
    // The contiguous tensors stay alive until the launch has submitted.
    let src_contig = ensure_contiguous(src)?;
    let index_contig = ensure_contiguous(index)?;
    let (src_buf, index_buf) = if src.numel() == 0 {
        (None, None)
    } else {
        (
            Some(get_tensor_buffer(&src_contig)?),
            Some(get_tensor_buffer(&index_contig)?),
        )
    };

    launch_scatter_reduce(
        client.pipeline_cache(),
        client.wgpu_queue(),
        ScatterReduceBuffers {
            src: src_buf.as_deref(),
            indices: index_buf.as_deref(),
            dst: &dst_buf,
            out: &out_buf,
        },
        index_type,
        dtype,
        src.shape(),
        shape,
        dim,
        op,
        include_self,
    )?;

    Ok(out)
}
