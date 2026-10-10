//! Scatter-with-reduction launcher.
//!
//! [`launch_scatter_reduce`] groups the source elements by destination with a
//! stable radix sort (see `scatter_reduce_sort.rs`), then reduces each
//! destination's contributions in increasing source position, the order the
//! CPU reference uses. No step accumulates through an atomic on the element
//! type, so the result is bit-identical run to run and, for f32, to CPU.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{PipelineCache, WORKGROUP_SIZE};
use super::scatter_reduce_sort::{
    ScatterIndexType, grid, record, record_sorted_keys, storage_buffer, uniform_buffer,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScatterReduceOp;

const REDUCE_SHADER_F32: &str = concat!(
    include_str!("../scatter_reduce_common.wgsl"),
    include_str!("../scatter_reduce_f32.wgsl"),
);
// The integer reductions build on the shared saturating and 64-bit helpers.
// WGSL has no include and no forward declarations, so the order is
// load-bearing.
const REDUCE_SHADER_I32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../int_matmul_acc.wgsl"),
    include_str!("../int_wide_div.wgsl"),
    include_str!("../scatter_reduce_common.wgsl"),
    include_str!("../scatter_reduce_i32.wgsl"),
);
const REDUCE_SHADER_U32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../int_matmul_acc.wgsl"),
    include_str!("../int_wide_div.wgsl"),
    include_str!("../scatter_reduce_common.wgsl"),
    include_str!("../scatter_reduce_u32.wgsl"),
);

/// Matches SrReduceParams in scatter_reduce_common.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    n: u32,
    dst_numel: u32,
    include_self: u32,
    _pad: u32,
}

/// Device buffers of one scatter_reduce call.
pub struct ScatterReduceBuffers<'a> {
    /// Source values, `src_shape` elements. `None` for an empty source.
    pub src: Option<&'a Buffer>,
    /// Indices shaped like the source. `None` for an empty source.
    pub indices: Option<&'a Buffer>,
    /// The destination, read for `include_self`.
    pub dst: &'a Buffer,
    /// The result, `dst_shape` elements, every one written.
    pub out: &'a Buffer,
}

/// Returns (shader, module key, entry point) of the reduce kernel.
fn reduce_kernel(
    dtype: DType,
    op: ScatterReduceOp,
) -> Result<(&'static str, &'static str, &'static str)> {
    use ScatterReduceOp::{Max, Mean, Min, Prod, Sum};
    let (shader, module_key) = match dtype {
        DType::F32 => (REDUCE_SHADER_F32, "scatter_reduce_f32"),
        DType::I32 => (REDUCE_SHADER_I32, "scatter_reduce_i32"),
        DType::U32 => (REDUCE_SHADER_U32, "scatter_reduce_u32"),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "scatter_reduce",
            });
        }
    };
    let entry = match (dtype, op) {
        (DType::F32, Sum) => "scatter_reduce_sum_f32",
        (DType::F32, Prod) => "scatter_reduce_prod_f32",
        (DType::F32, Max) => "scatter_reduce_max_f32",
        (DType::F32, Min) => "scatter_reduce_min_f32",
        (DType::F32, Mean) => "scatter_reduce_mean_f32",
        (DType::I32, Sum) => "scatter_reduce_sum_i32",
        (DType::I32, Prod) => "scatter_reduce_prod_i32",
        (DType::I32, Max) => "scatter_reduce_max_i32",
        (DType::I32, Min) => "scatter_reduce_min_i32",
        (DType::I32, Mean) => "scatter_reduce_mean_i32",
        (_, Sum) => "scatter_reduce_sum_u32",
        (_, Prod) => "scatter_reduce_prod_u32",
        (_, Max) => "scatter_reduce_max_u32",
        (_, Min) => "scatter_reduce_min_u32",
        (_, Mean) => "scatter_reduce_mean_u32",
    };
    Ok((shader, module_key, entry))
}

/// Launch scatter_reduce into `buffers.out`.
///
/// Source element `e` lands at its own coordinates with the coordinate on
/// `dim` replaced by `indices[e]`. An index outside `[0, dst_shape[dim])` is
/// skipped. `src_shape` may be shorter than `dst_shape` on any axis but `dim`;
/// the caller checks it is never longer. Every output element is written: the
/// reduction of its contributions, seeded from the destination when
/// `include_self` is set and from the reduction's identity otherwise.
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for any dtype but F32, I32 and U32,
/// [`Error::InvalidArgument`] when a non-empty source comes without its
/// buffers, and the limits of the key sort.
#[allow(clippy::too_many_arguments)]
pub fn launch_scatter_reduce(
    cache: &PipelineCache,
    queue: &Queue,
    buffers: ScatterReduceBuffers<'_>,
    index_type: ScatterIndexType,
    dtype: DType,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
    op: ScatterReduceOp,
    include_self: bool,
) -> Result<()> {
    let (shader, module_key, entry) = reduce_kernel(dtype, op)?;
    let n: usize = src_shape.iter().product();
    let dst_numel: usize = dst_shape.iter().product();
    if dst_numel == 0 {
        return Ok(());
    }
    if dst_numel >= u32::MAX as usize {
        return Err(Error::backend_limitation(
            "WebGPU",
            "scatter_reduce",
            format!(
                "destination has {dst_numel} elements; the shaders address fewer than {}",
                u32::MAX
            ),
        ));
    }

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_reduce"),
        });

    // An empty source reduces nothing: every run is empty and the output is
    // the seed. The reduce kernel still binds src, keys and vals, so one
    // placeholder word stands in for all three.
    let placeholder;
    let sorted;
    let (src, keys, vals) = match (n, buffers.src, buffers.indices) {
        (0, _, _) => {
            placeholder = storage_buffer(cache, "scatter_reduce_empty", 1);
            (&placeholder, &placeholder, &placeholder)
        }
        (_, Some(src), Some(indices)) => {
            sorted = record_sorted_keys(
                cache,
                queue,
                &mut encoder,
                indices,
                index_type,
                src_shape,
                dst_shape,
                dim,
            )?;
            (src, &sorted.keys, &sorted.vals)
        }
        _ => {
            return Err(Error::InvalidArgument {
                arg: "src",
                reason: format!(
                    "a source of shape {src_shape:?} needs its value and index buffers"
                ),
            });
        }
    };

    let params = ReduceParams {
        n: n as u32,
        dst_numel: dst_numel as u32,
        include_self: u32::from(include_self),
        _pad: 0,
    };
    let params_buf = uniform_buffer(cache, queue, "scatter_reduce_params", &params);
    record(
        cache,
        &mut encoder,
        module_key,
        shader,
        entry,
        &[src, keys, vals, buffers.dst, buffers.out, &params_buf],
        4,
        grid(cache, dst_numel.div_ceil(WORKGROUP_SIZE as usize) as u32),
    );

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses and validates `source` with naga, as pipeline creation does.
    fn validate_wgsl(name: &str, source: &str) {
        use wgpu::naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }

    #[test]
    fn the_reduce_shaders_validate() {
        validate_wgsl("scatter_reduce_f32", REDUCE_SHADER_F32);
        validate_wgsl("scatter_reduce_i32", REDUCE_SHADER_I32);
        validate_wgsl("scatter_reduce_u32", REDUCE_SHADER_U32);
    }

    #[test]
    fn every_dtype_and_op_names_a_kernel() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            for op in [
                ScatterReduceOp::Sum,
                ScatterReduceOp::Prod,
                ScatterReduceOp::Max,
                ScatterReduceOp::Min,
                ScatterReduceOp::Mean,
            ] {
                let (shader, _, entry) = reduce_kernel(dtype, op).expect("supported pair");
                assert!(shader.contains(&format!("fn {entry}(")), "{entry}");
            }
        }
        assert!(reduce_kernel(DType::F64, ScatterReduceOp::Sum).is_err());
    }
}
