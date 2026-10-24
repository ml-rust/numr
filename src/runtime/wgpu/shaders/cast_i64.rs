//! Launcher for the casts between I64 and the 32-bit dtypes (F32, I32, U32).
//!
//! WGSL has no 64-bit integer. The kernels bind an I64 buffer as
//! `array<vec2<u32>>` (low word, high word) and decode it in u32 arithmetic.
//! Every pair matches the CPU reference, which converts through f64 and then
//! applies a Rust `as` cast:
//!
//! - I64 -> I32 and I64 -> U32 saturate to the target range.
//! - I64 -> F32 rounds ties-to-even twice: to 53 bits, then to 24 bits.
//! - I32 -> I64 sign-extends. U32 -> I64 zero-extends.
//! - F32 -> I64 truncates toward zero, saturates at the i64 bounds, and maps
//!   NaN to 0.
//!
//! The kernels spread threads over a 2-D grid, so a tensor of any element
//! count that fits u32 runs.

use wgpu::{Buffer, Queue};

use super::copy::dispatch_grid;
use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

const CAST_I64_TO_F32_SHADER: &str = concat!(
    include_str!("cast_i64_common.wgsl"),
    include_str!("cast_i64_to_f32.wgsl")
);
const CAST_I64_TO_I32_SHADER: &str = concat!(
    include_str!("cast_i64_common.wgsl"),
    include_str!("cast_i64_to_i32.wgsl")
);
const CAST_I64_TO_U32_SHADER: &str = concat!(
    include_str!("cast_i64_common.wgsl"),
    include_str!("cast_i64_to_u32.wgsl")
);
const CAST_F32_TO_I64_SHADER: &str = concat!(
    include_str!("cast_i64_common.wgsl"),
    include_str!("cast_f32_to_i64.wgsl")
);
const CAST_I32_TO_I64_SHADER: &str = concat!(
    include_str!("cast_i64_common.wgsl"),
    include_str!("cast_i32_to_i64.wgsl")
);
const CAST_U32_TO_I64_SHADER: &str = concat!(
    include_str!("cast_i64_common.wgsl"),
    include_str!("cast_u32_to_i64.wgsl")
);

/// Module name, entry point and source for an I64 cast pair, or `None` when
/// the pair is not one of the six this launcher covers.
fn i64_cast_kernel(
    src_dtype: DType,
    dst_dtype: DType,
) -> Option<(&'static str, &'static str, &'static str)> {
    let kernel = match (src_dtype, dst_dtype) {
        (DType::I64, DType::F32) => ("cast_i64_f32", "cast_i64_to_f32", CAST_I64_TO_F32_SHADER),
        (DType::I64, DType::I32) => ("cast_i64_i32", "cast_i64_to_i32", CAST_I64_TO_I32_SHADER),
        (DType::I64, DType::U32) => ("cast_i64_u32", "cast_i64_to_u32", CAST_I64_TO_U32_SHADER),
        (DType::F32, DType::I64) => ("cast_f32_i64", "cast_f32_to_i64", CAST_F32_TO_I64_SHADER),
        (DType::I32, DType::I64) => ("cast_i32_i64", "cast_i32_to_i64", CAST_I32_TO_I64_SHADER),
        (DType::U32, DType::I64) => ("cast_u32_i64", "cast_u32_to_i64", CAST_U32_TO_I64_SHADER),
        _ => return None,
    };
    Some(kernel)
}

/// True when `src_dtype -> dst_dtype` is an I64 cast this launcher runs.
pub fn is_i64_cast_pair(src_dtype: DType, dst_dtype: DType) -> bool {
    i64_cast_kernel(src_dtype, dst_dtype).is_some()
}

/// Launch an I64 cast: `out[i] = DstType(a[i])` with CPU `as` semantics.
///
/// `params_buffer` holds a `CastParams { numel }`. `numel` must be nonzero.
///
/// # Errors
///
/// `UnsupportedDType` when the pair is not I64 <-> F32, I32 or U32.
pub fn launch_cast_i64_op(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    src_dtype: DType,
    dst_dtype: DType,
) -> Result<()> {
    let Some((module_name, entry_point, shader_source)) = i64_cast_kernel(src_dtype, dst_dtype)
    else {
        let dtype = if src_dtype == DType::I64 {
            dst_dtype
        } else {
            src_dtype
        };
        return Err(Error::UnsupportedDType {
            dtype,
            op: "cast (WebGPU I64 kernel)",
        });
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let (groups_x, groups_y) = dispatch_grid(numel);
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cast_i64"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cast_i64"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
