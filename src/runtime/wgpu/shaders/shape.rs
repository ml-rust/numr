//! Shape operation WGSL kernel launchers
//!
//! Provides launchers for shape operations including:
//! - cat: Concatenate tensors along a dimension
//! - stack: Stack tensors along a new dimension (uses cat + unsqueeze)
//! - split/chunk: Zero-copy views using narrow (no kernel needed)
//!
//! All copy operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::generator::generate_cat_shader;
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if dtype is supported for shape operations on WebGPU.
fn check_dtype_supported(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Get the static module/entry point name for a shape operation.
fn kernel_name(op: &'static str, dtype: DType) -> Result<&'static str> {
    match (op, dtype) {
        ("cat_copy", DType::F32) => Ok("cat_copy_f32"),
        ("cat_copy", DType::I32) => Ok("cat_copy_i32"),
        ("cat_copy", DType::U32) => Ok("cat_copy_u32"),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Cat Copy Operation
// ============================================================================

/// Launch a cat_copy operation kernel.
///
/// Copies data from a source tensor to the appropriate position in the
/// concatenated output tensor. This is called once per input tensor.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer (must be contiguous)
/// * `dst` - Destination tensor buffer (output)
/// * `params_buffer` - Uniform buffer containing CatParams
/// * `total_elements` - Total elements in source tensor
/// * `dtype` - Data type of tensors
#[allow(clippy::too_many_arguments)]
pub fn launch_cat_copy(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    check_dtype_supported(dtype, "cat_copy")?;

    let name = kernel_name("cat_copy", dtype)?;
    let shader_source = generate_cat_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cat_copy"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cat_copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
