//! Shape operation WGSL kernel launchers
//!
//! Provides launchers for shape operations including:
//! - cat: Concatenate tensors along a dimension
//! - stack: Stack tensors along a new dimension (uses cat + unsqueeze)
//! - split/chunk: Zero-copy views using narrow (no kernel needed)
//!
//! All copy operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::generator::{
    generate_arange_shader, generate_cat_shader, generate_eye_shader, generate_linspace_shader,
    generate_rand_shader, generate_randint_shader, generate_randn_shader,
};
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

// ============================================================================
// Arange Operation
// ============================================================================

/// Get the kernel name for arange operation.
fn arange_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("arange_f32"),
        DType::I32 => Ok("arange_i32"),
        DType::U32 => Ok("arange_u32"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "arange",
        }),
    }
}

/// Launch an arange operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing ArangeParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output
pub fn launch_arange(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let name = arange_kernel_name(dtype)?;
    let shader_source = generate_arange_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arange"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("arange"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Linspace Operation
// ============================================================================

/// Get the kernel name for linspace operation.
fn linspace_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("linspace_f32"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "linspace",
        }),
    }
}

/// Launch a linspace operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing LinspaceParams
/// * `steps` - Number of steps to generate
/// * `dtype` - Data type of output (must be float)
pub fn launch_linspace(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    steps: usize,
    dtype: DType,
) -> Result<()> {
    if steps == 0 {
        return Ok(());
    }

    let name = linspace_kernel_name(dtype)?;
    let shader_source = generate_linspace_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("linspace"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("linspace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(steps), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Eye Operation
// ============================================================================

/// Get the kernel name for eye operation.
fn eye_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("eye_f32"),
        DType::I32 => Ok("eye_i32"),
        DType::U32 => Ok("eye_u32"),
        _ => Err(Error::UnsupportedDType { dtype, op: "eye" }),
    }
}

/// Launch an eye (identity matrix) operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing EyeParams
/// * `numel` - Total elements (n * m)
/// * `dtype` - Data type of output
pub fn launch_eye(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let name = eye_kernel_name(dtype)?;
    let shader_source = generate_eye_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eye") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("eye"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Random Operations
// ============================================================================

/// Get the kernel name for rand operation.
fn rand_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("rand_f32"),
        _ => Err(Error::UnsupportedDType { dtype, op: "rand" }),
    }
}

/// Launch a rand operation kernel (uniform [0, 1)).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing RandParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output (must be F32)
pub fn launch_rand(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let name = rand_kernel_name(dtype)?;
    let shader_source = generate_rand_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rand"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rand"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Get the kernel name for randn operation.
fn randn_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("randn_f32"),
        _ => Err(Error::UnsupportedDType { dtype, op: "randn" }),
    }
}

/// Launch a randn operation kernel (standard normal N(0, 1)).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing RandnParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output (must be F32)
pub fn launch_randn(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let name = randn_kernel_name(dtype)?;
    let shader_source = generate_randn_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("randn"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("randn"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Get the kernel name for randint operation.
fn randint_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::I32 => Ok("randint_i32"),
        DType::U32 => Ok("randint_u32"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "randint",
        }),
    }
}

/// Launch a randint operation kernel (uniform integers in [low, high)).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing RandintParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output (must be I32 or U32)
pub fn launch_randint(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let name = randint_kernel_name(dtype)?;
    let shader_source = generate_randint_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("randint"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("randint"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
