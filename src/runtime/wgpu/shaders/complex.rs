//! Complex number operation compute shader launchers for WebGPU

use super::generator::complex::get_complex_shader_generator;
use super::pipeline::PipelineCache;
use crate::dtype::DType;
use crate::error::{Error, Result};
use wgpu::{Buffer, Queue};

/// Launch a complex operation on the GPU.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WebGPU queue for command submission
/// * `op` - Operation name ("conj", "real", "imag", "angle")
/// * `input_buf` - Input buffer (Complex64 data)
/// * `output_buf` - Output buffer (dtype depends on operation)
/// * `params_buf` - Parameters buffer containing numel
/// * `numel` - Number of elements
/// * `input_dtype` - Input data type (must be Complex64 for WebGPU)
pub fn launch_complex_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input_buf: &Buffer,
    output_buf: &Buffer,
    params_buf: &Buffer,
    numel: usize,
    input_dtype: DType,
) -> Result<()> {
    // Validate dtype (WebGPU only supports Complex64)
    if input_dtype != DType::Complex64 {
        let op_static: &'static str = match op {
            "conj" => "conj",
            "real" => "real",
            "imag" => "imag",
            "angle" => "angle",
            _ => "complex_op",
        };
        return Err(Error::UnsupportedDType {
            dtype: input_dtype,
            op: op_static,
        });
    }

    // Get shader generator for this operation
    let shader_gen = get_complex_shader_generator(op)?;
    let shader_src = shader_gen()?;

    // Entry point name: "conj_complex64", "real_complex64", etc.
    let entry_point = format!("{}_{}", op, "complex64");

    // Create shader module
    let module_name = format!("complex_{}_{}", op, "complex64");
    let module = cache.get_or_create_module_from_source(&module_name, &shader_src);

    // Create bind group layout (3 buffers: input storage, output storage, params uniform)
    let layout = cache.get_or_create_layout(super::pipeline::LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    // Get or create pipeline
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&module_name, &entry_point, &module, &layout);

    // Create bind group
    let bind_group = cache.create_bind_group(&layout, &[input_buf, output_buf, params_buf]);

    // Create command encoder
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Complex {}", op)),
        });

    // Dispatch compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("Complex {}", op)),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Workgroup size is 256, calculate number of workgroups
        let workgroup_size = 256;
        let num_workgroups = (numel + workgroup_size - 1) / workgroup_size;

        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch angle_real operation on WebGPU (for real F32 inputs).
///
/// Computes phase angle for real numbers: angle(x) = 0 if x >= 0, π if x < 0
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WebGPU queue for command submission
/// * `input_buf` - Input buffer (F32 data)
/// * `output_buf` - Output buffer (F32 data)
/// * `params_buf` - Parameters buffer containing numel
/// * `numel` - Number of elements
pub fn launch_angle_real(
    cache: &PipelineCache,
    queue: &Queue,
    input_buf: &Buffer,
    output_buf: &Buffer,
    params_buf: &Buffer,
    numel: usize,
) -> Result<()> {
    let shader_src = super::generator::complex::generate_angle_real_shader()?;
    let entry_point = "angle_real_f32";
    let module_name = "angle_real_f32";

    let module = cache.get_or_create_module_from_source(&module_name, &shader_src);
    let layout = cache.get_or_create_layout(super::pipeline::LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&module_name, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input_buf, output_buf, params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Angle Real"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Angle Real"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 256;
        let num_workgroups = (numel + workgroup_size - 1) / workgroup_size;

        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}
