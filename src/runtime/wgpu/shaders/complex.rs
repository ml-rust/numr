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
        num_readonly_storage: 0,
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
        num_readonly_storage: 0,
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

/// Launch from_real_imag operation on WebGPU.
///
/// Constructs Complex64 from separate F32 real and imaginary arrays.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WebGPU queue for command submission
/// * `real_buf` - Input buffer (F32 real parts)
/// * `imag_buf` - Input buffer (F32 imaginary parts)
/// * `output_buf` - Output buffer (Complex64 / vec2<f32> data)
/// * `params_buf` - Parameters buffer containing numel
/// * `numel` - Number of elements
pub fn launch_from_real_imag(
    cache: &PipelineCache,
    queue: &Queue,
    real_buf: &Buffer,
    imag_buf: &Buffer,
    output_buf: &Buffer,
    params_buf: &Buffer,
    numel: usize,
) -> Result<()> {
    let shader_src = super::generator::complex::generate_from_real_imag_shader()?;
    let entry_point = "from_real_imag_f32";
    let module_name = "from_real_imag_f32";

    let module = cache.get_or_create_module_from_source(&module_name, &shader_src);
    let layout = cache.get_or_create_layout(super::pipeline::LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&module_name, &entry_point, &module, &layout);
    let bind_group =
        cache.create_bind_group(&layout, &[real_buf, imag_buf, output_buf, params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("From Real Imag"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("From Real Imag"),
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

/// Launch complex × real multiplication operation on WebGPU.
///
/// Computes (a + bi) * r = ar + br*i element-wise.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WebGPU queue for command submission
/// * `complex_buf` - Input buffer (Complex64 / vec2<f32> data)
/// * `real_buf` - Input buffer (F32 real coefficients)
/// * `output_buf` - Output buffer (Complex64 / vec2<f32> data)
/// * `params_buf` - Parameters buffer containing numel
/// * `numel` - Number of elements
pub fn launch_complex_mul_real(
    cache: &PipelineCache,
    queue: &Queue,
    complex_buf: &Buffer,
    real_buf: &Buffer,
    output_buf: &Buffer,
    params_buf: &Buffer,
    numel: usize,
) -> Result<()> {
    let shader_src = super::generator::complex::generate_complex_mul_real_shader()?;
    let entry_point = "complex64_mul_real";
    let module_name = "complex64_mul_real";

    let module = cache.get_or_create_module_from_source(&module_name, &shader_src);
    let layout = cache.get_or_create_layout(super::pipeline::LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&module_name, &entry_point, &module, &layout);
    let bind_group =
        cache.create_bind_group(&layout, &[complex_buf, real_buf, output_buf, params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Complex Mul Real"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Complex Mul Real"),
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

/// Launch complex / real division operation on WebGPU.
///
/// Computes (a + bi) / r = (a/r) + (b/r)*i element-wise.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WebGPU queue for command submission
/// * `complex_buf` - Input buffer (Complex64 / vec2<f32> data)
/// * `real_buf` - Input buffer (F32 real divisors)
/// * `output_buf` - Output buffer (Complex64 / vec2<f32> data)
/// * `params_buf` - Parameters buffer containing numel
/// * `numel` - Number of elements
pub fn launch_complex_div_real(
    cache: &PipelineCache,
    queue: &Queue,
    complex_buf: &Buffer,
    real_buf: &Buffer,
    output_buf: &Buffer,
    params_buf: &Buffer,
    numel: usize,
) -> Result<()> {
    let shader_src = super::generator::complex::generate_complex_div_real_shader()?;
    let entry_point = "complex64_div_real";
    let module_name = "complex64_div_real";

    let module = cache.get_or_create_module_from_source(&module_name, &shader_src);
    let layout = cache.get_or_create_layout(super::pipeline::LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&module_name, &entry_point, &module, &layout);
    let bind_group =
        cache.create_bind_group(&layout, &[complex_buf, real_buf, output_buf, params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Complex Div Real"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Complex Div Real"),
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
