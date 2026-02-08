//! Distribution sampling WGSL kernel launchers
//!
//! Provides launchers for probability distribution sampling:
//! - Bernoulli, Beta, Gamma, Exponential, Poisson
//! - Binomial, Laplace, Chi-squared, Student's t, F distribution

use wgpu::{Buffer, Queue};

use super::generator::{
    generate_bernoulli_shader, generate_beta_dist_shader, generate_binomial_shader,
    generate_chi_squared_shader, generate_exponential_shader, generate_f_distribution_shader,
    generate_gamma_dist_shader, generate_laplace_shader, generate_multinomial_count_shader,
    generate_poisson_shader, generate_student_t_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Launch Bernoulli distribution sampling kernel.
pub fn launch_bernoulli(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "bernoulli")?;

    let name = "bernoulli_f32";
    let shader = generate_bernoulli_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bernoulli"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bernoulli"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Beta distribution sampling kernel.
pub fn launch_beta_dist(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "beta")?;

    let name = "beta_dist_f32";
    let shader = generate_beta_dist_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("beta_dist"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("beta_dist"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Gamma distribution sampling kernel.
pub fn launch_gamma_dist(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "gamma")?;

    let name = "gamma_dist_f32";
    let shader = generate_gamma_dist_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gamma_dist"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gamma_dist"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Exponential distribution sampling kernel.
pub fn launch_exponential(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "exponential")?;

    let name = "exponential_f32";
    let shader = generate_exponential_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exponential"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("exponential"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Poisson distribution sampling kernel.
pub fn launch_poisson(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "poisson")?;

    let name = "poisson_f32";
    let shader = generate_poisson_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("poisson"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("poisson"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Binomial distribution sampling kernel.
pub fn launch_binomial(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "binomial")?;

    let name = "binomial_f32";
    let shader = generate_binomial_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("binomial"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("binomial"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Laplace distribution sampling kernel.
pub fn launch_laplace(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "laplace")?;

    let name = "laplace_f32";
    let shader = generate_laplace_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("laplace"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("laplace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Chi-squared distribution sampling kernel.
pub fn launch_chi_squared(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "chi_squared")?;

    let name = "chi_squared_f32";
    let shader = generate_chi_squared_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("chi_squared"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chi_squared"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Student's t distribution sampling kernel.
pub fn launch_student_t(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "student_t")?;

    let name = "student_t_f32";
    let shader = generate_student_t_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("student_t"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("student_t"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch F distribution sampling kernel.
pub fn launch_f_distribution(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "f_distribution")?;

    let name = "f_distribution_f32";
    let shader = generate_f_distribution_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("f_distribution"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("f_distribution"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Params struct for multinomial count shader (must match WGSL layout)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MultinomialCountParams {
    /// Number of categories
    pub k: u32,
    /// Number of trials per sample
    pub n_trials: u32,
    /// Number of samples
    pub n_samples: u32,
    /// Padding for alignment
    pub _pad: u32,
}

/// Launch multinomial count kernel.
///
/// Performs CDF lookup for uniform samples and counts occurrences per category.
/// Used for multinomial sampling: given uniform samples and a CDF, counts how
/// many samples fall into each category.
///
/// # Arguments
/// * `cdf` - CDF array buffer [k]
/// * `uniforms` - Uniform samples buffer [n_samples, n_trials]
/// * `counts` - Output counts buffer [n_samples, k]
/// * `params` - Parameters buffer containing MultinomialCountParams
/// * `n_samples` - Number of samples (used for workgroup dispatch)
pub fn launch_multinomial_count(
    cache: &PipelineCache,
    queue: &Queue,
    cdf: &Buffer,
    uniforms: &Buffer,
    counts: &Buffer,
    params: &Buffer,
    n_samples: usize,
    dtype: DType,
) -> Result<()> {
    if n_samples == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "multinomial_count")?;

    let name = "multinomial_count_f32";
    let shader = generate_multinomial_count_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[cdf, uniforms, counts, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multinomial_count"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multinomial_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n_samples), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
