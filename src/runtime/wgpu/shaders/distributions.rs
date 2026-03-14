//! Distribution sampling WGSL kernel launchers (F32 only on WebGPU)
//!
//! Provides launchers for probability distribution sampling:
//! - Bernoulli, Beta, Gamma, Exponential, Poisson
//! - Binomial, Laplace, Chi-squared, Student's t, F distribution

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const BERNOULLI_SHADER: &str = include_str!("bernoulli_f32.wgsl");
// entry point: "bernoulli_f32"

const BETA_DIST_SHADER: &str = include_str!("beta_dist_f32.wgsl");
// entry point: "beta_dist_f32"

const GAMMA_DIST_SHADER: &str = include_str!("gamma_dist_f32.wgsl");
// entry point: "gamma_dist_f32"

const EXPONENTIAL_SHADER: &str = include_str!("exponential_f32.wgsl");
// entry point: "exponential_f32"

const POISSON_SHADER: &str = include_str!("poisson_f32.wgsl");
// entry point: "poisson_f32"

const BINOMIAL_SHADER: &str = include_str!("binomial_f32.wgsl");
// entry point: "binomial_f32"

const LAPLACE_SHADER: &str = include_str!("laplace_f32.wgsl");
// entry point: "laplace_f32"

const CHI_SQUARED_SHADER: &str = include_str!("chi_squared_f32.wgsl");
// entry point: "chi_squared_f32"

const STUDENT_T_SHADER: &str = include_str!("student_t_f32.wgsl");
// entry point: "student_t_f32"

const F_DISTRIBUTION_SHADER: &str = include_str!("f_distribution_f32.wgsl");
// entry point: "f_distribution_f32"

const MULTINOMIAL_COUNT_SHADER: &str = include_str!("multinomial_count_f32.wgsl");
// entry point: "multinomial_count_f32"

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

    let module = cache.get_or_create_module("bernoulli_f32", BERNOULLI_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("bernoulli_f32", "bernoulli_f32", &module, &layout);
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

    let module = cache.get_or_create_module("beta_dist_f32", BETA_DIST_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("beta_dist_f32", "beta_dist_f32", &module, &layout);
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

    let module = cache.get_or_create_module("gamma_dist_f32", GAMMA_DIST_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("gamma_dist_f32", "gamma_dist_f32", &module, &layout);
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

    let module = cache.get_or_create_module("exponential_f32", EXPONENTIAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("exponential_f32", "exponential_f32", &module, &layout);
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

    let module = cache.get_or_create_module("poisson_f32", POISSON_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("poisson_f32", "poisson_f32", &module, &layout);
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

    let module = cache.get_or_create_module("binomial_f32", BINOMIAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("binomial_f32", "binomial_f32", &module, &layout);
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

    let module = cache.get_or_create_module("laplace_f32", LAPLACE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("laplace_f32", "laplace_f32", &module, &layout);
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

    let module = cache.get_or_create_module("chi_squared_f32", CHI_SQUARED_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("chi_squared_f32", "chi_squared_f32", &module, &layout);
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

    let module = cache.get_or_create_module("student_t_f32", STUDENT_T_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("student_t_f32", "student_t_f32", &module, &layout);
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

    let module = cache.get_or_create_module("f_distribution_f32", F_DISTRIBUTION_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("f_distribution_f32", "f_distribution_f32", &module, &layout);
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
/// * `cdf` - CDF array buffer `[k]`
/// * `uniforms` - Uniform samples buffer `[n_samples, n_trials]`
/// * `counts` - Output counts buffer `[n_samples, k]`
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

    let module = cache.get_or_create_module("multinomial_count_f32", MULTINOMIAL_COUNT_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "multinomial_count_f32",
        "multinomial_count_f32",
        &module,
        &layout,
    );
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
