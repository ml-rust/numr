//! FFT kernel launchers for WebGPU
//!
//! Provides dispatch functions for FFT compute shaders.

use super::generator::{
    MAX_WORKGROUP_FFT_SIZE, generate_fftshift_shader, generate_hermitian_extend_shader,
    generate_irfft_unpack_shader, generate_rfft_pack_shader, generate_rfft_truncate_shader,
    generate_stockham_fft_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::error::Result;
use wgpu::{Buffer, Queue};

/// Launch batched Stockham FFT for small transforms (N <= MAX_WORKGROUP_FFT_SIZE)
///
/// Each workgroup processes one FFT using shared memory.
pub fn launch_stockham_fft_batched(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    if n > MAX_WORKGROUP_FFT_SIZE {
        return Err(crate::error::Error::Internal(format!(
            "FFT size {} exceeds max workgroup FFT size {}",
            n, MAX_WORKGROUP_FFT_SIZE
        )));
    }

    let shader = generate_stockham_fft_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("stockham_fft", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline = pipeline_cache.get_or_create_dynamic_pipeline(
        "stockham_fft",
        "stockham_fft_small",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stockham_fft_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch single stage of Stockham FFT for large transforms
pub fn launch_stockham_fft_stage(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let shader = generate_stockham_fft_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("stockham_fft", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline = pipeline_cache.get_or_create_dynamic_pipeline(
        "stockham_fft",
        "stockham_fft_stage",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft_stage_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fft_stage_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One thread per butterfly operation
        let butterflies = n / 2;
        pass.dispatch_workgroups(workgroup_count(butterflies), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch scale complex shader
pub fn launch_scale_complex(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
) -> Result<()> {
    let shader = generate_stockham_fft_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("stockham_fft", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline = pipeline_cache.get_or_create_dynamic_pipeline(
        "stockham_fft",
        "scale_complex",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scale_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scale_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch fftshift shader
pub fn launch_fftshift(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let shader = generate_fftshift_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("fftshift", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline("fftshift", "fftshift", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fftshift_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fftshift_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch ifftshift shader
pub fn launch_ifftshift(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let shader = generate_fftshift_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("fftshift", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline("fftshift", "ifftshift", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ifftshift_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ifftshift_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch rfft_pack shader (real to complex)
pub fn launch_rfft_pack(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let shader = generate_rfft_pack_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("rfft_pack", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline("rfft_pack", "rfft_pack", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rfft_pack_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rfft_pack_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch irfft_unpack shader (complex to real)
pub fn launch_irfft_unpack(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let shader = generate_irfft_unpack_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("irfft_unpack", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline = pipeline_cache.get_or_create_dynamic_pipeline(
        "irfft_unpack",
        "irfft_unpack",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("irfft_unpack_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("irfft_unpack_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch hermitian_extend shader
pub fn launch_hermitian_extend(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let shader = generate_hermitian_extend_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("hermitian_extend", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline = pipeline_cache.get_or_create_dynamic_pipeline(
        "hermitian_extend",
        "hermitian_extend",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hermitian_extend_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hermitian_extend_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch rfft_truncate shader
pub fn launch_rfft_truncate(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    half_n: usize,
    batch_size: usize,
) -> Result<()> {
    let shader = generate_rfft_truncate_shader()?;
    let module = pipeline_cache.get_or_create_module_from_source("rfft_truncate", &shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline = pipeline_cache.get_or_create_dynamic_pipeline(
        "rfft_truncate",
        "rfft_truncate",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rfft_truncate_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rfft_truncate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(half_n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
