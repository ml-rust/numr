//! Batched triangular substitution: one dispatch for every matrix and column.

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::linalg_shaders::triangular_batched::TRIANGULAR_BATCHED_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Invocations per workgroup; matches `@workgroup_size` in the shader.
const WORKGROUP_SIZE: usize = 64;
/// Largest workgroup count along one dispatch axis.
const GRID_X: usize = 65535;

/// The 2-D workgroup grid that covers `threads` invocations.
///
/// # Errors
///
/// `BackendLimitation` when the padded grid exceeds the shader's 32-bit
/// invocation index.
pub fn triangular_batched_grid(threads: usize) -> Result<(u32, u32)> {
    let groups = threads.div_ceil(WORKGROUP_SIZE).max(1);
    let x = groups.min(GRID_X);
    let y = groups.div_ceil(x);
    let padded = (x as u64) * (y as u64) * (WORKGROUP_SIZE as u64);
    if y > GRID_X || padded > u32::MAX as u64 {
        return Err(Error::BackendLimitation {
            backend: "wgpu",
            operation: "triangular_solve_batched",
            reason: format!("{threads} invocations exceed the 32-bit invocation index"),
        });
    }
    Ok((x as u32, y as u32))
}

/// Launch batched substitution over `[batch, n, n]` against `[batch, n, nrhs]`.
///
/// `params_buffer` holds `[n, nrhs, batch, unit_diagonal]` as `u32`.
/// `threads` is `batch * nrhs`. With `lower`, forward substitution runs.
/// Otherwise backward substitution runs and `unit_diagonal` is ignored.
#[allow(clippy::too_many_arguments)]
pub fn launch_triangular_solve_batched(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    x: &Buffer,
    params_buffer: &Buffer,
    threads: usize,
    lower: bool,
    dtype: DType,
) -> Result<()> {
    let label = if lower {
        "forward_sub_batched"
    } else {
        "backward_sub_batched"
    };
    check_dtype_f32!(dtype, label);
    let (grid_x, grid_y) = triangular_batched_grid(threads)?;
    let entry_point = if lower {
        "forward_sub_batched_f32"
    } else {
        "backward_sub_batched_f32"
    };

    let module = cache.get_or_create_module("linalg_triangular_batched", TRIANGULAR_BATCHED_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_triangular_batched", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, x, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(grid_x, grid_y, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_covers_threads() {
        for threads in [0, 1, 63, 64, 65, 65535 * 64, 65535 * 64 + 1, 10_000_000] {
            let (x, y) = triangular_batched_grid(threads).unwrap();
            let covered = x as usize * y as usize * WORKGROUP_SIZE;
            assert!(covered >= threads, "{threads}: {x} x {y}");
            assert!(x as usize <= GRID_X && y as usize <= GRID_X);
        }
    }

    #[test]
    fn grid_rejects_past_u32() {
        assert!(triangular_batched_grid(u32::MAX as usize + 1).is_err());
    }
}
