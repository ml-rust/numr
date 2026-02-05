//! WGSL shader launchers for logical operations.
//!
//! Provides GPU compute shader execution for element-wise logical operations
//! on U32 tensors (0 = false, non-zero = true).

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::error::Result;

// ============================================================================
// Shader Source
// ============================================================================

/// WGSL shader for logical operations on U32 tensors.
const LOGICAL_SHADER: &str = r#"
// Logical operations for U32 tensors
// 0 = false, non-zero = true

const WORKGROUP_SIZE: u32 = 256u;

// Binary params
struct LogicalBinaryParams {
    numel: u32,
}

// Unary params
struct LogicalUnaryParams {
    numel: u32,
}

// Binary operations
@group(0) @binding(0) var<storage, read> logical_a: array<u32>;
@group(0) @binding(1) var<storage, read> logical_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> logical_out: array<u32>;
@group(0) @binding(3) var<uniform> logical_binary_params: LogicalBinaryParams;

@compute @workgroup_size(256)
fn logical_and(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < logical_binary_params.numel) {
        // Both must be non-zero for result to be 1
        let a_bool = logical_a[idx] != 0u;
        let b_bool = logical_b[idx] != 0u;
        logical_out[idx] = select(0u, 1u, a_bool && b_bool);
    }
}

@compute @workgroup_size(256)
fn logical_or(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < logical_binary_params.numel) {
        // Either must be non-zero for result to be 1
        let a_bool = logical_a[idx] != 0u;
        let b_bool = logical_b[idx] != 0u;
        logical_out[idx] = select(0u, 1u, a_bool || b_bool);
    }
}

@compute @workgroup_size(256)
fn logical_xor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < logical_binary_params.numel) {
        // Exactly one must be non-zero for result to be 1
        let a_bool = logical_a[idx] != 0u;
        let b_bool = logical_b[idx] != 0u;
        logical_out[idx] = select(0u, 1u, a_bool != b_bool);
    }
}

// Unary operations - use different binding layout
@group(0) @binding(0) var<storage, read> logical_not_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> logical_not_out: array<u32>;
@group(0) @binding(2) var<uniform> logical_unary_params: LogicalUnaryParams;

@compute @workgroup_size(256)
fn logical_not(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < logical_unary_params.numel) {
        // If non-zero, output 0; if zero, output 1
        logical_not_out[idx] = select(1u, 0u, logical_not_a[idx] != 0u);
    }
}
"#;

// ============================================================================
// Launcher Functions
// ============================================================================

/// Launch logical AND operation.
pub fn launch_logical_and(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("logical", LOGICAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("logical", "logical_and", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logical_and"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logical_and"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch logical OR operation.
pub fn launch_logical_or(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("logical", LOGICAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("logical", "logical_or", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logical_or"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logical_or"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch logical XOR operation.
pub fn launch_logical_xor(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("logical", LOGICAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("logical", "logical_xor", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logical_xor"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logical_xor"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch logical NOT operation.
pub fn launch_logical_not(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("logical", LOGICAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("logical", "logical_not", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logical_not"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logical_not"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate WGSL shader syntax using naga parser
    fn validate_wgsl_syntax(source: &str) -> std::result::Result<(), String> {
        use wgpu::naga::front::wgsl;
        let mut frontend = wgsl::Frontend::new();
        frontend
            .parse(source)
            .map(|_| ())
            .map_err(|e| format!("WGSL parse error: {e}"))
    }

    #[test]
    fn test_logical_shader_syntax() {
        validate_wgsl_syntax(LOGICAL_SHADER).expect("Logical shader should be valid WGSL");
    }
}
