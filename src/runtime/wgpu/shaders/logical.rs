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

/// WGSL shader for binary logical operations on U32 tensors.
const LOGICAL_BINARY_SHADER: &str = r#"
struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn logical_and(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        out[idx] = select(0u, 1u, a[idx] != 0u && b[idx] != 0u);
    }
}

@compute @workgroup_size(256)
fn logical_or(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        out[idx] = select(0u, 1u, a[idx] != 0u || b[idx] != 0u);
    }
}

@compute @workgroup_size(256)
fn logical_xor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        let a_bool = a[idx] != 0u;
        let b_bool = b[idx] != 0u;
        out[idx] = select(0u, 1u, a_bool != b_bool);
    }
}
"#;

/// WGSL shader for unary logical operations on U32 tensors.
const LOGICAL_UNARY_SHADER: &str = r#"
struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn logical_not(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        out[idx] = select(1u, 0u, a[idx] != 0u);
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
    let module = cache.get_or_create_module("logical_binary", LOGICAL_BINARY_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let pipeline = cache.get_or_create_pipeline("logical_binary", "logical_and", &module, &layout);

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
    let module = cache.get_or_create_module("logical_binary", LOGICAL_BINARY_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let pipeline = cache.get_or_create_pipeline("logical_binary", "logical_or", &module, &layout);

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
    let module = cache.get_or_create_module("logical_binary", LOGICAL_BINARY_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let pipeline = cache.get_or_create_pipeline("logical_binary", "logical_xor", &module, &layout);

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
    let module = cache.get_or_create_module("logical_unary", LOGICAL_UNARY_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 1,
    });
    let pipeline = cache.get_or_create_pipeline("logical_unary", "logical_not", &module, &layout);

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
    fn test_logical_binary_shader_syntax() {
        validate_wgsl_syntax(LOGICAL_BINARY_SHADER)
            .expect("Logical binary shader should be valid WGSL");
    }

    #[test]
    fn test_logical_unary_shader_syntax() {
        validate_wgsl_syntax(LOGICAL_UNARY_SHADER)
            .expect("Logical unary shader should be valid WGSL");
    }
}
