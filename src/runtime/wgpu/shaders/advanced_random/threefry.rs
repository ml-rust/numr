//! ThreeFry4x32-20 PRNG WGSL shaders and launchers
//!
//! Counter-based PRNG with cryptographic quality.
//! Reference: Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

use wgpu::{Buffer, Queue};

use super::check_float_dtype;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache, workgroup_count};

const THREEFRY_UNIFORM_WGSL: &str = r#"
const THREEFRY_ROTATIONS: array<u32, 8> = array(10u, 26u, 11u, 21u, 13u, 27u, 23u, 5u);
const SKEIN_KS_PARITY: u32 = 0x1BD11BDAu;

struct ThreeFryParams {
    numel: u32,
    key_lo: u32,
    key_hi: u32,
    counter_lo: u32,
    counter_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: ThreeFryParams;

fn rotl(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn threefry4x32_20(ctr: vec4<u32>, key: vec4<u32>) -> vec4<u32> {
    var x = ctr;
    var ks = array<u32, 5>(key.x, key.y, key.z, key.w,
                          SKEIN_KS_PARITY ^ key.x ^ key.y ^ key.z ^ key.w);
    x.x = x.x + ks[0]; x.y = x.y + ks[1]; x.z = x.z + ks[2]; x.w = x.w + ks[3];

    for (var round = 0u; round < 20u; round++) {
        let rot_idx = round % 8u;
        x.x = x.x + x.y; x.y = rotl(x.y, THREEFRY_ROTATIONS[rot_idx]); x.y = x.y ^ x.x;
        x.z = x.z + x.w; x.w = rotl(x.w, THREEFRY_ROTATIONS[(rot_idx + 4u) % 8u]); x.w = x.w ^ x.z;
        let tmp = x.y; x.y = x.w; x.w = tmp;
        if ((round + 1u) % 4u == 0u) {
            let inject = (round + 1u) / 4u;
            x.x = x.x + ks[(inject + 0u) % 5u];
            x.y = x.y + ks[(inject + 1u) % 5u];
            x.z = x.z + ks[(inject + 2u) % 5u];
            x.w = x.w + ks[(inject + 3u) % 5u] + inject;
        }
    }
    return x;
}

@compute @workgroup_size(256)
fn threefry_uniform_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 4u;
    if (base_idx >= params.numel) { return; }
    let counter = vec4<u32>(params.counter_lo + gid.x, params.counter_hi, 0u, 0u);
    let key = vec4<u32>(params.key_lo, params.key_hi, 0u, 0u);
    let random = threefry4x32_20(counter, key);
    for (var j = 0u; j < 4u; j++) {
        let idx = base_idx + j;
        if (idx < params.numel) { output[idx] = f32(random[j]) / 4294967296.0; }
    }
}
"#;

const THREEFRY_RANDN_WGSL: &str = r#"
const THREEFRY_ROTATIONS: array<u32, 8> = array(10u, 26u, 11u, 21u, 13u, 27u, 23u, 5u);
const SKEIN_KS_PARITY: u32 = 0x1BD11BDAu;
const PI: f32 = 3.14159265359;

struct ThreeFryParams {
    numel: u32,
    key_lo: u32,
    key_hi: u32,
    counter_lo: u32,
    counter_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: ThreeFryParams;

fn rotl(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn threefry4x32_20(ctr: vec4<u32>, key: vec4<u32>) -> vec4<u32> {
    var x = ctr;
    var ks = array<u32, 5>(key.x, key.y, key.z, key.w,
                          SKEIN_KS_PARITY ^ key.x ^ key.y ^ key.z ^ key.w);
    x.x = x.x + ks[0]; x.y = x.y + ks[1]; x.z = x.z + ks[2]; x.w = x.w + ks[3];

    for (var round = 0u; round < 20u; round++) {
        let rot_idx = round % 8u;
        x.x = x.x + x.y; x.y = rotl(x.y, THREEFRY_ROTATIONS[rot_idx]); x.y = x.y ^ x.x;
        x.z = x.z + x.w; x.w = rotl(x.w, THREEFRY_ROTATIONS[(rot_idx + 4u) % 8u]); x.w = x.w ^ x.z;
        let tmp = x.y; x.y = x.w; x.w = tmp;
        if ((round + 1u) % 4u == 0u) {
            let inject = (round + 1u) / 4u;
            x.x = x.x + ks[(inject + 0u) % 5u];
            x.y = x.y + ks[(inject + 1u) % 5u];
            x.z = x.z + ks[(inject + 2u) % 5u];
            x.w = x.w + ks[(inject + 3u) % 5u] + inject;
        }
    }
    return x;
}

fn box_muller(u1: f32, u2: f32) -> vec2<f32> {
    let r = sqrt(-2.0 * log(max(u1, 1e-10)));
    let theta = 2.0 * PI * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(256)
fn threefry_randn_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 2u;
    if (base_idx >= params.numel) { return; }
    let counter = vec4<u32>(params.counter_lo + gid.x, params.counter_hi, 0u, 0u);
    let key = vec4<u32>(params.key_lo, params.key_hi, 0u, 0u);
    let random = threefry4x32_20(counter, key);
    let u1 = f32(random[0]) / 4294967296.0;
    let u2 = f32(random[1]) / 4294967296.0;
    let normals = box_muller(u1, u2);
    if (base_idx < params.numel) { output[base_idx] = normals.x; }
    if (base_idx + 1u < params.numel) { output[base_idx + 1u] = normals.y; }
}
"#;

/// Launch ThreeFry uniform kernel
pub fn launch_threefry_uniform(
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
    check_float_dtype(dtype, "threefry_uniform")?;

    let name = "threefry_uniform_f32";
    let module = cache.get_or_create_module(name, THREEFRY_UNIFORM_WGSL);
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
            label: Some("threefry_uniform"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("threefry_uniform"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count((numel + 3) / 4), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch ThreeFry randn kernel (Box-Muller transform)
pub fn launch_threefry_randn(
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
    check_float_dtype(dtype, "threefry_randn")?;

    let name = "threefry_randn_f32";
    let module = cache.get_or_create_module(name, THREEFRY_RANDN_WGSL);
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
            label: Some("threefry_randn"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("threefry_randn"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count((numel + 1) / 2), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
