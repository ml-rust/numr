//! Philox4x32-10 PRNG WGSL shaders and launchers
//!
//! Counter-based PRNG used by JAX and TensorFlow. Excellent GPU performance.
//! Reference: Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

use wgpu::{Buffer, Queue};

use super::check_float_dtype;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache, workgroup_count};

const PHILOX_UNIFORM_WGSL: &str = r#"
const PHILOX_M2X32_0: u32 = 0xD2511F53u;
const PHILOX_M2X32_1: u32 = 0xCD9E8D57u;
const PHILOX_W32_0: u32 = 0x9E3779B9u;
const PHILOX_W32_1: u32 = 0xBB67AE85u;

struct PhiloxParams {
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
@group(0) @binding(1) var<uniform> params: PhiloxParams;

fn mulhi(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let cross = a_lo * b_hi + a_hi * b_lo;
    return a_hi * b_hi + (cross >> 16u);
}

fn philox_round(ctr: ptr<function, vec4<u32>>, key: ptr<function, vec2<u32>>) {
    let prod0_lo = (*ctr).x * PHILOX_M2X32_0;
    let prod0_hi = mulhi((*ctr).x, PHILOX_M2X32_0);
    let prod1_lo = (*ctr).z * PHILOX_M2X32_1;
    let prod1_hi = mulhi((*ctr).z, PHILOX_M2X32_1);
    *ctr = vec4<u32>(prod1_hi ^ (*ctr).y ^ (*key).x, prod1_lo, prod0_hi ^ (*ctr).w ^ (*key).y, prod0_lo);
}

fn philox4x32_10(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var ctr = counter;
    var k = key;
    for (var round = 0u; round < 10u; round++) {
        philox_round(&ctr, &k);
        k.x = k.x + PHILOX_W32_0;
        k.y = k.y + PHILOX_W32_1;
    }
    return ctr;
}

@compute @workgroup_size(256)
fn philox_uniform_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 4u;
    if (base_idx >= params.numel) { return; }
    let counter = vec4<u32>(params.counter_lo + gid.x, params.counter_hi, 0u, 0u);
    let random = philox4x32_10(counter, vec2<u32>(params.key_lo, params.key_hi));
    for (var j = 0u; j < 4u; j++) {
        let idx = base_idx + j;
        if (idx < params.numel) {
            output[idx] = f32(random[j]) / 4294967296.0;
        }
    }
}
"#;

const PHILOX_RANDN_WGSL: &str = r#"
const PHILOX_M2X32_0: u32 = 0xD2511F53u;
const PHILOX_M2X32_1: u32 = 0xCD9E8D57u;
const PHILOX_W32_0: u32 = 0x9E3779B9u;
const PHILOX_W32_1: u32 = 0xBB67AE85u;
const PI: f32 = 3.14159265359;

struct PhiloxParams {
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
@group(0) @binding(1) var<uniform> params: PhiloxParams;

fn mulhi(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let cross = a_lo * b_hi + a_hi * b_lo;
    return a_hi * b_hi + (cross >> 16u);
}

fn philox_round(ctr: ptr<function, vec4<u32>>, key: ptr<function, vec2<u32>>) {
    let prod0_lo = (*ctr).x * PHILOX_M2X32_0;
    let prod0_hi = mulhi((*ctr).x, PHILOX_M2X32_0);
    let prod1_lo = (*ctr).z * PHILOX_M2X32_1;
    let prod1_hi = mulhi((*ctr).z, PHILOX_M2X32_1);
    *ctr = vec4<u32>(prod1_hi ^ (*ctr).y ^ (*key).x, prod1_lo, prod0_hi ^ (*ctr).w ^ (*key).y, prod0_lo);
}

fn philox4x32_10(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var ctr = counter;
    var k = key;
    for (var round = 0u; round < 10u; round++) {
        philox_round(&ctr, &k);
        k.x = k.x + PHILOX_W32_0;
        k.y = k.y + PHILOX_W32_1;
    }
    return ctr;
}

fn box_muller(u1: f32, u2: f32) -> vec2<f32> {
    let r = sqrt(-2.0 * log(max(u1, 1e-10)));
    let theta = 2.0 * PI * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(256)
fn philox_randn_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 2u;
    if (base_idx >= params.numel) { return; }
    let counter = vec4<u32>(params.counter_lo + gid.x, params.counter_hi, 0u, 0u);
    let random = philox4x32_10(counter, vec2<u32>(params.key_lo, params.key_hi));
    let u1 = f32(random[0]) / 4294967296.0;
    let u2 = f32(random[1]) / 4294967296.0;
    let normals = box_muller(u1, u2);
    if (base_idx < params.numel) { output[base_idx] = normals.x; }
    if (base_idx + 1u < params.numel) { output[base_idx + 1u] = normals.y; }
}
"#;

/// Launch Philox uniform kernel
pub fn launch_philox_uniform(
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
    check_float_dtype(dtype, "philox_uniform")?;

    let name = "philox_uniform_f32";
    let module = cache.get_or_create_module(name, PHILOX_UNIFORM_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("philox_uniform"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("philox_uniform"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count((numel + 3) / 4), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Philox randn kernel (Box-Muller transform)
pub fn launch_philox_randn(
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
    check_float_dtype(dtype, "philox_randn")?;

    let name = "philox_randn_f32";
    let module = cache.get_or_create_module(name, PHILOX_RANDN_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("philox_randn"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("philox_randn"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count((numel + 1) / 2), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
