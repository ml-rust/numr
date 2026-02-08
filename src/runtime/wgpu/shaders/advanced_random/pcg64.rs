//! PCG64 PRNG WGSL shaders and launchers
//!
//! Permuted Congruential Generator - NumPy's default PRNG.
//! Uses emulated 64-bit arithmetic since WGSL lacks native u64.
//! Reference: O'Neill "PCG: A Family of Simple Fast Space-Efficient..." (2014)

use wgpu::{Buffer, Queue};

use super::check_float_dtype;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache, workgroup_count};

const PCG64_UNIFORM_WGSL: &str = r#"
struct Pcg64Params {
    numel: u32,
    seed_lo: u32,
    seed_hi: u32,
    stream_lo: u32,
    stream_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: Pcg64Params;

fn add64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo = a_lo + b_lo;
    let carry = select(0u, 1u, lo < a_lo);
    return vec2<u32>(lo, a_hi + b_hi + carry);
}

fn mul64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo_lo = a_lo * b_lo;
    let a0 = a_lo & 0xFFFFu; let a1 = a_lo >> 16u;
    let b0 = b_lo & 0xFFFFu; let b1 = b_lo >> 16u;
    let cross = a0 * b1 + a1 * b0;
    let hi = a1 * b1 + (cross >> 16u) + a_lo * b_hi + a_hi * b_lo;
    return vec2<u32>(lo_lo, hi);
}

fn pcg_output(state_lo: u32, state_hi: u32) -> u32 {
    let xorshifted = ((state_hi << 14u) | (state_lo >> 18u)) ^ (state_hi >> 27u);
    let rot = state_hi >> 27u;
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

@compute @workgroup_size(256)
fn pcg64_uniform_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.numel) { return; }

    var state = add64(params.seed_lo, params.seed_hi, params.stream_lo, params.stream_hi);
    state = add64(state.x, state.y, idx, 0u);

    let mult_lo = 0x5851F42Du; let mult_hi = 0x4C957F2Du;
    let inc_lo = (params.stream_lo << 1u) | 1u;
    let inc_hi = params.stream_hi;

    state = mul64(state.x, state.y, mult_lo, mult_hi);
    state = add64(state.x, state.y, inc_lo, inc_hi);

    output[idx] = f32(pcg_output(state.x, state.y)) / 4294967296.0;
}
"#;

const PCG64_RANDN_WGSL: &str = r#"
const PI: f32 = 3.14159265359;

struct Pcg64Params {
    numel: u32,
    seed_lo: u32,
    seed_hi: u32,
    stream_lo: u32,
    stream_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: Pcg64Params;

fn add64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo = a_lo + b_lo;
    let carry = select(0u, 1u, lo < a_lo);
    return vec2<u32>(lo, a_hi + b_hi + carry);
}

fn mul64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo_lo = a_lo * b_lo;
    let a0 = a_lo & 0xFFFFu; let a1 = a_lo >> 16u;
    let b0 = b_lo & 0xFFFFu; let b1 = b_lo >> 16u;
    let cross = a0 * b1 + a1 * b0;
    let hi = a1 * b1 + (cross >> 16u) + a_lo * b_hi + a_hi * b_lo;
    return vec2<u32>(lo_lo, hi);
}

fn pcg_output(state_lo: u32, state_hi: u32) -> u32 {
    let xorshifted = ((state_hi << 14u) | (state_lo >> 18u)) ^ (state_hi >> 27u);
    let rot = state_hi >> 27u;
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

fn pcg_advance(state: vec2<u32>, inc_lo: u32, inc_hi: u32) -> vec2<u32> {
    let mult_lo = 0x5851F42Du; let mult_hi = 0x4C957F2Du;
    var s = mul64(state.x, state.y, mult_lo, mult_hi);
    return add64(s.x, s.y, inc_lo, inc_hi);
}

fn box_muller(u1: f32, u2: f32) -> vec2<f32> {
    let r = sqrt(-2.0 * log(max(u1, 1e-10)));
    let theta = 2.0 * PI * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(256)
fn pcg64_randn_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 2u;
    if (base_idx >= params.numel) { return; }

    var state = add64(params.seed_lo, params.seed_hi, params.stream_lo, params.stream_hi);
    state = add64(state.x, state.y, gid.x * 2u, 0u);

    let inc_lo = (params.stream_lo << 1u) | 1u;
    let inc_hi = params.stream_hi;

    state = pcg_advance(state, inc_lo, inc_hi);
    let r1 = pcg_output(state.x, state.y);
    state = pcg_advance(state, inc_lo, inc_hi);
    let r2 = pcg_output(state.x, state.y);

    let u1 = f32(r1) / 4294967296.0;
    let u2 = f32(r2) / 4294967296.0;
    let normals = box_muller(u1, u2);

    if (base_idx < params.numel) { output[base_idx] = normals.x; }
    if (base_idx + 1u < params.numel) { output[base_idx + 1u] = normals.y; }
}
"#;

/// Launch PCG64 uniform kernel
pub fn launch_pcg64_uniform(
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
    check_float_dtype(dtype, "pcg64_uniform")?;

    let name = "pcg64_uniform_f32";
    let module = cache.get_or_create_module(name, PCG64_UNIFORM_WGSL);
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
            label: Some("pcg64_uniform"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pcg64_uniform"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch PCG64 randn kernel (Box-Muller transform)
pub fn launch_pcg64_randn(
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
    check_float_dtype(dtype, "pcg64_randn")?;

    let name = "pcg64_randn_f32";
    let module = cache.get_or_create_module(name, PCG64_RANDN_WGSL);
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
            label: Some("pcg64_randn"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pcg64_randn"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count((numel + 1) / 2), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
