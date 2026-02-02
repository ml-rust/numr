//! Xoshiro256++ PRNG WGSL shaders and launchers
//!
//! Rust `rand` crate default. Uses emulated 64-bit arithmetic.
//! Reference: Blackman & Vigna "Scrambled Linear Pseudorandom Number Generators"

use wgpu::{Buffer, Queue};

use super::check_float_dtype;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache, workgroup_count};

const XOSHIRO256_UNIFORM_WGSL: &str = r#"
struct Xoshiro256Params {
    numel: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: Xoshiro256Params;

fn rotl64(lo: u32, hi: u32, k: u32) -> vec2<u32> {
    if (k == 0u) { return vec2<u32>(lo, hi); }
    if (k < 32u) {
        return vec2<u32>((lo << k) | (hi >> (32u - k)), (hi << k) | (lo >> (32u - k)));
    }
    let k2 = k - 32u;
    return vec2<u32>((hi << k2) | (lo >> (32u - k2)), (lo << k2) | (hi >> (32u - k2)));
}

fn add64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo = a_lo + b_lo;
    return vec2<u32>(lo, a_hi + b_hi + select(0u, 1u, lo < a_lo));
}

fn xor64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>(a_lo ^ b_lo, a_hi ^ b_hi);
}

fn shr64(lo: u32, hi: u32, k: u32) -> vec2<u32> {
    if (k == 0u) { return vec2<u32>(lo, hi); }
    if (k < 32u) { return vec2<u32>((lo >> k) | (hi << (32u - k)), hi >> k); }
    return vec2<u32>(hi >> (k - 32u), 0u);
}

fn mul64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let a0 = a_lo & 0xFFFFu; let a1 = a_lo >> 16u;
    let b0 = b_lo & 0xFFFFu; let b1 = b_lo >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = p01 + p10 + (p00 >> 16u);
    let lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = p11 + (mid >> 16u) + a_lo * b_hi + a_hi * b_lo;
    return vec2<u32>(lo, hi);
}

fn splitmix64(state: ptr<function, vec2<u32>>) -> vec2<u32> {
    *state = add64((*state).x, (*state).y, 0x7F4A7C15u, 0x9E3779B9u);
    var z = *state;
    let s30 = shr64(z.x, z.y, 30u); z = xor64(z.x, z.y, s30.x, s30.y);
    z = mul64(z.x, z.y, 0x1CE4E5B9u, 0xBF58476Du);
    let s27 = shr64(z.x, z.y, 27u); z = xor64(z.x, z.y, s27.x, s27.y);
    z = mul64(z.x, z.y, 0x133111EBu, 0x94D049BBu);
    let s31 = shr64(z.x, z.y, 31u); z = xor64(z.x, z.y, s31.x, s31.y);
    return z;
}

@compute @workgroup_size(256)
fn xoshiro256_uniform_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.numel) { return; }

    var sm_state = add64(params.seed_lo, params.seed_hi, idx, 0u);
    let s0 = splitmix64(&sm_state);
    let s1 = splitmix64(&sm_state);
    let s2 = splitmix64(&sm_state);
    let s3 = splitmix64(&sm_state);

    let sum03 = add64(s0.x, s0.y, s3.x, s3.y);
    let rotated = rotl64(sum03.x, sum03.y, 23u);
    let result = add64(rotated.x, rotated.y, s0.x, s0.y);

    output[idx] = f32(result.y) / 4294967296.0;
}
"#;

const XOSHIRO256_RANDN_WGSL: &str = r#"
const PI: f32 = 3.14159265359;

struct Xoshiro256Params {
    numel: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: Xoshiro256Params;

fn rotl64(lo: u32, hi: u32, k: u32) -> vec2<u32> {
    if (k == 0u) { return vec2<u32>(lo, hi); }
    if (k < 32u) {
        return vec2<u32>((lo << k) | (hi >> (32u - k)), (hi << k) | (lo >> (32u - k)));
    }
    let k2 = k - 32u;
    return vec2<u32>((hi << k2) | (lo >> (32u - k2)), (lo << k2) | (hi >> (32u - k2)));
}

fn add64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo = a_lo + b_lo;
    return vec2<u32>(lo, a_hi + b_hi + select(0u, 1u, lo < a_lo));
}

fn xor64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>(a_lo ^ b_lo, a_hi ^ b_hi);
}

fn shl64(lo: u32, hi: u32, k: u32) -> vec2<u32> {
    if (k == 0u) { return vec2<u32>(lo, hi); }
    if (k < 32u) { return vec2<u32>(lo << k, (hi << k) | (lo >> (32u - k))); }
    return vec2<u32>(0u, lo << (k - 32u));
}

fn shr64(lo: u32, hi: u32, k: u32) -> vec2<u32> {
    if (k == 0u) { return vec2<u32>(lo, hi); }
    if (k < 32u) { return vec2<u32>((lo >> k) | (hi << (32u - k)), hi >> k); }
    return vec2<u32>(hi >> (k - 32u), 0u);
}

fn mul64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let a0 = a_lo & 0xFFFFu; let a1 = a_lo >> 16u;
    let b0 = b_lo & 0xFFFFu; let b1 = b_lo >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = p01 + p10 + (p00 >> 16u);
    let lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = p11 + (mid >> 16u) + a_lo * b_hi + a_hi * b_lo;
    return vec2<u32>(lo, hi);
}

fn splitmix64(state: ptr<function, vec2<u32>>) -> vec2<u32> {
    *state = add64((*state).x, (*state).y, 0x7F4A7C15u, 0x9E3779B9u);
    var z = *state;
    let s30 = shr64(z.x, z.y, 30u); z = xor64(z.x, z.y, s30.x, s30.y);
    z = mul64(z.x, z.y, 0x1CE4E5B9u, 0xBF58476Du);
    let s27 = shr64(z.x, z.y, 27u); z = xor64(z.x, z.y, s27.x, s27.y);
    z = mul64(z.x, z.y, 0x133111EBu, 0x94D049BBu);
    let s31 = shr64(z.x, z.y, 31u); z = xor64(z.x, z.y, s31.x, s31.y);
    return z;
}

fn xoshiro_next(s: ptr<function, array<vec2<u32>, 4>>) -> u32 {
    let sum03 = add64((*s)[0].x, (*s)[0].y, (*s)[3].x, (*s)[3].y);
    let rotated = rotl64(sum03.x, sum03.y, 23u);
    let result = add64(rotated.x, rotated.y, (*s)[0].x, (*s)[0].y);

    let t = shl64((*s)[1].x, (*s)[1].y, 17u);
    (*s)[2] = xor64((*s)[2].x, (*s)[2].y, (*s)[0].x, (*s)[0].y);
    (*s)[3] = xor64((*s)[3].x, (*s)[3].y, (*s)[1].x, (*s)[1].y);
    (*s)[1] = xor64((*s)[1].x, (*s)[1].y, (*s)[2].x, (*s)[2].y);
    (*s)[0] = xor64((*s)[0].x, (*s)[0].y, (*s)[3].x, (*s)[3].y);
    (*s)[2] = xor64((*s)[2].x, (*s)[2].y, t.x, t.y);
    (*s)[3] = rotl64((*s)[3].x, (*s)[3].y, 45u);

    return result.y;
}

fn box_muller(u1: f32, u2: f32) -> vec2<f32> {
    let r = sqrt(-2.0 * log(max(u1, 1e-10)));
    let theta = 2.0 * PI * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(256)
fn xoshiro256_randn_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 2u;
    if (base_idx >= params.numel) { return; }

    var sm_state = add64(params.seed_lo, params.seed_hi, gid.x, 0u);
    var state: array<vec2<u32>, 4>;
    state[0] = splitmix64(&sm_state);
    state[1] = splitmix64(&sm_state);
    state[2] = splitmix64(&sm_state);
    state[3] = splitmix64(&sm_state);

    let r1 = xoshiro_next(&state);
    let r2 = xoshiro_next(&state);

    let u1 = f32(r1) / 4294967296.0;
    let u2 = f32(r2) / 4294967296.0;
    let normals = box_muller(u1, u2);

    if (base_idx < params.numel) { output[base_idx] = normals.x; }
    if (base_idx + 1u < params.numel) { output[base_idx + 1u] = normals.y; }
}
"#;

/// Launch Xoshiro256++ uniform kernel
pub fn launch_xoshiro256_uniform(
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
    check_float_dtype(dtype, "xoshiro256_uniform")?;

    let name = "xoshiro256_uniform_f32";
    let module = cache.get_or_create_module(name, XOSHIRO256_UNIFORM_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("xoshiro256_uniform"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xoshiro256_uniform"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Xoshiro256++ randn kernel (Box-Muller transform)
pub fn launch_xoshiro256_randn(
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
    check_float_dtype(dtype, "xoshiro256_randn")?;

    let name = "xoshiro256_randn_f32";
    let module = cache.get_or_create_module(name, XOSHIRO256_RANDN_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("xoshiro256_randn"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xoshiro256_randn"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count((numel + 1) / 2), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
