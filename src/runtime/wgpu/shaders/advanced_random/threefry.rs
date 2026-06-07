//! ThreeFry4x64-20 PRNG WGSL shaders and launchers
//!
//! Counter-based PRNG; bit-identical 64-bit stream to the CPU/CUDA reference
//! (64-bit words emulated as `vec2<u32>`). The uniform conversion matches CPU's
//! F32 path within f32 resolution; randn additionally differs by the f32 vs f64
//! Box-Muller transform.
//! Reference: Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

use wgpu::{Buffer, Queue};

use super::check_float_dtype;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache, workgroup_count};

const THREEFRY_UNIFORM_WGSL: &str = r#"
// ThreeFry4x64-20, bit-identical to the CPU/CUDA reference (64-bit words emulated
// as vec2<u32> = (lo, hi)). Only rot[0]/rot[1] of each 4-entry row are used,
// matching the CPU kernel.
const ROT0: array<u32, 8> = array(14u, 23u, 33u, 17u, 13u, 25u, 26u, 37u);
const ROT1: array<u32, 8> = array(16u, 40u, 48u, 34u, 50u, 29u, 24u, 38u);
// Skein parity 0x1BD11BDAA9FC1A22
const PARITY_LO: u32 = 0xA9FC1A22u;
const PARITY_HI: u32 = 0x1BD11BDAu;

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

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn rotl64(v: vec2<u32>, k: u32) -> vec2<u32> {
    if (k == 0u) { return v; }
    if (k < 32u) {
        return vec2<u32>((v.x << k) | (v.y >> (32u - k)), (v.y << k) | (v.x >> (32u - k)));
    }
    if (k == 32u) { return vec2<u32>(v.y, v.x); }
    let k2 = k - 32u;
    return vec2<u32>((v.y << k2) | (v.x >> (32u - k2)), (v.x << k2) | (v.y >> (32u - k2)));
}

@compute @workgroup_size(256)
fn threefry_uniform_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 4u;
    if (base_idx >= params.numel) { return; }

    let key = vec2<u32>(params.key_lo, params.key_hi);
    var ks = array<vec2<u32>, 5>(
        key,
        vec2<u32>(0u, 0u),
        vec2<u32>(0u, 0u),
        vec2<u32>(0u, 0u),
        xor64(key, vec2<u32>(PARITY_LO, PARITY_HI)),
    );

    // counter = counter_base + gid.x (64-bit), the rest of the block is zero.
    var x = array<vec2<u32>, 4>(
        add64(vec2<u32>(params.counter_lo, params.counter_hi), vec2<u32>(gid.x, 0u)),
        vec2<u32>(0u, 0u),
        vec2<u32>(0u, 0u),
        vec2<u32>(0u, 0u),
    );

    for (var r = 0u; r < 20u; r++) {
        if (r % 4u == 0u) {
            let d = r / 4u;
            x[0] = add64(x[0], ks[d % 5u]);
            x[1] = add64(x[1], ks[(d + 1u) % 5u]);
            x[2] = add64(x[2], ks[(d + 2u) % 5u]);
            x[3] = add64(add64(x[3], ks[(d + 3u) % 5u]), vec2<u32>(d, 0u));
        }
        x[0] = add64(x[0], x[1]);
        x[1] = xor64(rotl64(x[1], ROT0[r % 8u]), x[0]);
        x[2] = add64(x[2], x[3]);
        x[3] = xor64(rotl64(x[3], ROT1[r % 8u]), x[2]);
        let t = x[1]; x[1] = x[3]; x[3] = t;
    }
    // Final key injection (d = 5).
    x[0] = add64(x[0], ks[0]);
    x[1] = add64(x[1], ks[1]);
    x[2] = add64(x[2], ks[2]);
    x[3] = add64(add64(x[3], ks[3]), vec2<u32>(5u, 0u));

    for (var j = 0u; j < 4u; j++) {
        let idx = base_idx + j;
        if (idx < params.numel) {
            // Top 24 bits / 2^24 — exact in f32, matches CPU/CUDA F32 conversion.
            output[idx] = f32(x[j].y >> 8u) / 16777216.0;
        }
    }
}
"#;

const THREEFRY_RANDN_WGSL: &str = r#"
// ThreeFry4x64-20 + Box-Muller. Same 64-bit generator as the uniform variant
// and the CPU/CUDA reference (one block of 4 u64 → 4 normals). Note: the normal
// transform uses f32 sin/cos/log, so randn is not bit-identical across backends.
const ROT0: array<u32, 8> = array(14u, 23u, 33u, 17u, 13u, 25u, 26u, 37u);
const ROT1: array<u32, 8> = array(16u, 40u, 48u, 34u, 50u, 29u, 24u, 38u);
const PARITY_LO: u32 = 0xA9FC1A22u;
const PARITY_HI: u32 = 0x1BD11BDAu;
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

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn rotl64(v: vec2<u32>, k: u32) -> vec2<u32> {
    if (k == 0u) { return v; }
    if (k < 32u) {
        return vec2<u32>((v.x << k) | (v.y >> (32u - k)), (v.y << k) | (v.x >> (32u - k)));
    }
    if (k == 32u) { return vec2<u32>(v.y, v.x); }
    let k2 = k - 32u;
    return vec2<u32>((v.y << k2) | (v.x >> (32u - k2)), (v.x << k2) | (v.y >> (32u - k2)));
}

fn threefry4x64_20(counter_base: vec2<u32>, gid: u32, key: vec2<u32>) -> array<vec2<u32>, 4> {
    var ks = array<vec2<u32>, 5>(
        key, vec2<u32>(0u, 0u), vec2<u32>(0u, 0u), vec2<u32>(0u, 0u),
        xor64(key, vec2<u32>(PARITY_LO, PARITY_HI)),
    );
    var x = array<vec2<u32>, 4>(
        add64(counter_base, vec2<u32>(gid, 0u)),
        vec2<u32>(0u, 0u), vec2<u32>(0u, 0u), vec2<u32>(0u, 0u),
    );
    for (var r = 0u; r < 20u; r++) {
        if (r % 4u == 0u) {
            let d = r / 4u;
            x[0] = add64(x[0], ks[d % 5u]);
            x[1] = add64(x[1], ks[(d + 1u) % 5u]);
            x[2] = add64(x[2], ks[(d + 2u) % 5u]);
            x[3] = add64(add64(x[3], ks[(d + 3u) % 5u]), vec2<u32>(d, 0u));
        }
        x[0] = add64(x[0], x[1]);
        x[1] = xor64(rotl64(x[1], ROT0[r % 8u]), x[0]);
        x[2] = add64(x[2], x[3]);
        x[3] = xor64(rotl64(x[3], ROT1[r % 8u]), x[2]);
        let t = x[1]; x[1] = x[3]; x[3] = t;
    }
    x[0] = add64(x[0], ks[0]);
    x[1] = add64(x[1], ks[1]);
    x[2] = add64(x[2], ks[2]);
    x[3] = add64(add64(x[3], ks[3]), vec2<u32>(5u, 0u));
    return x;
}

fn to_unit(v: vec2<u32>) -> f32 {
    return f32(v.y >> 8u) / 16777216.0;
}

fn box_muller(u1: f32, u2: f32) -> vec2<f32> {
    let r = sqrt(-2.0 * log(max(u1, 1e-10)));
    let theta = 2.0 * PI * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(256)
fn threefry_randn_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * 4u;
    if (base_idx >= params.numel) { return; }
    let key = vec2<u32>(params.key_lo, params.key_hi);
    let x = threefry4x64_20(vec2<u32>(params.counter_lo, params.counter_hi), gid.x, key);
    let p0 = box_muller(to_unit(x[0]), to_unit(x[1]));
    let p1 = box_muller(to_unit(x[2]), to_unit(x[3]));
    if (base_idx < params.numel) { output[base_idx] = p0.x; }
    if (base_idx + 1u < params.numel) { output[base_idx + 1u] = p0.y; }
    if (base_idx + 2u < params.numel) { output[base_idx + 2u] = p1.x; }
    if (base_idx + 3u < params.numel) { output[base_idx + 3u] = p1.y; }
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
        pass.dispatch_workgroups(workgroup_count(numel.div_ceil(4)), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
