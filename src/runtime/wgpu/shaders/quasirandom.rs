//! Quasi-random sequence generation WGSL kernel launchers
//!
//! Provides launchers for quasi-random sequences:
//! - Sobol sequence (Gray code-based low-discrepancy)
//! - Halton sequence (van der Corput in prime bases)
//! - Latin Hypercube Sampling (stratified random sampling)

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Sobol Sequence
// ============================================================================

const SOBOL_WGSL: &str = r#"
// Direction numbers for Sobol sequence (6 dimensions only)
// Based on Joe & Kuo (2008) for primitive polynomials over GF(2)
//
// LIMITATION: Only 6 dimensions have direction numbers.
// Dimensions 7+ fall back to van der Corput (lower quality).
const SOBOL_DIRECTION_NUMBERS: array<array<u32, 30>, 6> = array(
    array(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
    array(1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
    array(1u, 3u, 7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
    array(1u, 1u, 5u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
    array(1u, 3u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
    array(1u, 1u, 3u, 7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
);

struct SobolParams {
    n_points: u32,
    dimension: u32,
    skip: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: SobolParams;

// van der Corput for dimensions > 6
fn van_der_corput_f32(index: u32, base: u32) -> f32 {
    var result = 0.0;
    var f = 1.0 / f32(base);
    var i = index;
    while (i > 0u) {
        result += f * f32(i % base);
        i = i / base;
        f = f / f32(base);
    }
    return result;
}

@compute @workgroup_size(256)
fn sobol_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) { return; }

    let point_index = params.skip + idx;

    for (var d = 0u; d < params.dimension; d++) {
        var value: f32;

        if (d < 6u) {
            // Use Sobol direction numbers
            var v: array<u32, 32>;
            for (var i = 0u; i < 30u; i++) {
                v[i] = SOBOL_DIRECTION_NUMBERS[d][i] << (31u - i);
            }
            v[30] = 0u;
            v[31] = 0u;

            // Gray code
            let gray = point_index ^ (point_index >> 1u);

            // Compute Sobol point
            var x = 0u;
            for (var bit = 0u; bit < 32u; bit++) {
                if ((gray & (1u << bit)) != 0u) {
                    x = x ^ v[bit];
                }
            }

            // Convert to float in [0, 1)
            value = f32(x) / 4294967296.0; // 2^32
        } else {
            // Fallback to van der Corput
            value = van_der_corput_f32(point_index, 2u + d);
        }

        output[idx * params.dimension + d] = value;
    }
}
"#;

pub fn launch_sobol(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "sobol")?;

    let name = "sobol_f32";
    let module = cache.get_or_create_module(name, SOBOL_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sobol"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sobol"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Dispatch based on n_points, not total_elements
        // Since each thread generates all dimensions for one point
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Halton Sequence
// ============================================================================

const HALTON_WGSL: &str = r#"
// First 100 prime numbers
const PRIMES: array<u32, 100> = array(
    2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u, 59u, 61u, 67u, 71u,
    73u, 79u, 83u, 89u, 97u, 101u, 103u, 107u, 109u, 113u, 127u, 131u, 137u, 139u, 149u, 151u, 157u, 163u, 167u, 173u,
    179u, 181u, 191u, 193u, 197u, 199u, 211u, 223u, 227u, 229u, 233u, 239u, 241u, 251u, 257u, 263u, 269u, 271u, 277u, 281u,
    283u, 293u, 307u, 311u, 313u, 317u, 331u, 337u, 347u, 349u, 353u, 359u, 367u, 373u, 379u, 383u, 389u, 397u, 401u, 409u,
    419u, 421u, 431u, 433u, 439u, 443u, 449u, 457u, 461u, 463u, 467u, 479u, 487u, 491u, 499u, 503u, 509u, 521u, 523u, 541u
);

struct HaltonParams {
    n_points: u32,
    dimension: u32,
    skip: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: HaltonParams;

fn van_der_corput_f32(index: u32, base: u32) -> f32 {
    var result = 0.0;
    var f = 1.0 / f32(base);
    var i = index;
    while (i > 0u) {
        result += f * f32(i % base);
        i = i / base;
        f = f / f32(base);
    }
    return result;
}

@compute @workgroup_size(256)
fn halton_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) { return; }

    let point_index = params.skip + idx;

    for (var d = 0u; d < params.dimension; d++) {
        let base = PRIMES[d];
        output[idx * params.dimension + d] = van_der_corput_f32(point_index, base);
    }
}
"#;

pub fn launch_halton(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "halton")?;

    let name = "halton_f32";
    let module = cache.get_or_create_module(name, HALTON_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("halton"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("halton"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Latin Hypercube Sampling
// ============================================================================

const LATIN_HYPERCUBE_WGSL: &str = r#"
struct LatinHypercubeParams {
    n_samples: u32,
    dimension: u32,
    seed: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: LatinHypercubeParams;

// xorshift32 RNG
fn xorshift32(state_ptr: ptr<function, u32>) -> u32 {
    var x = *state_ptr;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state_ptr = x;
    return x;
}

fn uniform_f32(state_ptr: ptr<function, u32>) -> f32 {
    return f32(xorshift32(state_ptr)) / 4294967296.0;
}

@compute @workgroup_size(256)
fn latin_hypercube_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each workgroup handles one dimension
    let dim = gid.x;
    if (dim >= params.dimension) { return; }

    // Thread-local RNG state
    var rng_state = params.seed + dim * 1234567u + gid.y * 987654u;

    // Each thread in the workgroup processes multiple samples
    let samples_per_thread = (params.n_samples + 255u) / 256u;
    let start_sample = gid.y * samples_per_thread;
    let end_sample = min(start_sample + samples_per_thread, params.n_samples);

    for (var i = start_sample; i < end_sample; i++) {
        // Simple linear congruential shuffle approximation
        // For proper LHS, we'd need shared memory and synchronization
        var interval = (i + (dim * 7919u) % params.n_samples) % params.n_samples;

        let lower = f32(interval) / f32(params.n_samples);
        let upper = f32(interval + 1u) / f32(params.n_samples);
        let random_offset = uniform_f32(&rng_state);

        output[i * params.dimension + dim] = lower + random_offset * (upper - lower);
    }
}
"#;

pub fn launch_latin_hypercube(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    total_workgroups: usize,
    dtype: DType,
) -> Result<()> {
    if total_workgroups == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "latin_hypercube")?;

    let name = "latin_hypercube_f32";
    let module = cache.get_or_create_module(name, LATIN_HYPERCUBE_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("latin_hypercube"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("latin_hypercube"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per dimension
        pass.dispatch_workgroups(total_workgroups as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
