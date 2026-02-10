//! WGSL shader generation for utility operations: arange, linspace, eye, rand, randn, randint

use super::common::{dtype_suffix, is_wgsl_float, is_wgsl_int, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for arange operation
pub fn generate_arange_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated arange operation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ArangeParams {{
    numel: u32,
    start: f32,
    step: f32,
}}

@group(0) @binding(0) var<storage, read_write> arange_out: array<{t}>;
@group(0) @binding(1) var<uniform> arange_params: ArangeParams;

@compute @workgroup_size(256)
fn arange_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < arange_params.numel) {{
        let value = arange_params.start + arange_params.step * f32(idx);
        arange_out[idx] = {t}(value);
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for linspace operation
pub fn generate_linspace_shader(dtype: DType) -> Result<String> {
    // linspace only makes sense for float types
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "linspace",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated linspace operation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct LinspaceParams {{
    steps: u32,
    start: f32,
    stop: f32,
}}

@group(0) @binding(0) var<storage, read_write> linspace_out: array<{t}>;
@group(0) @binding(1) var<uniform> linspace_params: LinspaceParams;

@compute @workgroup_size(256)
fn linspace_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < linspace_params.steps) {{
        let t_val = f32(idx) / f32(linspace_params.steps - 1u);
        let value = linspace_params.start + (linspace_params.stop - linspace_params.start) * t_val;
        linspace_out[idx] = {t}(value);
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for eye operation (identity matrix)
pub fn generate_eye_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Determine the correct "one" and "zero" values based on type
    let (one_val, zero_val) = if is_wgsl_float(dtype) {
        ("1.0", "0.0")
    } else {
        ("1", "0")
    };

    Ok(format!(
        r#"// Auto-generated eye (identity matrix) operation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct EyeParams {{
    n: u32,      // rows
    m: u32,      // cols
    numel: u32,  // n * m
}}

@group(0) @binding(0) var<storage, read_write> eye_out: array<{t}>;
@group(0) @binding(1) var<uniform> eye_params: EyeParams;

@compute @workgroup_size(256)
fn eye_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < eye_params.numel) {{
        let row = idx / eye_params.m;
        let col = idx % eye_params.m;
        if (row == col) {{
            eye_out[idx] = {t}({one_val});
        }} else {{
            eye_out[idx] = {t}({zero_val});
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix,
        one_val = one_val,
        zero_val = zero_val
    ))
}

// ============================================================================
// Random Number Generation Shaders
// ============================================================================

/// WGSL implementation of PCG hash for random number generation
/// This produces high-quality random numbers suitable for most applications.
const PCG_HASH_WGSL: &str = r#"
// PCG hash function for random number generation
// Based on PCG Random Number Generation by Melissa O'Neill
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Initialize PCG state from seed and index
fn pcg_init(seed: u32, idx: u32) -> u32 {
    return pcg_hash(seed ^ pcg_hash(idx));
}

// Generate uniform float in [0, 1)
fn pcg_uniform(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / 4294967296.0;  // Divide by 2^32
}

// Box-Muller transform for normal distribution
// Generates one normal value, requires two uniform values
fn box_muller(u1: f32, u2: f32) -> f32 {
    let u1_safe = max(u1, 0.0000001);  // Avoid log(0)
    let r = sqrt(-2.0 * log(u1_safe));
    let theta = 6.28318530718 * u2;  // 2 * PI
    return r * cos(theta);
}
"#;

/// Generate WGSL shader for rand operation (uniform [0, 1))
pub fn generate_rand_shader(dtype: DType) -> Result<String> {
    // rand only makes sense for float types
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "rand" });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated rand operation for {t}
{pcg_hash}
const WORKGROUP_SIZE: u32 = 256u;

struct RandParams {{
    numel: u32,
    seed: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> rand_out: array<{t}>;
@group(0) @binding(1) var<uniform> rand_params: RandParams;

@compute @workgroup_size(256)
fn rand_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < rand_params.numel) {{
        var state = pcg_init(rand_params.seed, idx);
        let value = pcg_uniform(&state);
        rand_out[idx] = {t}(value);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        pcg_hash = PCG_HASH_WGSL
    ))
}

/// Generate WGSL shader for randn operation (standard normal N(0, 1))
pub fn generate_randn_shader(dtype: DType) -> Result<String> {
    // randn only makes sense for float types
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "randn" });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated randn operation for {t}
{pcg_hash}
const WORKGROUP_SIZE: u32 = 256u;

struct RandnParams {{
    numel: u32,
    seed: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> randn_out: array<{t}>;
@group(0) @binding(1) var<uniform> randn_params: RandnParams;

@compute @workgroup_size(256)
fn randn_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < randn_params.numel) {{
        // Use two uniform random values for Box-Muller
        var state = pcg_init(randn_params.seed, idx);
        let u1 = pcg_uniform(&state);
        let u2 = pcg_uniform(&state);
        let value = box_muller(u1, u2);
        randn_out[idx] = {t}(value);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        pcg_hash = PCG_HASH_WGSL
    ))
}

/// Generate WGSL shader for randint operation (uniform integers in [low, high))
///
/// For signed integers (I32): low is stored as i32, arithmetic done in i32
/// For unsigned integers (U32): low is stored as u32, arithmetic done in u32
///
/// This ensures correct handling of negative bounds for signed types and
/// avoids overflow issues with large unsigned ranges.
pub fn generate_randint_shader(dtype: DType) -> Result<String> {
    // randint only makes sense for integer types
    if !is_wgsl_int(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "randint",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Generate completely separate shaders for signed vs unsigned
    // This avoids type casting issues and overflow problems
    let is_signed = matches!(dtype, DType::I32);

    if is_signed {
        // Signed integer version: low stored as i32, arithmetic in i32
        Ok(format!(
            r#"// Auto-generated randint operation for {t} (signed)
{pcg_hash}
const WORKGROUP_SIZE: u32 = 256u;

struct RandintParams {{
    numel: u32,
    low: i32,           // Low bound as signed integer
    range: u32,         // high - low (always positive, fits in u32)
    seed: u32,
}}

@group(0) @binding(0) var<storage, read_write> randint_out: array<{t}>;
@group(0) @binding(1) var<uniform> randint_params: RandintParams;

@compute @workgroup_size(256)
fn randint_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < randint_params.numel) {{
        var state = pcg_init(randint_params.seed, idx);
        let r = pcg_hash(state);
        // Compute offset in unsigned space, then add to signed low
        let offset = r % randint_params.range;
        // Safe: offset < range, so low + offset won't overflow if inputs are valid
        randint_out[idx] = randint_params.low + i32(offset);
    }}
}}
"#,
            t = t,
            suffix = suffix,
            pcg_hash = PCG_HASH_WGSL
        ))
    } else {
        // Unsigned integer version: all arithmetic in u32
        Ok(format!(
            r#"// Auto-generated randint operation for {t} (unsigned)
{pcg_hash}
const WORKGROUP_SIZE: u32 = 256u;

struct RandintParams {{
    numel: u32,
    low: u32,           // Low bound as unsigned integer
    range: u32,         // high - low
    seed: u32,
}}

@group(0) @binding(0) var<storage, read_write> randint_out: array<{t}>;
@group(0) @binding(1) var<uniform> randint_params: RandintParams;

@compute @workgroup_size(256)
fn randint_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < randint_params.numel) {{
        var state = pcg_init(randint_params.seed, idx);
        let r = pcg_hash(state);
        // Pure unsigned arithmetic - no overflow for valid inputs
        let offset = r % randint_params.range;
        randint_out[idx] = randint_params.low + offset;
    }}
}}
"#,
            t = t,
            suffix = suffix,
            pcg_hash = PCG_HASH_WGSL
        ))
    }
}

/// Generate WGSL shader for multinomial sampling with replacement
///
/// Uses inverse transform sampling (CDF method):
/// 1. Compute cumulative sum of normalized probabilities
/// 2. For each sample, draw uniform random u ∈ `[0, 1)`
/// 3. Find smallest index i where `CDF[i]` ≥ u (linear search)
pub fn generate_multinomial_with_replacement_shader() -> Result<String> {
    Ok(format!(
        r#"// Auto-generated multinomial_with_replacement operation for f32
{pcg_hash}
const WORKGROUP_SIZE: u32 = 256u;

struct MultinomialParams {{
    num_distributions: u32,
    num_categories: u32,
    num_samples: u32,
    seed: u32,
}}

@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> multinomial_out: array<i32>;
@group(0) @binding(2) var<uniform> multinomial_params: MultinomialParams;

@compute @workgroup_size(256)
fn multinomial_with_replacement_f32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = multinomial_params.num_distributions * multinomial_params.num_samples;
    if (idx >= total) {{
        return;
    }}

    let dist = idx / multinomial_params.num_samples;
    let sample = idx % multinomial_params.num_samples;

    // Initialize RNG for this thread
    var state = pcg_init(multinomial_params.seed, idx);

    // Get pointer to this distribution's probabilities
    let prob_offset = dist * multinomial_params.num_categories;

    // Compute sum of probabilities for normalization
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {{
        sum = sum + probs[prob_offset + i];
    }}

    // Generate uniform random value
    let u = pcg_uniform(&state);

    // Linear search using CDF (on-the-fly computation)
    // Find smallest index where cumsum/sum >= u
    var cumsum: f32 = 0.0;
    var result: u32 = multinomial_params.num_categories - 1u;  // Default to last category
    for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {{
        cumsum = cumsum + probs[prob_offset + i];
        if (cumsum / sum >= u) {{
            result = i;
            break;
        }}
    }}

    multinomial_out[dist * multinomial_params.num_samples + sample] = i32(result);
}}
"#,
        pcg_hash = PCG_HASH_WGSL
    ))
}

/// Generate WGSL shader for multinomial sampling without replacement
///
/// Uses sequential sampling within each distribution. Each workgroup handles
/// one distribution. Selected categories are zeroed out in shared memory to
/// prevent resampling.
///
/// Note: This kernel is less parallelizable than with-replacement because
/// samples within a distribution must be sequential to ensure uniqueness.
pub fn generate_multinomial_without_replacement_shader() -> Result<String> {
    Ok(format!(
        r#"// Auto-generated multinomial_without_replacement operation for f32
{pcg_hash}
const WORKGROUP_SIZE: u32 = 256u;
const MAX_CATEGORIES: u32 = 1024u;  // Maximum supported categories

struct MultinomialParams {{
    num_distributions: u32,
    num_categories: u32,
    num_samples: u32,
    seed: u32,
}}

@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> multinomial_out: array<i32>;
@group(0) @binding(2) var<uniform> multinomial_params: MultinomialParams;

var<workgroup> shared_probs: array<f32, MAX_CATEGORIES>;

@compute @workgroup_size(256)
fn multinomial_without_replacement_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let dist = gid.x / WORKGROUP_SIZE;
    if (dist >= multinomial_params.num_distributions) {{
        return;
    }}

    // Copy probabilities to shared memory (each thread copies some elements)
    let prob_offset = dist * multinomial_params.num_categories;
    let elements_per_thread = (multinomial_params.num_categories + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {{
        let idx = lid.x * elements_per_thread + i;
        if (idx < multinomial_params.num_categories) {{
            shared_probs[idx] = probs[prob_offset + idx];
        }}
    }}

    workgroupBarrier();

    // Only thread 0 does the sequential sampling
    if (lid.x != 0u) {{
        return;
    }}

    // Initialize RNG
    var state = pcg_init(multinomial_params.seed, dist);

    // Sample without replacement
    for (var s: u32 = 0u; s < multinomial_params.num_samples; s = s + 1u) {{
        // Compute sum of remaining probabilities
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {{
            sum = sum + shared_probs[i];
        }}

        // Generate uniform random value
        let u = pcg_uniform(&state);

        // Linear search using CDF
        var cumsum: f32 = 0.0;
        var result: u32 = multinomial_params.num_categories - 1u;
        for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {{
            cumsum = cumsum + shared_probs[i];
            if (cumsum / sum >= u) {{
                result = i;
                break;
            }}
        }}

        multinomial_out[dist * multinomial_params.num_samples + s] = i32(result);

        // Zero out selected category
        shared_probs[result] = 0.0;
    }}
}}
"#,
        pcg_hash = PCG_HASH_WGSL
    ))
}
