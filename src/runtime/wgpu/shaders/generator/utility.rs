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
