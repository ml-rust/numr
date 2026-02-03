//! WGSL shader generation for cumulative operations
//!
//! Generates shaders for:
//! - cumsum: cumulative sum along a dimension
//! - cumprod: cumulative product along a dimension
//! - logsumexp: numerically stable log-sum-exp reduction

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for cumulative sum operation (simple/contiguous)
pub fn generate_cumsum_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let zero = match dtype {
        DType::F32 | DType::F16 => "0.0",
        DType::I32 => "0",
        DType::U32 => "0u",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "cumsum",
            });
        }
    };

    Ok(format!(
        r#"// Auto-generated cumsum shader for {t}

struct CumsumParams {{
    scan_size: u32,
    outer_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output: array<{t}>;
@group(0) @binding(2) var<uniform> params: CumsumParams;

@compute @workgroup_size(256)
fn cumsum_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {{
        return;
    }}

    let base = outer_idx * params.scan_size;
    var acc: {t} = {zero};
    for (var i: u32 = 0u; i < params.scan_size; i = i + 1u) {{
        acc = acc + input[base + i];
        output[base + i] = acc;
    }}
}}
"#,
        t = t,
        suffix = suffix,
        zero = zero,
    ))
}

/// Generate WGSL shader for strided cumulative sum
pub fn generate_cumsum_strided_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let zero = match dtype {
        DType::F32 | DType::F16 => "0.0",
        DType::I32 => "0",
        DType::U32 => "0u",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "cumsum_strided",
            });
        }
    };

    Ok(format!(
        r#"// Auto-generated strided cumsum shader for {t}

struct CumsumStridedParams {{
    scan_size: u32,
    outer_size: u32,
    inner_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output: array<{t}>;
@group(0) @binding(2) var<uniform> params: CumsumStridedParams;

@compute @workgroup_size(256)
fn cumsum_strided_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let total_inner = params.outer_size * params.inner_size;
    if (idx >= total_inner) {{
        return;
    }}

    let outer_idx = idx / params.inner_size;
    let inner_idx = idx % params.inner_size;

    var acc: {t} = {zero};
    for (var s: u32 = 0u; s < params.scan_size; s = s + 1u) {{
        let offset = outer_idx * params.scan_size * params.inner_size + s * params.inner_size + inner_idx;
        acc = acc + input[offset];
        output[offset] = acc;
    }}
}}
"#,
        t = t,
        suffix = suffix,
        zero = zero,
    ))
}

/// Generate WGSL shader for cumulative product operation (simple/contiguous)
pub fn generate_cumprod_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let one = match dtype {
        DType::F32 | DType::F16 => "1.0",
        DType::I32 => "1",
        DType::U32 => "1u",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "cumprod",
            });
        }
    };

    Ok(format!(
        r#"// Auto-generated cumprod shader for {t}

struct CumprodParams {{
    scan_size: u32,
    outer_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output: array<{t}>;
@group(0) @binding(2) var<uniform> params: CumprodParams;

@compute @workgroup_size(256)
fn cumprod_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {{
        return;
    }}

    let base = outer_idx * params.scan_size;
    var acc: {t} = {one};
    for (var i: u32 = 0u; i < params.scan_size; i = i + 1u) {{
        acc = acc * input[base + i];
        output[base + i] = acc;
    }}
}}
"#,
        t = t,
        suffix = suffix,
        one = one,
    ))
}

/// Generate WGSL shader for strided cumulative product
pub fn generate_cumprod_strided_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let one = match dtype {
        DType::F32 | DType::F16 => "1.0",
        DType::I32 => "1",
        DType::U32 => "1u",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "cumprod_strided",
            });
        }
    };

    Ok(format!(
        r#"// Auto-generated strided cumprod shader for {t}

struct CumprodStridedParams {{
    scan_size: u32,
    outer_size: u32,
    inner_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output: array<{t}>;
@group(0) @binding(2) var<uniform> params: CumprodStridedParams;

@compute @workgroup_size(256)
fn cumprod_strided_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let total_inner = params.outer_size * params.inner_size;
    if (idx >= total_inner) {{
        return;
    }}

    let outer_idx = idx / params.inner_size;
    let inner_idx = idx % params.inner_size;

    var acc: {t} = {one};
    for (var s: u32 = 0u; s < params.scan_size; s = s + 1u) {{
        let offset = outer_idx * params.scan_size * params.inner_size + s * params.inner_size + inner_idx;
        acc = acc * input[offset];
        output[offset] = acc;
    }}
}}
"#,
        t = t,
        suffix = suffix,
        one = one,
    ))
}

/// Generate WGSL shader for log-sum-exp reduction (simple/contiguous)
///
/// Computes log(sum(exp(x))) in a numerically stable way:
/// logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
pub fn generate_logsumexp_shader(dtype: DType) -> Result<String> {
    // logsumexp only supported for float types
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "logsumexp",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let min_val = match dtype {
        DType::F32 => "-3.402823e+38",
        DType::F16 => "-65504.0",
        _ => "-3.402823e+38",
    };

    Ok(format!(
        r#"// Auto-generated logsumexp shader for {t}

struct LogsumexpParams {{
    reduce_size: u32,
    outer_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output: array<{t}>;
@group(0) @binding(2) var<uniform> params: LogsumexpParams;

@compute @workgroup_size(256)
fn logsumexp_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {{
        return;
    }}

    let base = outer_idx * params.reduce_size;

    // Step 1: Find max value
    var max_val: {t} = {min_val};
    for (var i: u32 = 0u; i < params.reduce_size; i = i + 1u) {{
        let val = input[base + i];
        max_val = max(max_val, val);
    }}

    // Step 2: Compute sum(exp(x - max))
    var sum_exp: {t} = 0.0;
    for (var i: u32 = 0u; i < params.reduce_size; i = i + 1u) {{
        sum_exp = sum_exp + exp(input[base + i] - max_val);
    }}

    // Step 3: Result = max + log(sum)
    output[outer_idx] = max_val + log(sum_exp);
}}
"#,
        t = t,
        suffix = suffix,
        min_val = min_val,
    ))
}

/// Generate WGSL shader for strided log-sum-exp reduction
pub fn generate_logsumexp_strided_shader(dtype: DType) -> Result<String> {
    // logsumexp only supported for float types
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "logsumexp_strided",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let min_val = match dtype {
        DType::F32 => "-3.402823e+38",
        DType::F16 => "-65504.0",
        _ => "-3.402823e+38",
    };

    Ok(format!(
        r#"// Auto-generated strided logsumexp shader for {t}

struct LogsumexpStridedParams {{
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output: array<{t}>;
@group(0) @binding(2) var<uniform> params: LogsumexpStridedParams;

@compute @workgroup_size(256)
fn logsumexp_strided_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let total_inner = params.outer_size * params.inner_size;
    if (idx >= total_inner) {{
        return;
    }}

    let outer_idx = idx / params.inner_size;
    let inner_idx = idx % params.inner_size;

    // Step 1: Find max value along reduce dimension
    let first_offset = outer_idx * params.reduce_size * params.inner_size + inner_idx;
    var max_val: {t} = {min_val};
    for (var r: u32 = 0u; r < params.reduce_size; r = r + 1u) {{
        let offset = outer_idx * params.reduce_size * params.inner_size + r * params.inner_size + inner_idx;
        max_val = max(max_val, input[offset]);
    }}

    // Step 2: Compute sum(exp(x - max))
    var sum_exp: {t} = 0.0;
    for (var r: u32 = 0u; r < params.reduce_size; r = r + 1u) {{
        let offset = outer_idx * params.reduce_size * params.inner_size + r * params.inner_size + inner_idx;
        sum_exp = sum_exp + exp(input[offset] - max_val);
    }}

    // Step 3: Write result
    output[outer_idx * params.inner_size + inner_idx] = max_val + log(sum_exp);
}}
"#,
        t = t,
        suffix = suffix,
        min_val = min_val,
    ))
}
