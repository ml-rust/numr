//! WGSL shader generation for utility operations: arange, linspace, eye

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
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
