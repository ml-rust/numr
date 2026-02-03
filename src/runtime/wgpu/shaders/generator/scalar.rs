//! WGSL shader generation for scalar element-wise operations and fill operations

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for scalar element-wise operations
pub fn generate_scalar_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let float_ops = if is_wgsl_float(dtype) {
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = pow(scalar_a[idx], {t}(scalar_params.scalar));
    }}
}}

// Leaky ReLU: max(negative_slope * x, x)
@compute @workgroup_size(256)
fn leaky_relu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        let x = scalar_a[idx];
        let slope = {t}(scalar_params.scalar);
        scalar_out[idx] = max(slope * x, x);
    }}
}}

// ELU: x if x > 0, else alpha * (exp(x) - 1)
@compute @workgroup_size(256)
fn elu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        let x = scalar_a[idx];
        let alpha = {t}(scalar_params.scalar);
        scalar_out[idx] = select(alpha * (exp(x) - 1.0), x, x > 0.0);
    }}
}}
"#,
            suffix = suffix,
            t = t
        )
    } else {
        // Integer pow_scalar
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        var base = scalar_a[idx];
        var exp = {t}(scalar_params.scalar);
        var result: {t} = 1;
        for (var i: {t} = 0; i < exp; i = i + 1) {{
            result = result * base;
        }}
        scalar_out[idx] = result;
    }}
}}
"#,
            suffix = suffix,
            t = t
        )
    };

    Ok(format!(
        r#"// Auto-generated scalar operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ScalarParams {{
    numel: u32,
    scalar: f32,  // Always f32 for uniform, cast in shader
}}

@group(0) @binding(0) var<storage, read_write> scalar_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> scalar_out: array<{t}>;
@group(0) @binding(2) var<uniform> scalar_params: ScalarParams;

@compute @workgroup_size(256)
fn add_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] + {t}(scalar_params.scalar);
    }}
}}

@compute @workgroup_size(256)
fn sub_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] - {t}(scalar_params.scalar);
    }}
}}

@compute @workgroup_size(256)
fn mul_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] * {t}(scalar_params.scalar);
    }}
}}

@compute @workgroup_size(256)
fn div_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] / {t}(scalar_params.scalar);
    }}
}}

{float_ops}
"#,
        t = t,
        suffix = suffix,
        float_ops = float_ops
    ))
}

/// Generate WGSL shader for fill operation (set all elements to a constant value)
pub fn generate_fill_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated fill operation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct FillParams {{
    numel: u32,
    value: f32,  // Always f32 for uniform, cast in shader
}}

@group(0) @binding(0) var<storage, read_write> fill_out: array<{t}>;
@group(0) @binding(1) var<uniform> fill_params: FillParams;

@compute @workgroup_size(256)
fn fill_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < fill_params.numel) {{
        fill_out[idx] = {t}(fill_params.value);
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}
