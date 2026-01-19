//! WGSL shader generation for binary element-wise operations

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for binary element-wise operations
pub fn generate_binary_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let float_ops = if is_wgsl_float(dtype) {
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = pow(binary_a[idx], binary_b[idx]);
    }}
}}
"#,
            suffix = suffix
        )
    } else {
        // Integer pow requires loop implementation
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        var base = binary_a[idx];
        var exp = binary_b[idx];
        var result: {t} = 1;
        // Simple integer power loop
        for (var i: {t} = 0; i < exp; i = i + 1) {{
            result = result * base;
        }}
        binary_out[idx] = result;
    }}
}}
"#,
            suffix = suffix,
            t = t
        )
    };

    Ok(format!(
        r#"// Auto-generated binary operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct BinaryParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> binary_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> binary_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> binary_out: array<{t}>;
@group(0) @binding(3) var<uniform> binary_params: BinaryParams;

@compute @workgroup_size(256)
fn add_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] + binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn sub_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] - binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn mul_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] * binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn div_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] / binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn max_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = max(binary_a[idx], binary_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn min_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = min(binary_a[idx], binary_b[idx]);
    }}
}}

{float_ops}
"#,
        t = t,
        suffix = suffix,
        float_ops = float_ops
    ))
}
