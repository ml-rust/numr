//! WGSL shader generation for unary element-wise operations

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for unary element-wise operations
pub fn generate_unary_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Float-only operations
    let float_ops = if is_wgsl_float(dtype) {
        format!(
            r#"
@compute @workgroup_size(256)
fn sqrt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = sqrt(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn exp_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = exp(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn log_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = log(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn sin_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = sin(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn cos_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = cos(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn tan_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = tan(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn tanh_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = tanh(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn recip_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = 1.0 / unary_a[idx];
    }}
}}

@compute @workgroup_size(256)
fn floor_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = floor(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn ceil_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = ceil(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn round_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = round(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn relu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = max(unary_a[idx], 0.0);
    }}
}}

@compute @workgroup_size(256)
fn sigmoid_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = 1.0 / (1.0 + exp(-unary_a[idx]));
    }}
}}

@compute @workgroup_size(256)
fn silu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        unary_out[idx] = x / (1.0 + exp(-x));
    }}
}}

@compute @workgroup_size(256)
fn gelu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        let c = 0.7978845608028654; // sqrt(2/pi)
        unary_out[idx] = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
    }}
}}

@compute @workgroup_size(256)
fn isnan_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        // NaN != NaN in IEEE 754
        unary_out[idx] = select(0.0, 1.0, x != x);
    }}
}}

@compute @workgroup_size(256)
fn isinf_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        // Inf detection: x == x (not NaN) && x + 1 == x (overflow) && x != 0
        let is_inf = (x == x) && (x + 1.0 == x) && (x != 0.0);
        unary_out[idx] = select(0.0, 1.0, is_inf);
    }}
}}
"#,
            suffix = suffix
        )
    } else {
        // Integer types don't have these operations
        String::new()
    };

    Ok(format!(
        r#"// Auto-generated unary operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct UnaryParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> unary_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<{t}>;
@group(0) @binding(2) var<uniform> unary_params: UnaryParams;

@compute @workgroup_size(256)
fn neg_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = -{neg_prefix}unary_a[idx];
    }}
}}

@compute @workgroup_size(256)
fn abs_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = abs(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn square_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        unary_out[idx] = x * x;
    }}
}}

@compute @workgroup_size(256)
fn sign_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = sign(unary_a[idx]);
    }}
}}

{float_ops}
"#,
        t = t,
        suffix = suffix,
        neg_prefix = if dtype == DType::U32 {
            "/*u32 neg*/"
        } else {
            ""
        },
        float_ops = float_ops
    ))
}
