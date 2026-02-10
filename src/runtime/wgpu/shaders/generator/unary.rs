//! WGSL shader generation for unary element-wise operations

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for unary element-wise operations
pub fn generate_unary_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Signed-only operations (F32, I32 - not U32)
    let signed_ops = if dtype != DType::U32 {
        format!(
            r#"
@compute @workgroup_size(256)
fn neg_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = -unary_a[idx];
    }}
}}
"#,
            suffix = suffix
        )
    } else {
        // U32 doesn't support negation
        String::new()
    };

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
fn atan_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = atan(unary_a[idx]);
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
        // Match CPU/CUDA behavior: ties round away from zero.
        let x = unary_a[idx];
        unary_out[idx] = select(ceil(x - 0.5), floor(x + 0.5), x >= 0.0);
    }}
}}

@compute @workgroup_size(256)
fn trunc_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = trunc(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn rsqrt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = inverseSqrt(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn cbrt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        // cbrt(x) = sign(x) * pow(abs(x), 1/3)
        unary_out[idx] = sign(x) * pow(abs(x), 1.0 / 3.0);
    }}
}}

@compute @workgroup_size(256)
fn exp2_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = exp2(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn expm1_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = exp(unary_a[idx]) - 1.0;
    }}
}}

@compute @workgroup_size(256)
fn log2_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = log2(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn log10_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        // log10(x) = log(x) / log(10) = log(x) * 0.4342944819032518
        unary_out[idx] = log(unary_a[idx]) * 0.4342944819032518;
    }}
}}

@compute @workgroup_size(256)
fn log1p_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = log(1.0 + unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn asin_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        let y = sqrt(max(0.0, 1.0 - x * x));
        unary_out[idx] = atan2(x, y);
    }}
}}

@compute @workgroup_size(256)
fn acos_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        let y = sqrt(max(0.0, 1.0 - x * x));
        unary_out[idx] = atan2(y, x);
    }}
}}

@compute @workgroup_size(256)
fn sinh_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = sinh(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn cosh_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = cosh(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn asinh_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = asinh(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn acosh_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = acosh(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn atanh_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = atanh(unary_a[idx]);
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
        let bits = bitcast<u32>(f32(x));
        let exp = bits & 0x7f800000u;
        let mant = bits & 0x007fffffu;
        let is_nan = (exp == 0x7f800000u) && (mant != 0u);
        unary_out[idx] = select(0.0, 1.0, is_nan);
    }}
}}

@compute @workgroup_size(256)
fn isinf_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        let bits = bitcast<u32>(f32(x));
        let exp = bits & 0x7f800000u;
        let mant = bits & 0x007fffffu;
        let is_inf = (exp == 0x7f800000u) && (mant == 0u);
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

{signed_ops}
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
        signed_ops = signed_ops,
        float_ops = float_ops
    ))
}
