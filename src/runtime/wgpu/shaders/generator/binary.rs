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

@compute @workgroup_size(256)
fn atan2_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = atan2(binary_a[idx], binary_b[idx]);
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

/// Generate WGSL shader for broadcast binary element-wise operations.
///
/// This shader handles tensors with different shapes that need broadcasting.
/// Strides are passed as storage buffers with 0 for broadcast dimensions.
pub fn generate_broadcast_binary_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let float_ops = if is_wgsl_float(dtype) {
        format!(
            r#"
@compute @workgroup_size(256)
fn broadcast_pow_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {{
        return;
    }}

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {{
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }}

    broadcast_out[idx] = pow(broadcast_a[a_offset], broadcast_b[b_offset]);
}}
"#,
            suffix = suffix
        )
    } else {
        String::new() // Integer pow not commonly needed for broadcast
    };

    // Define all broadcast binary operations
    let ops = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")];

    let mut op_shaders = String::new();
    for (op_name, op_sym) in ops.iter() {
        op_shaders.push_str(&format!(
            r#"
@compute @workgroup_size(256)
fn broadcast_{op_name}_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {{
        return;
    }}

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {{
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }}

    broadcast_out[idx] = broadcast_a[a_offset] {op_sym} broadcast_b[b_offset];
}}
"#,
            op_name = op_name,
            suffix = suffix,
            op_sym = op_sym,
        ));
    }

    // max/min use built-in functions
    op_shaders.push_str(&format!(
        r#"
@compute @workgroup_size(256)
fn broadcast_max_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {{
        return;
    }}

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {{
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }}

    broadcast_out[idx] = max(broadcast_a[a_offset], broadcast_b[b_offset]);
}}

@compute @workgroup_size(256)
fn broadcast_min_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {{
        return;
    }}

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {{
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }}

    broadcast_out[idx] = min(broadcast_a[a_offset], broadcast_b[b_offset]);
}}
"#,
        suffix = suffix
    ));

    Ok(format!(
        r#"// Auto-generated broadcast binary operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct BroadcastBinaryParams {{
    numel: u32,
    ndim: u32,
}}

@group(0) @binding(0) var<storage, read_write> broadcast_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> broadcast_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> broadcast_out: array<{t}>;
@group(0) @binding(3) var<storage, read_write> broadcast_a_strides: array<u32>;
@group(0) @binding(4) var<storage, read_write> broadcast_b_strides: array<u32>;
@group(0) @binding(5) var<storage, read_write> broadcast_out_strides: array<u32>;
@group(0) @binding(6) var<uniform> broadcast_params: BroadcastBinaryParams;

{op_shaders}
{float_ops}
"#,
        t = t,
        op_shaders = op_shaders,
        float_ops = float_ops
    ))
}
