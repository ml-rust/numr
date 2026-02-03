//! WGSL shader generation for convolution operations

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for conv1d operation.
///
/// Input layout: (N, C_in, L)
/// Weight layout: (C_out, C_in/groups, K)
/// Output layout: (N, C_out, L_out)
pub fn generate_conv1d_shader(dtype: DType) -> Result<String> {
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "conv1d",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let zero = if dtype == DType::F16 { "0.0h" } else { "0.0" };

    Ok(format!(
        r#"// Auto-generated conv1d shader for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct Conv1dParams {{
    batch: u32,
    c_in: u32,
    length: u32,
    c_out: u32,
    kernel_size: u32,
    output_length: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    has_bias: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> conv1d_input: array<{t}>;
@group(0) @binding(1) var<storage, read> conv1d_weight: array<{t}>;
@group(0) @binding(2) var<storage, read> conv1d_bias: array<{t}>;
@group(0) @binding(3) var<storage, read_write> conv1d_output: array<{t}>;
@group(0) @binding(4) var<uniform> conv1d_params: Conv1dParams;

@compute @workgroup_size(256)
fn conv1d_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = conv1d_params.batch * conv1d_params.c_out * conv1d_params.output_length;
    if (idx >= total) {{ return; }}

    let ox = idx % conv1d_params.output_length;
    let oc = (idx / conv1d_params.output_length) % conv1d_params.c_out;
    let b = idx / (conv1d_params.c_out * conv1d_params.output_length);

    let c_in_per_group = conv1d_params.c_in / conv1d_params.groups;
    let c_out_per_group = conv1d_params.c_out / conv1d_params.groups;
    let g = oc / c_out_per_group;
    let c_in_start = g * c_in_per_group;

    var sum: {t} = {zero};

    for (var ic: u32 = 0u; ic < c_in_per_group; ic = ic + 1u) {{
        let c_in_idx = c_in_start + ic;

        for (var kx: u32 = 0u; kx < conv1d_params.kernel_size; kx = kx + 1u) {{
            let ix_signed = i32(ox * conv1d_params.stride + kx * conv1d_params.dilation) - i32(conv1d_params.padding);

            if (ix_signed >= 0 && u32(ix_signed) < conv1d_params.length) {{
                let ix = u32(ix_signed);
                let input_idx = b * conv1d_params.c_in * conv1d_params.length + c_in_idx * conv1d_params.length + ix;
                let weight_idx = oc * c_in_per_group * conv1d_params.kernel_size + ic * conv1d_params.kernel_size + kx;
                sum = sum + conv1d_input[input_idx] * conv1d_weight[weight_idx];
            }}
        }}
    }}

    if (conv1d_params.has_bias != 0u) {{
        sum = sum + conv1d_bias[oc];
    }}

    conv1d_output[idx] = sum;
}}
"#,
        t = t,
        suffix = suffix,
        zero = zero,
    ))
}

/// Generate WGSL shader for conv2d operation.
///
/// Input layout: (N, C_in, H, W)
/// Weight layout: (C_out, C_in/groups, K_h, K_w)
/// Output layout: (N, C_out, H_out, W_out)
pub fn generate_conv2d_shader(dtype: DType) -> Result<String> {
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "conv2d",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let zero = if dtype == DType::F16 { "0.0h" } else { "0.0" };

    Ok(format!(
        r#"// Auto-generated conv2d shader for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct Conv2dParams {{
    batch: u32,
    c_in: u32,
    height: u32,
    width: u32,
    c_out: u32,
    kernel_h: u32,
    kernel_w: u32,
    output_h: u32,
    output_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
    has_bias: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> conv2d_input: array<{t}>;
@group(0) @binding(1) var<storage, read> conv2d_weight: array<{t}>;
@group(0) @binding(2) var<storage, read> conv2d_bias: array<{t}>;
@group(0) @binding(3) var<storage, read_write> conv2d_output: array<{t}>;
@group(0) @binding(4) var<uniform> conv2d_params: Conv2dParams;

@compute @workgroup_size(256)
fn conv2d_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = conv2d_params.batch * conv2d_params.c_out * conv2d_params.output_h * conv2d_params.output_w;
    if (idx >= total) {{ return; }}

    let ox = idx % conv2d_params.output_w;
    let oy = (idx / conv2d_params.output_w) % conv2d_params.output_h;
    let oc = (idx / (conv2d_params.output_w * conv2d_params.output_h)) % conv2d_params.c_out;
    let b = idx / (conv2d_params.c_out * conv2d_params.output_h * conv2d_params.output_w);

    let c_in_per_group = conv2d_params.c_in / conv2d_params.groups;
    let c_out_per_group = conv2d_params.c_out / conv2d_params.groups;
    let g = oc / c_out_per_group;
    let c_in_start = g * c_in_per_group;

    var sum: {t} = {zero};

    for (var ic: u32 = 0u; ic < c_in_per_group; ic = ic + 1u) {{
        let c_in_idx = c_in_start + ic;

        for (var ky: u32 = 0u; ky < conv2d_params.kernel_h; ky = ky + 1u) {{
            for (var kx: u32 = 0u; kx < conv2d_params.kernel_w; kx = kx + 1u) {{
                let iy_signed = i32(oy * conv2d_params.stride_h + ky * conv2d_params.dilation_h) - i32(conv2d_params.pad_h);
                let ix_signed = i32(ox * conv2d_params.stride_w + kx * conv2d_params.dilation_w) - i32(conv2d_params.pad_w);

                if (iy_signed >= 0 && u32(iy_signed) < conv2d_params.height && ix_signed >= 0 && u32(ix_signed) < conv2d_params.width) {{
                    let iy = u32(iy_signed);
                    let ix = u32(ix_signed);
                    let input_idx = b * conv2d_params.c_in * conv2d_params.height * conv2d_params.width
                        + c_in_idx * conv2d_params.height * conv2d_params.width
                        + iy * conv2d_params.width
                        + ix;
                    let weight_idx = oc * c_in_per_group * conv2d_params.kernel_h * conv2d_params.kernel_w
                        + ic * conv2d_params.kernel_h * conv2d_params.kernel_w
                        + ky * conv2d_params.kernel_w
                        + kx;
                    sum = sum + conv2d_input[input_idx] * conv2d_weight[weight_idx];
                }}
            }}
        }}
    }}

    if (conv2d_params.has_bias != 0u) {{
        sum = sum + conv2d_bias[oc];
    }}

    conv2d_output[idx] = sum;
}}
"#,
        t = t,
        suffix = suffix,
        zero = zero,
    ))
}

/// Generate WGSL shader for depthwise conv2d operation.
///
/// Input layout: (N, C, H, W)
/// Weight layout: (C, 1, K_h, K_w)
/// Output layout: (N, C, H_out, W_out)
pub fn generate_depthwise_conv2d_shader(dtype: DType) -> Result<String> {
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "depthwise_conv2d",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let zero = if dtype == DType::F16 { "0.0h" } else { "0.0" };

    Ok(format!(
        r#"// Auto-generated depthwise conv2d shader for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct DepthwiseConv2dParams {{
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_h: u32,
    kernel_w: u32,
    output_h: u32,
    output_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    has_bias: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> depthwise_input: array<{t}>;
@group(0) @binding(1) var<storage, read> depthwise_weight: array<{t}>;
@group(0) @binding(2) var<storage, read> depthwise_bias: array<{t}>;
@group(0) @binding(3) var<storage, read_write> depthwise_output: array<{t}>;
@group(0) @binding(4) var<uniform> depthwise_params: DepthwiseConv2dParams;

@compute @workgroup_size(256)
fn depthwise_conv2d_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = depthwise_params.batch * depthwise_params.channels * depthwise_params.output_h * depthwise_params.output_w;
    if (idx >= total) {{ return; }}

    let ox = idx % depthwise_params.output_w;
    let oy = (idx / depthwise_params.output_w) % depthwise_params.output_h;
    let c = (idx / (depthwise_params.output_w * depthwise_params.output_h)) % depthwise_params.channels;
    let b = idx / (depthwise_params.channels * depthwise_params.output_h * depthwise_params.output_w);

    var sum: {t} = {zero};

    for (var ky: u32 = 0u; ky < depthwise_params.kernel_h; ky = ky + 1u) {{
        for (var kx: u32 = 0u; kx < depthwise_params.kernel_w; kx = kx + 1u) {{
            let iy_signed = i32(oy * depthwise_params.stride_h + ky * depthwise_params.dilation_h) - i32(depthwise_params.pad_h);
            let ix_signed = i32(ox * depthwise_params.stride_w + kx * depthwise_params.dilation_w) - i32(depthwise_params.pad_w);

            if (iy_signed >= 0 && u32(iy_signed) < depthwise_params.height && ix_signed >= 0 && u32(ix_signed) < depthwise_params.width) {{
                let iy = u32(iy_signed);
                let ix = u32(ix_signed);
                let input_idx = b * depthwise_params.channels * depthwise_params.height * depthwise_params.width
                    + c * depthwise_params.height * depthwise_params.width
                    + iy * depthwise_params.width
                    + ix;
                let weight_idx = c * depthwise_params.kernel_h * depthwise_params.kernel_w + ky * depthwise_params.kernel_w + kx;
                sum = sum + depthwise_input[input_idx] * depthwise_weight[weight_idx];
            }}
        }}
    }}

    if (depthwise_params.has_bias != 0u) {{
        sum = sum + depthwise_bias[c];
    }}

    depthwise_output[idx] = sum;
}}
"#,
        t = t,
        suffix = suffix,
        zero = zero,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_wgsl_syntax(source: &str) -> std::result::Result<(), String> {
        use wgpu::naga::front::wgsl;
        let mut frontend = wgsl::Frontend::new();
        frontend
            .parse(source)
            .map(|_| ())
            .map_err(|e| format!("WGSL parse error: {e}"))
    }

    #[test]
    fn test_conv1d_shader_syntax() {
        let shader = generate_conv1d_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for conv1d shader:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_conv2d_shader_syntax() {
        let shader = generate_conv2d_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for conv2d shader:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_depthwise_conv2d_shader_syntax() {
        let shader = generate_depthwise_conv2d_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for depthwise_conv2d shader:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_conv_shaders_int_fails() {
        assert!(generate_conv1d_shader(DType::I32).is_err());
        assert!(generate_conv2d_shader(DType::I32).is_err());
        assert!(generate_depthwise_conv2d_shader(DType::I32).is_err());
    }
}
