//! WGSL shader generation for normalization operations

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for normalization operations (float types only)
pub fn generate_norm_shader(dtype: DType) -> Result<String> {
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "normalization (requires float type)",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated normalization operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> norm_shared: array<{t}, 256>;
var<workgroup> ln_shared_mean: array<{t}, 256>;
var<workgroup> ln_shared_var: array<{t}, 256>;

struct RmsNormParams {{
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}}

@group(0) @binding(0) var<storage, read_write> rms_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> rms_weight: array<{t}>;
@group(0) @binding(2) var<storage, read_write> rms_output: array<{t}>;
@group(0) @binding(3) var<uniform> rms_params: RmsNormParams;

@compute @workgroup_size(256)
fn rms_norm_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= rms_params.batch_size) {{
        return;
    }}

    let hidden_size = rms_params.hidden_size;
    let eps = {t}(rms_params.eps);
    let base_offset = batch_idx * hidden_size;

    // Compute sum of squares
    var sum_sq: {t} = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {{
        let val = rms_input[base_offset + i];
        sum_sq = sum_sq + val * val;
        i = i + WORKGROUP_SIZE;
    }}

    norm_shared[tid] = sum_sq;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            norm_shared[tid] = norm_shared[tid] + norm_shared[tid + s];
        }}
        workgroupBarrier();
    }}

    let rms = sqrt(norm_shared[0] / {t}(hidden_size) + eps);
    workgroupBarrier();

    // Normalize and apply weight
    i = tid;
    while (i < hidden_size) {{
        rms_output[base_offset + i] = rms_input[base_offset + i] / rms * rms_weight[i];
        i = i + WORKGROUP_SIZE;
    }}
}}

struct LayerNormParams {{
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}}

@group(0) @binding(0) var<storage, read_write> ln_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> ln_weight: array<{t}>;
@group(0) @binding(2) var<storage, read_write> ln_bias: array<{t}>;
@group(0) @binding(3) var<storage, read_write> ln_output: array<{t}>;
@group(0) @binding(4) var<uniform> ln_params: LayerNormParams;

@compute @workgroup_size(256)
fn layer_norm_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= ln_params.batch_size) {{
        return;
    }}

    let hidden_size = ln_params.hidden_size;
    let eps = {t}(ln_params.eps);
    let base_offset = batch_idx * hidden_size;

    // Compute mean
    var sum: {t} = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {{
        sum = sum + ln_input[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }}

    ln_shared_mean[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            ln_shared_mean[tid] = ln_shared_mean[tid] + ln_shared_mean[tid + s];
        }}
        workgroupBarrier();
    }}

    let mean_val = ln_shared_mean[0] / {t}(hidden_size);
    workgroupBarrier();

    // Compute variance
    var var_sum: {t} = 0.0;
    i = tid;
    while (i < hidden_size) {{
        let diff = ln_input[base_offset + i] - mean_val;
        var_sum = var_sum + diff * diff;
        i = i + WORKGROUP_SIZE;
    }}

    ln_shared_var[tid] = var_sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            ln_shared_var[tid] = ln_shared_var[tid] + ln_shared_var[tid + s];
        }}
        workgroupBarrier();
    }}

    let variance = ln_shared_var[0] / {t}(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Normalize and apply affine
    i = tid;
    while (i < hidden_size) {{
        let normalized = (ln_input[base_offset + i] - mean_val) * inv_std;
        ln_output[base_offset + i] = normalized * ln_weight[i] + ln_bias[i];
        i = i + WORKGROUP_SIZE;
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}
