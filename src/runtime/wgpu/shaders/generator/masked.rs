//! WGSL shader generation for masked operations (masked_fill and masked_select)

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for masked_fill operation
pub fn generate_masked_fill_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated masked_fill operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct MaskedFillParams {{
    numel: u32,
    fill_value: f32,
}}

@group(0) @binding(0) var<storage, read> input: array<{t}>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<{t}>;
@group(0) @binding(3) var<uniform> params: MaskedFillParams;

@compute @workgroup_size(256)
fn masked_fill_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.numel) {{
        return;
    }}

    if (mask[idx] != 0u) {{
        output[idx] = {t}(params.fill_value);
    }} else {{
        output[idx] = input[idx];
    }}
}}
"#,
        t = t,
        suffix = suffix,
    ))
}

/// Generate WGSL shader for masked_select operation
/// This is a two-phase operation:
/// 1. Count phase: count how many elements are selected (uses atomic)
/// 2. Prefix sum phase: compute exclusive prefix sum of mask
/// 3. Gather phase: copy selected elements to output
pub fn generate_masked_select_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated masked_select operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

// Phase 1: Count masked elements
struct CountParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read> count_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> count_result: atomic<u32>;
@group(0) @binding(2) var<uniform> count_params: CountParams;

var<workgroup> shared_count: atomic<u32>;

@compute @workgroup_size(256)
fn masked_count(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {{
    if (lid.x == 0u) {{
        atomicStore(&shared_count, 0u);
    }}
    workgroupBarrier();

    var local_count: u32 = 0u;
    var i = gid.x;
    while (i < count_params.numel) {{
        if (count_mask[i] != 0u) {{
            local_count = local_count + 1u;
        }}
        i = i + 256u * 256u; // Grid stride
    }}

    atomicAdd(&shared_count, local_count);
    workgroupBarrier();

    if (lid.x == 0u) {{
        atomicAdd(&count_result, atomicLoad(&shared_count));
    }}
}}

// Phase 2: Compute prefix sum (sequential - for small arrays)
struct PrefixSumParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read> prefix_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_sum: array<u32>;
@group(0) @binding(2) var<uniform> prefix_params: PrefixSumParams;

@compute @workgroup_size(1)
fn masked_prefix_sum(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{
        return;
    }}

    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < prefix_params.numel; i = i + 1u) {{
        prefix_sum[i] = sum;
        if (prefix_mask[i] != 0u) {{
            sum = sum + 1u;
        }}
    }}
}}

// Phase 3: Gather selected elements
struct SelectParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read> select_input: array<{t}>;
@group(0) @binding(1) var<storage, read> select_mask: array<u32>;
@group(0) @binding(2) var<storage, read> select_prefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> select_output: array<{t}>;
@group(0) @binding(4) var<uniform> select_params: SelectParams;

@compute @workgroup_size(256)
fn masked_select_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= select_params.numel) {{
        return;
    }}

    if (select_mask[idx] != 0u) {{
        let out_idx = select_prefix[idx];
        select_output[out_idx] = select_input[idx];
    }}
}}
"#,
        t = t,
        suffix = suffix,
    ))
}
