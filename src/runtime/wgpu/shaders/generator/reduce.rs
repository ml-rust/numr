//! WGSL shader generation for reduction operations

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for reduction operations
pub fn generate_reduce_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Workgroup shared memory for reductions
    Ok(format!(
        r#"// Auto-generated reduce operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> reduce_shared: array<{t}, 256>;

struct ReduceParams {{
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}}

@group(0) @binding(0) var<storage, read_write> reduce_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> reduce_output: array<{t}>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

@compute @workgroup_size(256)
fn reduce_sum_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let outer_idx = group_id.x;

    if (outer_idx >= reduce_params.outer_size) {{
        return;
    }}

    let reduce_size = reduce_params.reduce_size;
    let base_offset = outer_idx * reduce_size;

    // Each thread accumulates multiple elements
    var sum: {t} = {zero};
    var i: u32 = tid;
    while (i < reduce_size) {{
        sum = sum + reduce_input[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }}

    reduce_shared[tid] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        reduce_output[outer_idx] = reduce_shared[0];
    }}
}}

@compute @workgroup_size(256)
fn reduce_max_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let outer_idx = group_id.x;

    if (outer_idx >= reduce_params.outer_size) {{
        return;
    }}

    let reduce_size = reduce_params.reduce_size;
    let base_offset = outer_idx * reduce_size;

    var max_val: {t} = {min_val};
    var i: u32 = tid;
    while (i < reduce_size) {{
        max_val = max(max_val, reduce_input[base_offset + i]);
        i = i + WORKGROUP_SIZE;
    }}

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        reduce_output[outer_idx] = reduce_shared[0];
    }}
}}

@compute @workgroup_size(256)
fn reduce_min_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let outer_idx = group_id.x;

    if (outer_idx >= reduce_params.outer_size) {{
        return;
    }}

    let reduce_size = reduce_params.reduce_size;
    let base_offset = outer_idx * reduce_size;

    var min_val: {t} = {max_val};
    var i: u32 = tid;
    while (i < reduce_size) {{
        min_val = min(min_val, reduce_input[base_offset + i]);
        i = i + WORKGROUP_SIZE;
    }}

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        reduce_output[outer_idx] = reduce_shared[0];
    }}
}}
"#,
        t = t,
        suffix = suffix,
        zero = match dtype {
            DType::F32 | DType::F16 => "0.0",
            _ => "0",
        },
        min_val = match dtype {
            DType::F32 => "-3.402823e+38", // -FLT_MAX
            DType::F16 => "-65504.0",
            DType::I32 => "-2147483648",
            DType::U32 => "0u",
            _ => "0",
        },
        max_val = match dtype {
            DType::F32 => "3.402823e+38", // FLT_MAX
            DType::F16 => "65504.0",
            DType::I32 => "2147483647",
            DType::U32 => "4294967295u",
            _ => "0",
        },
    ))
}
