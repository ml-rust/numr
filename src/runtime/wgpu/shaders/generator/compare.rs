//! WGSL shader generation for comparison operations

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for comparison operations
pub fn generate_compare_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Output is always f32 for consistency (1.0 = true, 0.0 = false)
    Ok(format!(
        r#"// Auto-generated compare operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct CompareParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> compare_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> compare_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> compare_out: array<f32>;
@group(0) @binding(3) var<uniform> compare_params: CompareParams;

@compute @workgroup_size(256)
fn eq_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] == compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn ne_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] != compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn lt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] < compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn le_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] <= compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn gt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] > compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn ge_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] >= compare_b[idx]);
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}
