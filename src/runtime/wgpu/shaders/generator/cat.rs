//! WGSL shader generation for concatenation operations

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for cat_copy operation (one tensor at a time)
///
/// This kernel copies data from a source tensor to the appropriate position
/// in the concatenated output tensor. It's called once per input tensor.
pub fn generate_cat_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated cat operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct CatParams {{
    outer_size: u32,
    src_cat_size: u32,
    dst_cat_size: u32,
    cat_offset: u32,
    inner_size: u32,
    total_elements: u32,
}}

@group(0) @binding(0) var<storage, read_write> cat_src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> cat_dst: array<{t}>;
@group(0) @binding(2) var<uniform> cat_params: CatParams;

@compute @workgroup_size(256)
fn cat_copy_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= cat_params.total_elements) {{
        return;
    }}

    // Decompose idx into (outer, cat_i, inner) for source tensor
    let inner = idx % cat_params.inner_size;
    let remaining = idx / cat_params.inner_size;
    let cat_i = remaining % cat_params.src_cat_size;
    let outer = remaining / cat_params.src_cat_size;

    // Compute destination index
    let dst_idx = outer * cat_params.dst_cat_size * cat_params.inner_size
                + (cat_params.cat_offset + cat_i) * cat_params.inner_size
                + inner;

    cat_dst[dst_idx] = cat_src[idx];
}}
"#,
        t = t,
        suffix = suffix
    ))
}
