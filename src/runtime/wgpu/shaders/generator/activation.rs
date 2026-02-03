//! WGSL shader generation for parameterized activation operations
//!
//! Handles activation functions that require more than one parameter,
//! like clamp (min, max).

use super::common::{dtype_suffix, is_wgsl_float, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for clamp operation
///
/// Clamp requires two parameters (min, max) so uses a dedicated params struct.
pub fn generate_clamp_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Only float types support clamp with float bounds
    if !is_wgsl_float(dtype) {
        return Ok(String::new());
    }

    Ok(format!(
        r#"// Auto-generated clamp operation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ClampParams {{
    numel: u32,
    min_val: f32,
    max_val: f32,
    _pad0: u32,
}}

@group(0) @binding(0) var<storage, read_write> clamp_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> clamp_out: array<{t}>;
@group(0) @binding(2) var<uniform> clamp_params: ClampParams;

@compute @workgroup_size(256)
fn clamp_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < clamp_params.numel) {{
        clamp_out[idx] = clamp(clamp_a[idx], {t}(clamp_params.min_val), {t}(clamp_params.max_val));
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}
