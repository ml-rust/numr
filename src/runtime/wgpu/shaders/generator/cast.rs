//! WGSL shader generation for dtype cast operations

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for dtype cast operations
///
/// WebGPU-supported casts:
/// - F32 ↔ I32 ↔ U32
///
/// Each cast direction requires a separate entry point since WGSL
/// doesn't support templates.
pub fn generate_cast_shader(src_dtype: DType, dst_dtype: DType) -> Result<String> {
    let src_t = wgsl_type(src_dtype)?;
    let dst_t = wgsl_type(dst_dtype)?;
    let src_suffix = dtype_suffix(src_dtype)?;
    let dst_suffix = dtype_suffix(dst_dtype)?;

    // For same-type cast, just return a no-op shader (shouldn't be called)
    if src_dtype == dst_dtype {
        return Ok(format!(
            r#"// No-op cast shader for {src_t} -> {dst_t}
// This should be optimized away at dispatch time
"#
        ));
    }

    Ok(format!(
        r#"// Auto-generated cast operation: {src_t} -> {dst_t}

const WORKGROUP_SIZE: u32 = 256u;

struct CastParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read> cast_input: array<{src_t}>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<{dst_t}>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_{src_suffix}_to_{dst_suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < cast_params.numel) {{
        cast_output[idx] = {dst_t}(cast_input[idx]);
    }}
}}
"#,
        src_t = src_t,
        dst_t = dst_t,
        src_suffix = src_suffix,
        dst_suffix = dst_suffix
    ))
}

/// Generate all cast shaders for a given source dtype
///
/// Returns a combined shader with all casts from the source type.
pub fn generate_all_casts_from(src_dtype: DType) -> Result<String> {
    let src_t = wgsl_type(src_dtype)?;
    let src_suffix = dtype_suffix(src_dtype)?;

    let targets: &[DType] = match src_dtype {
        DType::F32 => &[DType::I32, DType::U32],
        DType::I32 => &[DType::F32, DType::U32],
        DType::U32 => &[DType::F32, DType::I32],
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: src_dtype,
                op: "cast",
            });
        }
    };

    let mut shader = format!(
        r#"// Auto-generated cast operations from {src_t}

const WORKGROUP_SIZE: u32 = 256u;

struct CastParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read> cast_input: array<{src_t}>;
"#
    );

    for &dst_dtype in targets {
        let dst_t = wgsl_type(dst_dtype)?;
        let dst_suffix = dtype_suffix(dst_dtype)?;

        shader.push_str(&format!(
            r#"
// Cast {src_t} -> {dst_t}
@group(0) @binding(1) var<storage, read_write> cast_output_{dst_suffix}: array<{dst_t}>;
@group(0) @binding(2) var<uniform> cast_params_{dst_suffix}: CastParams;

@compute @workgroup_size(256)
fn cast_{src_suffix}_to_{dst_suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < cast_params_{dst_suffix}.numel) {{
        cast_output_{dst_suffix}[idx] = {dst_t}(cast_input[idx]);
    }}
}}
"#
        ));
    }

    Ok(shader)
}
