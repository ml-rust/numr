//! WGSL shader generation for sparse utility operations.
//!
//! Finding diagonal indices and copying vectors.

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::common::{is_wgpu_supported, wgsl_type};

/// Generate WGSL shader for finding diagonal indices
pub fn generate_find_diag_indices_shader() -> String {
    r#"// Find diagonal indices in CSR matrix

struct DiagParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> diag_indices: array<i32>;
@group(0) @binding(3) var<uniform> params: DiagParams;

@compute @workgroup_size(256)
fn find_diag_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= params.n) {
        return;
    }

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    diag_indices[row] = -1;  // Default: no diagonal found

    for (var idx = start; idx < end; idx = idx + 1) {
        if (col_indices[idx] == row) {
            diag_indices[row] = idx;
            break;
        }
    }
}
"#
    .to_string()
}

/// Generate WGSL shader for copying vectors
pub fn generate_copy_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "copy" });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => return Err(Error::UnsupportedDType { dtype, op: "copy" }),
    };

    Ok(format!(
        r#"// Copy vector

struct CopyParams {{
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

// Note: All buffers use read_write for compatibility with LayoutKey-based layouts
@group(0) @binding(0) var<storage, read_write> src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> dst: array<{t}>;
@group(0) @binding(2) var<uniform> params: CopyParams;

@compute @workgroup_size(256)
fn copy_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < params.n) {{
        dst[idx] = src[idx];
    }}
}}
"#,
        t = t,
        suffix = suffix
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
    fn test_find_diag_indices_shader_syntax() {
        let shader = generate_find_diag_indices_shader();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for find_diag_indices:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_copy_shader_syntax() {
        let shader = generate_copy_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader)
            .unwrap_or_else(|e| panic!("Invalid WGSL for copy:\n{}\n\nShader:\n{}", e, shader));
    }

    #[test]
    fn test_f64_not_supported() {
        assert!(generate_copy_shader(DType::F64).is_err());
    }
}
