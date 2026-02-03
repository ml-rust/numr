//! WGSL shader generation for sparse matrix splitting operations.
//!
//! Split LU and extract lower triangle operations.

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::common::{is_wgpu_supported, wgsl_type};

/// Generate WGSL shader for counting L and U non-zeros per row (split_lu step 1)
pub fn generate_split_lu_count_shader() -> String {
    r#"// Count L and U non-zeros per row for split_lu

struct SplitLuCountParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> l_counts: array<i32>;
@group(0) @binding(3) var<storage, read_write> u_counts: array<i32>;
@group(0) @binding(4) var<uniform> params: SplitLuCountParams;

@compute @workgroup_size(256)
fn split_lu_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= params.n) {
        return;
    }

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    var l_count = 0i;
    var u_count = 0i;

    for (var idx = start; idx < end; idx = idx + 1) {
        let col = col_indices[idx];
        if (col < row) {
            l_count = l_count + 1;
        } else {
            u_count = u_count + 1;
        }
    }

    l_counts[row] = l_count;
    u_counts[row] = u_count;
}
"#
    .to_string()
}

/// Generate WGSL shader for scattering values into L and U (split_lu step 2)
pub fn generate_split_lu_scatter_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "split_lu_scatter",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "split_lu_scatter",
            });
        }
    };

    Ok(format!(
        r#"// Scatter values into L and U matrices

struct SplitLuScatterParams {{
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> values: array<{t}>;
@group(0) @binding(3) var<storage, read_write> l_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read_write> l_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> l_values: array<{t}>;
@group(0) @binding(6) var<storage, read_write> u_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> u_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> u_values: array<{t}>;
@group(0) @binding(9) var<uniform> params: SplitLuScatterParams;

@compute @workgroup_size(256)
fn split_lu_scatter_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = i32(gid.x);
    if (u32(row) >= params.n) {{
        return;
    }}

    let src_start = row_ptrs[row];
    let src_end = row_ptrs[row + 1];

    var l_write_pos = l_row_ptrs[row];
    var u_write_pos = u_row_ptrs[row];

    for (var idx = src_start; idx < src_end; idx = idx + 1) {{
        let col = col_indices[idx];
        let val = values[idx];

        if (col < row) {{
            // Lower triangle
            l_col_indices[l_write_pos] = col;
            l_values[l_write_pos] = val;
            l_write_pos = l_write_pos + 1;
        }} else {{
            // Upper triangle (includes diagonal)
            u_col_indices[u_write_pos] = col;
            u_values[u_write_pos] = val;
            u_write_pos = u_write_pos + 1;
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for scattering values into L matrix only (split_lu part 1)
pub fn generate_split_lu_scatter_l_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "split_lu_scatter_l",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "split_lu_scatter_l",
            });
        }
    };

    Ok(format!(
        r#"// Scatter values into L matrix (lower triangle)

struct SplitLuScatterParams {{
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> values: array<{t}>;
@group(0) @binding(3) var<storage, read_write> l_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read_write> l_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> l_values: array<{t}>;
@group(0) @binding(6) var<uniform> params: SplitLuScatterParams;

@compute @workgroup_size(256)
fn split_lu_scatter_l_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = i32(gid.x);
    if (u32(row) >= params.n) {{
        return;
    }}

    let src_start = row_ptrs[row];
    let src_end = row_ptrs[row + 1];
    var l_write_pos = l_row_ptrs[row];

    for (var idx = src_start; idx < src_end; idx = idx + 1) {{
        let col = col_indices[idx];
        if (col < row) {{
            l_col_indices[l_write_pos] = col;
            l_values[l_write_pos] = values[idx];
            l_write_pos = l_write_pos + 1;
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for scattering values into U matrix only (split_lu part 2)
pub fn generate_split_lu_scatter_u_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "split_lu_scatter_u",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "split_lu_scatter_u",
            });
        }
    };

    Ok(format!(
        r#"// Scatter values into U matrix (upper triangle + diagonal)

struct SplitLuScatterParams {{
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> values: array<{t}>;
@group(0) @binding(3) var<storage, read_write> u_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read_write> u_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> u_values: array<{t}>;
@group(0) @binding(6) var<uniform> params: SplitLuScatterParams;

@compute @workgroup_size(256)
fn split_lu_scatter_u_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = i32(gid.x);
    if (u32(row) >= params.n) {{
        return;
    }}

    let src_start = row_ptrs[row];
    let src_end = row_ptrs[row + 1];
    var u_write_pos = u_row_ptrs[row];

    for (var idx = src_start; idx < src_end; idx = idx + 1) {{
        let col = col_indices[idx];
        if (col >= row) {{
            u_col_indices[u_write_pos] = col;
            u_values[u_write_pos] = values[idx];
            u_write_pos = u_write_pos + 1;
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for counting lower triangle non-zeros per row
pub fn generate_extract_lower_count_shader() -> String {
    r#"// Count lower triangle non-zeros per row

struct ExtractLowerCountParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> l_counts: array<i32>;
@group(0) @binding(3) var<uniform> params: ExtractLowerCountParams;

@compute @workgroup_size(256)
fn extract_lower_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= params.n) {
        return;
    }

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    var count = 0i;

    for (var idx = start; idx < end; idx = idx + 1) {
        let col = col_indices[idx];
        if (col <= row) {
            count = count + 1;
        }
    }

    l_counts[row] = count;
}
"#
    .to_string()
}

/// Generate WGSL shader for scattering lower triangle values
pub fn generate_extract_lower_scatter_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "extract_lower_scatter",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "extract_lower_scatter",
            });
        }
    };

    Ok(format!(
        r#"// Scatter lower triangle values

struct ExtractLowerScatterParams {{
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> values: array<{t}>;
@group(0) @binding(3) var<storage, read_write> l_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read_write> l_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> l_values: array<{t}>;
@group(0) @binding(6) var<uniform> params: ExtractLowerScatterParams;

@compute @workgroup_size(256)
fn extract_lower_scatter_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = i32(gid.x);
    if (u32(row) >= params.n) {{
        return;
    }}

    let src_start = row_ptrs[row];
    let src_end = row_ptrs[row + 1];

    var write_pos = l_row_ptrs[row];

    for (var idx = src_start; idx < src_end; idx = idx + 1) {{
        let col = col_indices[idx];
        if (col <= row) {{
            l_col_indices[write_pos] = col;
            l_values[write_pos] = values[idx];
            write_pos = write_pos + 1;
        }}
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
    fn test_split_lu_count_shader_syntax() {
        let shader = generate_split_lu_count_shader();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for split_lu_count:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_split_lu_scatter_shader_syntax() {
        let shader = generate_split_lu_scatter_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for split_lu_scatter:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_split_lu_scatter_l_shader_syntax() {
        let shader = generate_split_lu_scatter_l_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for split_lu_scatter_l:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_split_lu_scatter_u_shader_syntax() {
        let shader = generate_split_lu_scatter_u_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for split_lu_scatter_u:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_extract_lower_count_shader_syntax() {
        let shader = generate_extract_lower_count_shader();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for extract_lower_count:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_extract_lower_scatter_shader_syntax() {
        let shader = generate_extract_lower_scatter_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for extract_lower_scatter:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_f64_not_supported() {
        assert!(generate_split_lu_scatter_shader(DType::F64).is_err());
        assert!(generate_split_lu_scatter_l_shader(DType::F64).is_err());
        assert!(generate_split_lu_scatter_u_shader(DType::F64).is_err());
        assert!(generate_extract_lower_scatter_shader(DType::F64).is_err());
    }
}
