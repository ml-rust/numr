//! WGSL shader generation for sparse triangular solve operations.
//!
//! Level-scheduled sparse triangular solve (forward and backward substitution).

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::common::{is_wgpu_supported, wgsl_type};

/// Generate WGSL shader for level-scheduled sparse lower triangular solve
pub fn generate_sparse_trsv_lower_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_trsv_lower",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_trsv_lower",
            });
        }
    };

    Ok(format!(
        r#"// Level-scheduled sparse lower triangular solve (forward substitution)
// Processes all rows in a single level in parallel

struct TrsvParams {{
    level_size: u32,
    n: u32,
    unit_diagonal: u32,
    level_start: u32,
}}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read> values: array<{t}>;
@group(0) @binding(4) var<storage, read> b: array<{t}>;
@group(0) @binding(5) var<storage, read_write> x: array<{t}>;
@group(0) @binding(6) var<uniform> params: TrsvParams;

@compute @workgroup_size(256)
fn sparse_trsv_lower_level_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    if (tid >= params.level_size) {{
        return;
    }}

    let row = level_rows[params.level_start + tid];
    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    var sum = b[row];
    var diag = {t}(1.0);

    for (var idx = start; idx < end; idx = idx + 1) {{
        let col = col_indices[idx];
        if (col < row) {{
            sum = sum - values[idx] * x[col];
        }} else if (col == row && params.unit_diagonal == 0u) {{
            diag = values[idx];
        }}
    }}

    if (params.unit_diagonal == 0u) {{
        sum = sum / diag;
    }}

    x[row] = sum;
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for level-scheduled sparse upper triangular solve
pub fn generate_sparse_trsv_upper_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_trsv_upper",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_trsv_upper",
            });
        }
    };

    Ok(format!(
        r#"// Level-scheduled sparse upper triangular solve (backward substitution)

struct TrsvParams {{
    level_size: u32,
    n: u32,
    _pad0: u32,
    level_start: u32,
}}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read> values: array<{t}>;
@group(0) @binding(4) var<storage, read> b: array<{t}>;
@group(0) @binding(5) var<storage, read_write> x: array<{t}>;
@group(0) @binding(6) var<uniform> params: TrsvParams;

@compute @workgroup_size(256)
fn sparse_trsv_upper_level_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    if (tid >= params.level_size) {{
        return;
    }}

    let row = level_rows[params.level_start + tid];
    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    var sum = b[row];
    var diag = {t}(1.0);

    for (var idx = start; idx < end; idx = idx + 1) {{
        let col = col_indices[idx];
        if (col > row) {{
            sum = sum - values[idx] * x[col];
        }} else if (col == row) {{
            diag = values[idx];
        }}
    }}

    x[row] = sum / diag;
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for multi-RHS level-scheduled sparse lower triangular solve
/// Handles b and x with shape [n, nrhs] in row-major order
pub fn generate_sparse_trsv_lower_multi_rhs_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_trsv_lower_multi_rhs",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_trsv_lower_multi_rhs",
            });
        }
    };

    Ok(format!(
        r#"// Multi-RHS level-scheduled sparse lower triangular solve (forward substitution)
// Processes all (row, rhs_column) pairs in a single level in parallel

struct TrsvMultiRhsParams {{
    level_size: u32,
    nrhs: u32,
    n: u32,
    unit_diagonal: u32,
    level_start: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read> values: array<{t}>;
@group(0) @binding(4) var<storage, read> b: array<{t}>;
@group(0) @binding(5) var<storage, read_write> x: array<{t}>;
@group(0) @binding(6) var<uniform> params: TrsvMultiRhsParams;

@compute @workgroup_size(256)
fn sparse_trsv_lower_level_multi_rhs_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let total_work = params.level_size * params.nrhs;
    if (tid >= total_work) {{
        return;
    }}

    let row_idx = tid / params.nrhs;
    let rhs_col = tid % params.nrhs;
    let row = level_rows[params.level_start + row_idx];

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    var sum = b[u32(row) * params.nrhs + rhs_col];
    var diag = {t}(1.0);

    for (var idx = start; idx < end; idx = idx + 1) {{
        let col = col_indices[idx];
        if (col < row) {{
            sum = sum - values[idx] * x[u32(col) * params.nrhs + rhs_col];
        }} else if (col == row && params.unit_diagonal == 0u) {{
            diag = values[idx];
        }}
    }}

    if (params.unit_diagonal == 0u) {{
        sum = sum / diag;
    }}

    x[u32(row) * params.nrhs + rhs_col] = sum;
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for multi-RHS level-scheduled sparse upper triangular solve
pub fn generate_sparse_trsv_upper_multi_rhs_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_trsv_upper_multi_rhs",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_trsv_upper_multi_rhs",
            });
        }
    };

    Ok(format!(
        r#"// Multi-RHS level-scheduled sparse upper triangular solve (backward substitution)

struct TrsvMultiRhsParams {{
    level_size: u32,
    nrhs: u32,
    n: u32,
    _pad0: u32,
    level_start: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read> values: array<{t}>;
@group(0) @binding(4) var<storage, read> b: array<{t}>;
@group(0) @binding(5) var<storage, read_write> x: array<{t}>;
@group(0) @binding(6) var<uniform> params: TrsvMultiRhsParams;

@compute @workgroup_size(256)
fn sparse_trsv_upper_level_multi_rhs_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let total_work = params.level_size * params.nrhs;
    if (tid >= total_work) {{
        return;
    }}

    let row_idx = tid / params.nrhs;
    let rhs_col = tid % params.nrhs;
    let row = level_rows[params.level_start + row_idx];

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    var sum = b[u32(row) * params.nrhs + rhs_col];
    var diag = {t}(1.0);

    for (var idx = start; idx < end; idx = idx + 1) {{
        let col = col_indices[idx];
        if (col > row) {{
            sum = sum - values[idx] * x[u32(col) * params.nrhs + rhs_col];
        }} else if (col == row) {{
            diag = values[idx];
        }}
    }}

    x[u32(row) * params.nrhs + rhs_col] = sum / diag;
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
    fn test_sparse_trsv_lower_shader_syntax() {
        let shader = generate_sparse_trsv_lower_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for sparse_trsv_lower:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_sparse_trsv_upper_shader_syntax() {
        let shader = generate_sparse_trsv_upper_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for sparse_trsv_upper:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_f64_not_supported() {
        assert!(generate_sparse_trsv_lower_shader(DType::F64).is_err());
        assert!(generate_sparse_trsv_upper_shader(DType::F64).is_err());
    }
}
