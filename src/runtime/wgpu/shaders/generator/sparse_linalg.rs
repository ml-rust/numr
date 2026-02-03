//! WGSL shader generation for sparse linear algebra operations
//!
//! Level-scheduled sparse triangular solve, ILU(0), and IC(0).

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
    _padding: u32,
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

/// Generate WGSL shader for ILU(0) level kernel
pub fn generate_ilu0_level_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "ilu0_level",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "ilu0_level",
            });
        }
    };

    Ok(format!(
        r#"// Level-scheduled ILU(0) factorization kernel

struct Ilu0Params {{
    level_size: u32,
    n: u32,
    diagonal_shift: {t},
    level_start: u32,
}}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> values: array<{t}>;
@group(0) @binding(4) var<storage, read> diag_indices: array<i32>;
@group(0) @binding(5) var<uniform> params: Ilu0Params;

@compute @workgroup_size(256)
fn ilu0_level_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    if (tid >= params.level_size) {{
        return;
    }}

    let i = level_rows[params.level_start + tid];
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1];

    // Process columns k < i (for L factor)
    for (var idx_ik = row_start; idx_ik < row_end; idx_ik = idx_ik + 1) {{
        let k = col_indices[idx_ik];
        if (k >= i) {{
            break;
        }}

        // Get diagonal U[k,k]
        let diag_k = diag_indices[k];
        var diag_val = values[diag_k];

        // Handle zero pivot
        if (abs(diag_val) < 1e-15) {{
            if (params.diagonal_shift > 0.0) {{
                values[diag_k] = params.diagonal_shift;
                diag_val = params.diagonal_shift;
            }}
        }}

        // L[i,k] = A[i,k] / U[k,k]
        let l_ik = values[idx_ik] / diag_val;
        values[idx_ik] = l_ik;

        // Update row i for columns j > k
        let k_start = row_ptrs[k];
        let k_end = row_ptrs[k + 1];

        for (var idx_kj = k_start; idx_kj < k_end; idx_kj = idx_kj + 1) {{
            let j = col_indices[idx_kj];
            if (j <= k) {{
                continue;
            }}

            // Find A[i,j] if it exists (zero fill-in constraint)
            for (var idx_ij = row_start; idx_ij < row_end; idx_ij = idx_ij + 1) {{
                if (col_indices[idx_ij] == j) {{
                    values[idx_ij] = values[idx_ij] - l_ik * values[idx_kj];
                    break;
                }}
                if (col_indices[idx_ij] > j) {{
                    break;
                }}
            }}
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for IC(0) level kernel
pub fn generate_ic0_level_shader(dtype: DType) -> Result<String> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "ic0_level",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = match dtype {
        DType::F32 => "f32",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "ic0_level",
            });
        }
    };

    Ok(format!(
        r#"// Level-scheduled IC(0) factorization kernel

struct Ic0Params {{
    level_size: u32,
    n: u32,
    diagonal_shift: {t},
    level_start: u32,
}}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> values: array<{t}>;
@group(0) @binding(4) var<storage, read> diag_indices: array<i32>;
@group(0) @binding(5) var<uniform> params: Ic0Params;

@compute @workgroup_size(256)
fn ic0_level_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    if (tid >= params.level_size) {{
        return;
    }}

    let i = level_rows[params.level_start + tid];
    let i_start = row_ptrs[i];
    let i_end = row_ptrs[i + 1];

    // Process off-diagonal entries in row i (columns k < i)
    for (var idx_ik = i_start; idx_ik < i_end; idx_ik = idx_ik + 1) {{
        let k = col_indices[idx_ik];
        if (k >= i) {{
            break;
        }}

        let k_start = row_ptrs[k];
        let k_end = row_ptrs[k + 1];

        // Compute inner product contribution
        var sum = values[idx_ik];

        for (var idx_kj = k_start; idx_kj < k_end; idx_kj = idx_kj + 1) {{
            let j = col_indices[idx_kj];
            if (j >= k) {{
                break;
            }}

            // Check if L[i,j] exists
            for (var idx_ij = i_start; idx_ij < i_end; idx_ij = idx_ij + 1) {{
                if (col_indices[idx_ij] == j) {{
                    sum = sum - values[idx_ij] * values[idx_kj];
                    break;
                }}
                if (col_indices[idx_ij] > j) {{
                    break;
                }}
            }}
        }}

        // Divide by L[k,k]
        let diag_k = diag_indices[k];
        values[idx_ik] = sum / values[diag_k];
    }}

    // Compute diagonal L[i,i]
    let diag_i = diag_indices[i];
    var diag_sum = values[diag_i] + params.diagonal_shift;

    for (var idx_ij = i_start; idx_ij < i_end; idx_ij = idx_ij + 1) {{
        let j = col_indices[idx_ij];
        if (j >= i) {{
            break;
        }}
        diag_sum = diag_sum - values[idx_ij] * values[idx_ij];
    }}

    if (diag_sum <= 0.0) {{
        diag_sum = select(1e-10, params.diagonal_shift, params.diagonal_shift > 0.0);
    }}

    values[diag_i] = sqrt(diag_sum);
}}
"#,
        t = t,
        suffix = suffix
    ))
}

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

@group(0) @binding(0) var<storage, read> src: array<{t}>;
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
    fn test_ilu0_level_shader_syntax() {
        let shader = generate_ilu0_level_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!("Invalid WGSL for ilu0_level:\n{}\n\nShader:\n{}", e, shader)
        });
    }

    #[test]
    fn test_ic0_level_shader_syntax() {
        let shader = generate_ic0_level_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!("Invalid WGSL for ic0_level:\n{}\n\nShader:\n{}", e, shader)
        });
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
        assert!(generate_sparse_trsv_lower_shader(DType::F64).is_err());
        assert!(generate_sparse_trsv_upper_shader(DType::F64).is_err());
        assert!(generate_ilu0_level_shader(DType::F64).is_err());
        assert!(generate_ic0_level_shader(DType::F64).is_err());
    }
}
