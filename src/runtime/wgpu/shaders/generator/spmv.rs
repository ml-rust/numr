//! WGSL shader generation for sparse matrix-vector and matrix-matrix multiplication.
//!
//! SpMV (y = A * x) and SpMM (C = A * B) for CSR format matrices.
//! Row-parallel implementation that doesn't require atomics.

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for CSR SpMV: y = A * x
///
/// Each workgroup thread processes one row of the sparse matrix.
pub fn generate_csr_spmv_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// CSR Sparse Matrix-Vector Multiplication: y = A * x
// Row-parallel implementation: one thread per row

const WORKGROUP_SIZE: u32 = 256u;

struct SpmvParams {{
    nrows: u32,
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
}}

// CSR format
@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> values: array<{t}>;
// Dense vector x
@group(0) @binding(3) var<storage, read> x: array<{t}>;
// Output vector y
@group(0) @binding(4) var<storage, read_write> y: array<{t}>;
// Parameters
@group(0) @binding(5) var<uniform> params: SpmvParams;

@compute @workgroup_size(256)
fn csr_spmv_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.nrows) {{
        return;
    }}

    let row_start = row_ptrs[row];
    let row_end = row_ptrs[row + 1u];

    var sum: {t} = {zero};
    for (var j: i32 = row_start; j < row_end; j = j + 1) {{
        let col = col_indices[j];
        sum = sum + values[j] * x[col];
    }}

    y[row] = sum;
}}
"#,
        t = t,
        suffix = suffix,
        zero = zero_literal(dtype),
    ))
}

/// Generate WGSL shader for CSR SpMM: C = A * B
///
/// Row-parallel implementation where each thread computes one element of C.
/// Thread (row, col) computes C[row, col] = sum(A[row, :] * B[:, col])
pub fn generate_csr_spmm_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// CSR Sparse Matrix-Dense Matrix Multiplication: C = A * B
// Each thread computes one output element C[row, col]

const WORKGROUP_SIZE: u32 = 256u;

struct SpmmParams {{
    m: u32,       // Number of rows in A (and C)
    k: u32,       // Number of columns in A (and rows in B)
    n: u32,       // Number of columns in B (and C)
    _pad: u32,
}}

// CSR format for A
@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> a_values: array<{t}>;
// Dense matrix B (k x n, row-major)
@group(0) @binding(3) var<storage, read> b: array<{t}>;
// Output matrix C (m x n, row-major)
@group(0) @binding(4) var<storage, read_write> c: array<{t}>;
// Parameters
@group(0) @binding(5) var<uniform> params: SpmmParams;

@compute @workgroup_size(256)
fn csr_spmm_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = params.m * params.n;
    if (idx >= total) {{
        return;
    }}

    let row = idx / params.n;
    let col = idx % params.n;

    let row_start = row_ptrs[row];
    let row_end = row_ptrs[row + 1u];

    var sum: {t} = {zero};
    for (var j: i32 = row_start; j < row_end; j = j + 1) {{
        let a_col = col_indices[j];
        let a_val = a_values[j];
        // B is row-major: B[a_col, col] = b[a_col * n + col]
        let b_idx = u32(a_col) * params.n + col;
        sum = sum + a_val * b[b_idx];
    }}

    // C is row-major: C[row, col] = c[row * n + col]
    c[idx] = sum;
}}
"#,
        t = t,
        suffix = suffix,
        zero = zero_literal(dtype),
    ))
}

/// Get zero literal for dtype
fn zero_literal(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 | DType::F16 => "0.0",
        DType::I32 => "0",
        DType::U32 => "0u",
        _ => "0.0",
    }
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
    fn test_csr_spmv_shader_syntax_f32() {
        let shader = generate_csr_spmv_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpMV shader should be valid WGSL");
    }

    #[test]
    fn test_csr_spmm_shader_syntax_f32() {
        let shader = generate_csr_spmm_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpMM shader should be valid WGSL");
    }
}
