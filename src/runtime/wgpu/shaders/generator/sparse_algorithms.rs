//! WGSL shader generation for sparse matrix algorithms.
//!
//! Implements:
//! - Column-Parallel DSMM: Dense × Sparse Matrix Multiplication
//! - Row-Parallel SpGEMM: Sparse × Sparse Matrix Multiplication (simplified GPU version)

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for column-parallel DSMM: C = A * B
///
/// Dense A [M, K] × Sparse B CSC [K, N] → Dense C [M, N]
///
/// Algorithm:
/// For each column j in B:
///   For each non-zero B[k, j]:
///     C[:, j] += A[:, k] * B[k, j]
///
/// GPU parallelization:
/// - Each thread computes one element C[row, col]
/// - Thread reads A[row, :] and accumulates with sparse column of B
pub fn generate_dsmm_csc_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Column-Parallel Dense × Sparse Matrix Multiplication: C = A * B
// Dense A [M, K] × Sparse B CSC [K, N] → Dense C [M, N]
// Each thread computes one element C[row, col]

const WORKGROUP_SIZE: u32 = 256u;

struct DsmmParams {{
    m: u32,       // Number of rows in A (and C)
    k: u32,       // Number of columns in A (and rows in B)
    n: u32,       // Number of columns in B (and C)
    _pad: u32,
}}

// Dense matrix A (m x k, row-major)
@group(0) @binding(0) var<storage, read> a: array<{t}>;
// CSC format for B
@group(0) @binding(1) var<storage, read> col_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> row_indices: array<i32>;
@group(0) @binding(3) var<storage, read> b_values: array<{t}>;
// Output matrix C (m x n, row-major)
@group(0) @binding(4) var<storage, read_write> c: array<{t}>;
// Parameters
@group(0) @binding(5) var<uniform> params: DsmmParams;

@compute @workgroup_size(256)
fn dsmm_csc_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = params.m * params.n;
    if (idx >= total) {{
        return;
    }}

    let row = idx / params.n;
    let col = idx % params.n;

    // Accumulate C[row, col] = sum over non-zeros in column 'col' of B
    // For each B[k, col], add A[row, k] * B[k, col]
    let col_start = col_ptrs[col];
    let col_end = col_ptrs[col + 1u];

    var sum: {t} = {zero};
    for (var j: i32 = col_start; j < col_end; j = j + 1) {{
        let k = row_indices[j];         // row index in B = column index in A
        let b_val = b_values[j];
        // A is row-major: A[row, k] = a[row * k_dim + k]
        let a_idx = row * params.k + u32(k);
        sum = sum + a[a_idx] * b_val;
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

/// Generate WGSL shader for SpGEMM symbolic phase: count NNZ per output row.
///
/// CSR A [M, K] × CSR B [K, N] → row_nnz[M]
///
/// For small N (< 4096), uses a bitmap to track unique columns.
/// Each workgroup processes one row of the output.
pub fn generate_spgemm_symbolic_shader(dtype: DType) -> Result<String> {
    let suffix = dtype_suffix(dtype)?;
    let _ = wgsl_type(dtype)?; // validate dtype

    Ok(format!(
        r#"// SpGEMM Symbolic Phase: Count NNZ per output row
// CSR A [M, K] × CSR B [K, N] → row_nnz[M]
// Uses bitmap in workgroup memory for small N

const WORKGROUP_SIZE: u32 = 256u;
const MAX_BITMAP_SIZE: u32 = 4096u;  // Max columns we can handle with bitmap

struct SymbolicParams {{
    m: u32,       // Number of rows in A (and output)
    n: u32,       // Number of columns in B (and output)
    _pad0: u32,
    _pad1: u32,
}}

// CSR format for A
@group(0) @binding(0) var<storage, read> a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_col_indices: array<i32>;
// CSR format for B
@group(0) @binding(2) var<storage, read> b_row_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> b_col_indices: array<i32>;
// Output: NNZ per row
@group(0) @binding(4) var<storage, read_write> row_nnz: array<i32>;
// Parameters
@group(0) @binding(5) var<uniform> params: SymbolicParams;
// Global bitmap storage (one bitmap per row, M * ((N+31)/32) u32 words)
@group(0) @binding(6) var<storage, read_write> bitmap: array<atomic<u32>>;

@compute @workgroup_size(256)
fn spgemm_symbolic_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.m) {{
        return;
    }}

    // Calculate bitmap offset for this row
    let words_per_row = (params.n + 31u) / 32u;
    let bitmap_offset = row * words_per_row;

    // Clear this row's bitmap
    for (var w: u32 = 0u; w < words_per_row; w = w + 1u) {{
        atomicStore(&bitmap[bitmap_offset + w], 0u);
    }}

    // For each non-zero in row 'row' of A
    let a_start = a_row_ptrs[row];
    let a_end = a_row_ptrs[row + 1u];

    for (var ai: i32 = a_start; ai < a_end; ai = ai + 1) {{
        let k = a_col_indices[ai];  // column in A = row in B

        // For each non-zero in row k of B
        let b_start = b_row_ptrs[k];
        let b_end = b_row_ptrs[k + 1];

        for (var bi: i32 = b_start; bi < b_end; bi = bi + 1) {{
            let j = b_col_indices[bi];  // column in B = column in output

            // Set bit j in bitmap
            let word_idx = bitmap_offset + u32(j) / 32u;
            let bit_idx = u32(j) % 32u;
            atomicOr(&bitmap[word_idx], 1u << bit_idx);
        }}
    }}

    // Count set bits (popcount)
    var count: i32 = 0;
    for (var w: u32 = 0u; w < words_per_row; w = w + 1u) {{
        let word = atomicLoad(&bitmap[bitmap_offset + w]);
        count = count + i32(countOneBits(word));
    }}

    row_nnz[row] = count;
}}
"#,
        suffix = suffix,
    ))
}

/// Generate WGSL shader for SpGEMM numeric phase: compute output values.
///
/// Uses pre-computed row_ptrs from symbolic phase.
/// Each thread processes one output row using dense accumulator in global memory.
pub fn generate_spgemm_numeric_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// SpGEMM Numeric Phase: Compute output values
// CSR A [M, K] × CSR B [K, N] → CSR C [M, N]
// Uses dense accumulator array per row

const WORKGROUP_SIZE: u32 = 256u;

struct NumericParams {{
    m: u32,       // Number of rows in output
    n: u32,       // Number of columns in output
    threshold: {t},  // Zero tolerance threshold
    _pad: u32,
}}

// CSR format for A
@group(0) @binding(0) var<storage, read> a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> a_values: array<{t}>;
// CSR format for B
@group(0) @binding(3) var<storage, read> b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> b_values: array<{t}>;
// Output CSR (row_ptrs from symbolic, col_indices/values to fill)
@group(0) @binding(6) var<storage, read> c_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> c_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> c_values: array<{t}>;
// Parameters
@group(0) @binding(9) var<uniform> params: NumericParams;
// Dense accumulator (M * N elements, used as temporary per-row storage)
@group(0) @binding(10) var<storage, read_write> accum: array<{t}>;
// Flag array to track which columns have values (M * N elements)
@group(0) @binding(11) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(256)
fn spgemm_numeric_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.m) {{
        return;
    }}

    let accum_offset = row * params.n;

    // Clear accumulator and flags for this row
    for (var col: u32 = 0u; col < params.n; col = col + 1u) {{
        accum[accum_offset + col] = {zero};
        flags[accum_offset + col] = 0u;
    }}

    // Accumulate: C[row, :] = sum over k of A[row, k] * B[k, :]
    let a_start = a_row_ptrs[row];
    let a_end = a_row_ptrs[row + 1u];

    for (var ai: i32 = a_start; ai < a_end; ai = ai + 1) {{
        let k = a_col_indices[ai];
        let a_val = a_values[ai];

        let b_start = b_row_ptrs[k];
        let b_end = b_row_ptrs[k + 1];

        for (var bi: i32 = b_start; bi < b_end; bi = bi + 1) {{
            let j = b_col_indices[bi];
            let b_val = b_values[bi];
            let idx = accum_offset + u32(j);
            accum[idx] = accum[idx] + a_val * b_val;
            flags[idx] = 1u;  // Mark column as having a value
        }}
    }}

    // Write non-zero values to output CSR (sorted by column)
    let c_start = c_row_ptrs[row];
    var write_idx: i32 = c_start;

    for (var col: u32 = 0u; col < params.n; col = col + 1u) {{
        let idx = accum_offset + col;
        if (flags[idx] != 0u) {{
            let val = accum[idx];
            // Check threshold
            if (abs(val) > params.threshold) {{
                c_col_indices[write_idx] = i32(col);
                c_values[write_idx] = val;
                write_idx = write_idx + 1;
            }}
        }}
    }}
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
    fn test_dsmm_csc_shader_syntax_f32() {
        let shader = generate_dsmm_csc_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("DSMM shader should be valid WGSL");
    }

    #[test]
    fn test_spgemm_symbolic_shader_syntax_f32() {
        let shader = generate_spgemm_symbolic_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpGEMM symbolic shader should be valid WGSL");
    }

    #[test]
    fn test_spgemm_numeric_shader_syntax_f32() {
        let shader = generate_spgemm_numeric_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpGEMM numeric shader should be valid WGSL");
    }
}
