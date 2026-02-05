//! WGSL shader generators for sparse format conversions.
//!
//! Generates shaders for converting between COO, CSR, and CSC formats.
//! Algorithms:
//! - CSR/CSC → COO: Expand pointers to explicit indices
//! - COO → CSR/CSC: Histogram + scan + scatter (counting sort)
//! - CSR ↔ CSC: Direct transpose via histogram + scan + scatter

use crate::dtype::DType;
use crate::error::Result;

use super::common::wgsl_type;

/// Generate shader for expanding CSR row pointers to explicit row indices (CSR → COO).
///
/// Input: row_ptrs[nrows+1], nnz elements total
/// Output: row_indices[nnz] where each element i gets the row index it belongs to
pub fn generate_expand_row_ptrs_shader() -> Result<String> {
    Ok(r#"
// Expand CSR row pointers to explicit row indices
// One thread per row

struct ExpandParams {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> row_indices: array<i32>;
@group(0) @binding(2) var<uniform> params: ExpandParams;

@compute @workgroup_size(256)
fn expand_row_ptrs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.nrows) {
        return;
    }

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1u];

    // Fill all indices in this row with the row number
    for (var i = start; i < end; i = i + 1) {
        row_indices[i] = i32(row);
    }
}
"#
    .to_string())
}

/// Generate shader for expanding CSC column pointers to explicit column indices (CSC → COO).
pub fn generate_expand_col_ptrs_shader() -> Result<String> {
    Ok(r#"
// Expand CSC column pointers to explicit column indices
// One thread per column

struct ExpandParams {
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(2) var<uniform> params: ExpandParams;

@compute @workgroup_size(256)
fn expand_col_ptrs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.ncols) {
        return;
    }

    let start = col_ptrs[col];
    let end = col_ptrs[col + 1u];

    // Fill all indices in this column with the column number
    for (var i = start; i < end; i = i + 1) {
        col_indices[i] = i32(col);
    }
}
"#
    .to_string())
}

/// Generate histogram shader for counting elements per row/column.
///
/// Used by COO→CSR/CSC and CSR↔CSC conversions.
pub fn generate_histogram_shader() -> Result<String> {
    Ok(r#"
// Count elements per bucket (row or column)
// One thread per element

struct HistogramParams {
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params: HistogramParams;

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.nnz) {
        return;
    }

    let bucket = indices[idx];
    atomicAdd(&counts[bucket], 1);
}
"#
    .to_string())
}

/// Generate shader for COO→CSR scatter operation.
///
/// Given sorted row indices and their scatter positions, place elements
/// at their correct positions in the CSR output.
pub fn generate_coo_to_csr_scatter_shader(dtype: DType) -> Result<String> {
    let wgsl_t = wgsl_type(dtype)?;

    Ok(format!(
        r#"
// Scatter COO elements to CSR format using atomic position tracking
// One thread per element

struct ScatterParams {{
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> in_row_indices: array<i32>;
@group(0) @binding(1) var<storage, read> in_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> in_values: array<{wgsl_t}>;
@group(0) @binding(3) var<storage, read_write> row_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> out_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> out_values: array<{wgsl_t}>;
@group(0) @binding(6) var<uniform> params: ScatterParams;

@compute @workgroup_size(256)
fn coo_to_csr_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.nnz) {{
        return;
    }}

    let row = in_row_indices[idx];
    let col = in_col_indices[idx];
    let val = in_values[idx];

    // Atomically get position within this row's segment
    let pos = atomicAdd(&row_ptrs_atomic[row], 1);

    out_col_indices[pos] = col;
    out_values[pos] = val;
}}
"#,
        wgsl_t = wgsl_t
    ))
}

/// Generate shader for COO→CSC scatter operation.
pub fn generate_coo_to_csc_scatter_shader(dtype: DType) -> Result<String> {
    let wgsl_t = wgsl_type(dtype)?;

    Ok(format!(
        r#"
// Scatter COO elements to CSC format using atomic position tracking
// One thread per element

struct ScatterParams {{
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> in_row_indices: array<i32>;
@group(0) @binding(1) var<storage, read> in_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> in_values: array<{wgsl_t}>;
@group(0) @binding(3) var<storage, read_write> col_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> out_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> out_values: array<{wgsl_t}>;
@group(0) @binding(6) var<uniform> params: ScatterParams;

@compute @workgroup_size(256)
fn coo_to_csc_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.nnz) {{
        return;
    }}

    let row = in_row_indices[idx];
    let col = in_col_indices[idx];
    let val = in_values[idx];

    // Atomically get position within this column's segment
    let pos = atomicAdd(&col_ptrs_atomic[col], 1);

    out_row_indices[pos] = row;
    out_values[pos] = val;
}}
"#,
        wgsl_t = wgsl_t
    ))
}

/// Generate shader for CSR→CSC transpose scatter operation.
///
/// Directly converts CSR to CSC without going through COO.
pub fn generate_csr_to_csc_scatter_shader(dtype: DType) -> Result<String> {
    let wgsl_t = wgsl_type(dtype)?;

    Ok(format!(
        r#"
// Scatter CSR elements to CSC format (transpose)
// One thread per row, iterates over row's elements

struct TransposeParams {{
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> in_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> in_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> in_values: array<{wgsl_t}>;
@group(0) @binding(3) var<storage, read_write> col_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> out_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> out_values: array<{wgsl_t}>;
@group(0) @binding(6) var<uniform> params: TransposeParams;

@compute @workgroup_size(256)
fn csr_to_csc_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.nrows) {{
        return;
    }}

    let start = in_row_ptrs[row];
    let end = in_row_ptrs[row + 1u];

    for (var i = start; i < end; i = i + 1) {{
        let col = in_col_indices[i];
        let val = in_values[i];

        // Atomically get position within this column's segment
        let pos = atomicAdd(&col_ptrs_atomic[col], 1);

        out_row_indices[pos] = i32(row);
        out_values[pos] = val;
    }}
}}
"#,
        wgsl_t = wgsl_t
    ))
}

/// Generate shader for CSC→CSR transpose scatter operation.
pub fn generate_csc_to_csr_scatter_shader(dtype: DType) -> Result<String> {
    let wgsl_t = wgsl_type(dtype)?;

    Ok(format!(
        r#"
// Scatter CSC elements to CSR format (transpose)
// One thread per column, iterates over column's elements

struct TransposeParams {{
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> in_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> in_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> in_values: array<{wgsl_t}>;
@group(0) @binding(3) var<storage, read_write> row_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> out_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> out_values: array<{wgsl_t}>;
@group(0) @binding(6) var<uniform> params: TransposeParams;

@compute @workgroup_size(256)
fn csc_to_csr_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    if (col >= params.ncols) {{
        return;
    }}

    let start = in_col_ptrs[col];
    let end = in_col_ptrs[col + 1u];

    for (var i = start; i < end; i = i + 1) {{
        let row = in_row_indices[i];
        let val = in_values[i];

        // Atomically get position within this row's segment
        let pos = atomicAdd(&row_ptrs_atomic[row], 1);

        out_col_indices[pos] = i32(col);
        out_values[pos] = val;
    }}
}}
"#,
        wgsl_t = wgsl_t
    ))
}

/// Generate shader to copy row_ptrs before scatter (since scatter modifies them atomically).
pub fn generate_copy_ptrs_shader() -> Result<String> {
    Ok(r#"
// Copy pointers array (preserves original before scatter)

struct CopyParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<i32>;
@group(0) @binding(1) var<storage, read_write> dst: array<i32>;
@group(0) @binding(2) var<uniform> params: CopyParams;

@compute @workgroup_size(256)
fn copy_ptrs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }
    dst[idx] = src[idx];
}
"#
    .to_string())
}

/// Generate shader for CSR to dense conversion.
///
/// Each thread handles one row, scattering values into the dense output.
pub fn generate_csr_to_dense_shader(dtype: DType) -> Result<String> {
    let wgsl_t = wgsl_type(dtype)?;

    Ok(format!(
        r#"
// Convert CSR sparse matrix to dense format
// One thread per row

struct CsrToDenseParams {{
    nrows: u32,
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
}}

@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> values: array<{wgsl_t}>;
@group(0) @binding(3) var<storage, read_write> dense: array<{wgsl_t}>;
@group(0) @binding(4) var<uniform> params: CsrToDenseParams;

@compute @workgroup_size(256)
fn csr_to_dense(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.nrows) {{
        return;
    }}

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1u];
    let ncols = params.ncols;

    // Scatter this row's values into the dense matrix
    for (var i = start; i < end; i = i + 1) {{
        let col = u32(col_indices[i]);
        let val = values[i];
        // Dense matrix is row-major: index = row * ncols + col
        dense[row * ncols + col] = val;
    }}
}}
"#,
        wgsl_t = wgsl_t
    ))
}

/// Generate shader to count non-zero elements in dense matrix.
///
/// Each thread counts non-zeros in a chunk, atomically adds to global counter.
pub fn generate_count_nonzeros_shader(dtype: DType) -> Result<String> {
    let wgsl_t = wgsl_type(dtype)?;
    let zero_check = match dtype {
        DType::F32 | DType::F64 => "abs(val) >= threshold",
        _ => "val != zero_val",
    };

    Ok(format!(
        r#"
// Count non-zero elements in dense matrix
// Returns total count via atomic counter

struct CountParams {{
    total_elems: u32,
    threshold_bits: u32,
    _pad0: u32,
    _pad1: u32,
}}

@group(0) @binding(0) var<storage, read> dense: array<{wgsl_t}>;
@group(0) @binding(1) var<storage, read_write> count: atomic<u32>;
@group(0) @binding(2) var<uniform> params: CountParams;

@compute @workgroup_size(256)
fn count_nonzeros(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.total_elems) {{
        return;
    }}

    let val = dense[idx];
    let threshold = bitcast<{wgsl_t}>(params.threshold_bits);
    let zero_val = {wgsl_t}(0);

    if ({zero_check}) {{
        atomicAdd(&count, 1u);
    }}
}}
"#,
        wgsl_t = wgsl_t,
        zero_check = zero_check
    ))
}

/// Generate shader for dense to COO conversion (scatter pass).
///
/// Each thread checks one element, if non-zero, atomically gets position and writes to COO.
pub fn generate_dense_to_coo_scatter_shader(dtype: DType) -> Result<String> {
    let wgsl_t = wgsl_type(dtype)?;
    let zero_check = match dtype {
        DType::F32 | DType::F64 => "abs(val) >= threshold",
        _ => "val != zero_val",
    };

    Ok(format!(
        r#"
// Scatter non-zero elements from dense matrix to COO format
// One thread per element, atomic position tracking

struct DenseToCooParams {{
    nrows: u32,
    ncols: u32,
    threshold_bits: u32,
    _pad0: u32,
}}

@group(0) @binding(0) var<storage, read> dense: array<{wgsl_t}>;
@group(0) @binding(1) var<storage, read_write> row_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> values: array<{wgsl_t}>;
@group(0) @binding(4) var<storage, read_write> write_pos: atomic<u32>;
@group(0) @binding(5) var<uniform> params: DenseToCooParams;

@compute @workgroup_size(256)
fn dense_to_coo_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = params.nrows * params.ncols;
    if (idx >= total) {{
        return;
    }}

    let val = dense[idx];
    let threshold = bitcast<{wgsl_t}>(params.threshold_bits);
    let zero_val = {wgsl_t}(0);

    if ({zero_check}) {{
        // Compute row and column from linear index
        let row = idx / params.ncols;
        let col = idx % params.ncols;

        // Atomically get write position
        let pos = atomicAdd(&write_pos, 1u);

        // Write COO entry
        row_indices[pos] = i32(row);
        col_indices[pos] = i32(col);
        values[pos] = val;
    }}
}}
"#,
        wgsl_t = wgsl_t,
        zero_check = zero_check
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
    fn test_expand_row_ptrs_shader_syntax() {
        let shader = generate_expand_row_ptrs_shader().unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for expand_row_ptrs:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_expand_col_ptrs_shader_syntax() {
        let shader = generate_expand_col_ptrs_shader().unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for expand_col_ptrs:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_histogram_shader_syntax() {
        let shader = generate_histogram_shader().unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!("Invalid WGSL for histogram:\n{}\n\nShader:\n{}", e, shader)
        });
    }

    #[test]
    fn test_coo_to_csr_scatter_shader_syntax() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            let shader = generate_coo_to_csr_scatter_shader(dtype).unwrap();
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for coo_to_csr_scatter {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_coo_to_csc_scatter_shader_syntax() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            let shader = generate_coo_to_csc_scatter_shader(dtype).unwrap();
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for coo_to_csc_scatter {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_csr_to_csc_scatter_shader_syntax() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            let shader = generate_csr_to_csc_scatter_shader(dtype).unwrap();
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for csr_to_csc_scatter {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_csc_to_csr_scatter_shader_syntax() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            let shader = generate_csc_to_csr_scatter_shader(dtype).unwrap();
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for csc_to_csr_scatter {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_copy_ptrs_shader_syntax() {
        let shader = generate_copy_ptrs_shader().unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!("Invalid WGSL for copy_ptrs:\n{}\n\nShader:\n{}", e, shader)
        });
    }

    #[test]
    fn test_csr_to_dense_shader_syntax() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            let shader = generate_csr_to_dense_shader(dtype).unwrap();
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for csr_to_dense {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_count_nonzeros_shader_syntax() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            let shader = generate_count_nonzeros_shader(dtype).unwrap();
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for count_nonzeros {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_dense_to_coo_scatter_shader_syntax() {
        for dtype in [DType::F32, DType::I32, DType::U32] {
            let shader = generate_dense_to_coo_scatter_shader(dtype).unwrap();
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for dense_to_coo_scatter {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }
}
