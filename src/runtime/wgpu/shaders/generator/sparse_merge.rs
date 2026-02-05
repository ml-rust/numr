//! WGSL shader generation for sparse matrix element-wise merge operations
//!
//! Implements two-pass algorithms for CSR/CSC/COO element-wise operations:
//! - add, sub: union semantics (output has nonzeros from both A and B)
//! - mul, div: intersection semantics (output only where both A and B have nonzeros)
//!
//! Each format requires:
//! 1. Count kernel: count output elements per row/column/entry
//! 2. Compute kernel: perform merge and operation

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

// ============================================================================
// CSR Format Shaders
// ============================================================================

/// Generate WGSL shader for CSR merge count (add/sub - union semantics)
///
/// Counts output nonzeros per row for operations that produce union of sparsity patterns.
pub fn generate_csr_merge_count_shader() -> String {
    r#"// CSR merge count kernel (union semantics for add/sub)

const WORKGROUP_SIZE: u32 = 256u;

struct CountParams {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> b_row_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> b_col_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> row_counts: array<i32>;
@group(0) @binding(5) var<uniform> params: CountParams;

@compute @workgroup_size(256)
fn csr_merge_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.nrows) {
        return;
    }

    let a_start = a_row_ptrs[row];
    let a_end = a_row_ptrs[row + 1u];
    let b_start = b_row_ptrs[row];
    let b_end = b_row_ptrs[row + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    // Merge sorted column indices, count unique columns
    while (i < a_end && j < b_end) {
        let a_col = a_col_indices[i];
        let b_col = b_col_indices[j];

        count = count + 1;
        if (a_col < b_col) {
            i = i + 1;
        } else if (a_col > b_col) {
            j = j + 1;
        } else {
            i = i + 1;
            j = j + 1;
        }
    }

    // Add remaining elements from A
    count = count + (a_end - i);
    // Add remaining elements from B
    count = count + (b_end - j);

    row_counts[row] = count;
}
"#
    .to_string()
}

/// Generate WGSL shader for CSR mul count (intersection semantics)
///
/// Counts output nonzeros per row for operations that produce intersection of sparsity patterns.
pub fn generate_csr_mul_count_shader() -> String {
    r#"// CSR mul count kernel (intersection semantics for mul/div)

const WORKGROUP_SIZE: u32 = 256u;

struct CountParams {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> b_row_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> b_col_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> row_counts: array<i32>;
@group(0) @binding(5) var<uniform> params: CountParams;

@compute @workgroup_size(256)
fn csr_mul_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.nrows) {
        return;
    }

    let a_start = a_row_ptrs[row];
    let a_end = a_row_ptrs[row + 1u];
    let b_start = b_row_ptrs[row];
    let b_end = b_row_ptrs[row + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    // Count matching column indices only (intersection)
    while (i < a_end && j < b_end) {
        let a_col = a_col_indices[i];
        let b_col = b_col_indices[j];

        if (a_col < b_col) {
            i = i + 1;
        } else if (a_col > b_col) {
            j = j + 1;
        } else {
            count = count + 1;
            i = i + 1;
            j = j + 1;
        }
    }

    row_counts[row] = count;
}
"#
    .to_string()
}

/// Generate WGSL shader for CSR add compute
pub fn generate_csr_add_compute_shader(dtype: DType) -> Result<String> {
    generate_csr_binary_compute_shader(dtype, "add", "a_val + b_val", "a_val", "b_val")
}

/// Generate WGSL shader for CSR sub compute
pub fn generate_csr_sub_compute_shader(dtype: DType) -> Result<String> {
    generate_csr_binary_compute_shader(dtype, "sub", "a_val - b_val", "a_val", "-b_val")
}

/// Generate WGSL shader for CSR mul compute
pub fn generate_csr_mul_compute_shader(dtype: DType) -> Result<String> {
    generate_csr_intersection_compute_shader(dtype, "mul", "a_val * b_val")
}

/// Generate WGSL shader for CSR div compute
pub fn generate_csr_div_compute_shader(dtype: DType) -> Result<String> {
    generate_csr_intersection_compute_shader(dtype, "div", "a_val / b_val")
}

/// Internal helper for CSR add/sub compute (union semantics)
fn generate_csr_binary_compute_shader(
    dtype: DType,
    op_name: &str,
    both_expr: &str,
    a_only_expr: &str,
    b_only_expr: &str,
) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// CSR {op_name} compute kernel (union semantics)

const WORKGROUP_SIZE: u32 = 256u;

struct ComputeParams {{
    nrows: u32,
}}

@group(0) @binding(0) var<storage, read> a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> a_values: array<{t}>;
@group(0) @binding(3) var<storage, read> b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> b_values: array<{t}>;
@group(0) @binding(6) var<storage, read> out_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> out_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> out_values: array<{t}>;
@group(0) @binding(9) var<uniform> params: ComputeParams;

@compute @workgroup_size(256)
fn csr_{op_name}_compute_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.nrows) {{
        return;
    }}

    let a_start = a_row_ptrs[row];
    let a_end = a_row_ptrs[row + 1u];
    let b_start = b_row_ptrs[row];
    let b_end = b_row_ptrs[row + 1u];

    var out_idx = out_row_ptrs[row];
    var i: i32 = a_start;
    var j: i32 = b_start;

    // Merge sorted column indices
    while (i < a_end && j < b_end) {{
        let a_col = a_col_indices[i];
        let b_col = b_col_indices[j];
        let a_val = a_values[i];
        let b_val = b_values[j];

        if (a_col < b_col) {{
            out_col_indices[out_idx] = a_col;
            out_values[out_idx] = {a_only_expr};
            out_idx = out_idx + 1;
            i = i + 1;
        }} else if (a_col > b_col) {{
            out_col_indices[out_idx] = b_col;
            out_values[out_idx] = {b_only_expr};
            out_idx = out_idx + 1;
            j = j + 1;
        }} else {{
            out_col_indices[out_idx] = a_col;
            out_values[out_idx] = {both_expr};
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }}
    }}

    // Copy remaining from A
    while (i < a_end) {{
        out_col_indices[out_idx] = a_col_indices[i];
        out_values[out_idx] = a_values[i];
        out_idx = out_idx + 1;
        i = i + 1;
    }}

    // Copy remaining from B
    while (j < b_end) {{
        out_col_indices[out_idx] = b_col_indices[j];
        out_values[out_idx] = {b_only_expr_for_b};
        out_idx = out_idx + 1;
        j = j + 1;
    }}
}}
"#,
        t = t,
        op_name = op_name,
        suffix = suffix,
        both_expr = both_expr,
        a_only_expr = a_only_expr,
        b_only_expr = b_only_expr,
        b_only_expr_for_b = if op_name == "sub" {
            "-b_values[j]"
        } else {
            "b_values[j]"
        },
    ))
}

/// Internal helper for CSR mul/div compute (intersection semantics)
fn generate_csr_intersection_compute_shader(
    dtype: DType,
    op_name: &str,
    expr: &str,
) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// CSR {op_name} compute kernel (intersection semantics)

const WORKGROUP_SIZE: u32 = 256u;

struct ComputeParams {{
    nrows: u32,
}}

@group(0) @binding(0) var<storage, read> a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> a_values: array<{t}>;
@group(0) @binding(3) var<storage, read> b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> b_values: array<{t}>;
@group(0) @binding(6) var<storage, read> out_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> out_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> out_values: array<{t}>;
@group(0) @binding(9) var<uniform> params: ComputeParams;

@compute @workgroup_size(256)
fn csr_{op_name}_compute_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= params.nrows) {{
        return;
    }}

    let a_start = a_row_ptrs[row];
    let a_end = a_row_ptrs[row + 1u];
    let b_start = b_row_ptrs[row];
    let b_end = b_row_ptrs[row + 1u];

    var out_idx = out_row_ptrs[row];
    var i: i32 = a_start;
    var j: i32 = b_start;

    // Only output where both A and B have nonzeros (intersection)
    while (i < a_end && j < b_end) {{
        let a_col = a_col_indices[i];
        let b_col = b_col_indices[j];

        if (a_col < b_col) {{
            i = i + 1;
        }} else if (a_col > b_col) {{
            j = j + 1;
        }} else {{
            let a_val = a_values[i];
            let b_val = b_values[j];
            out_col_indices[out_idx] = a_col;
            out_values[out_idx] = {expr};
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }}
    }}
}}
"#,
        t = t,
        op_name = op_name,
        suffix = suffix,
        expr = expr,
    ))
}

// ============================================================================
// CSC Format Shaders (analogous to CSR but operates on columns)
// ============================================================================

/// Generate WGSL shader for CSC merge count (union semantics)
pub fn generate_csc_merge_count_shader() -> String {
    r#"// CSC merge count kernel (union semantics for add/sub)

const WORKGROUP_SIZE: u32 = 256u;

struct CountParams {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> b_col_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> b_row_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> col_counts: array<i32>;
@group(0) @binding(5) var<uniform> params: CountParams;

@compute @workgroup_size(256)
fn csc_merge_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.ncols) {
        return;
    }

    let a_start = a_col_ptrs[col];
    let a_end = a_col_ptrs[col + 1u];
    let b_start = b_col_ptrs[col];
    let b_end = b_col_ptrs[col + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = a_row_indices[i];
        let b_row = b_row_indices[j];

        count = count + 1;
        if (a_row < b_row) {
            i = i + 1;
        } else if (a_row > b_row) {
            j = j + 1;
        } else {
            i = i + 1;
            j = j + 1;
        }
    }

    count = count + (a_end - i);
    count = count + (b_end - j);

    col_counts[col] = count;
}
"#
    .to_string()
}

/// Generate WGSL shader for CSC mul count (intersection semantics)
pub fn generate_csc_mul_count_shader() -> String {
    r#"// CSC mul count kernel (intersection semantics for mul/div)

const WORKGROUP_SIZE: u32 = 256u;

struct CountParams {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> b_col_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> b_row_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> col_counts: array<i32>;
@group(0) @binding(5) var<uniform> params: CountParams;

@compute @workgroup_size(256)
fn csc_mul_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.ncols) {
        return;
    }

    let a_start = a_col_ptrs[col];
    let a_end = a_col_ptrs[col + 1u];
    let b_start = b_col_ptrs[col];
    let b_end = b_col_ptrs[col + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = a_row_indices[i];
        let b_row = b_row_indices[j];

        if (a_row < b_row) {
            i = i + 1;
        } else if (a_row > b_row) {
            j = j + 1;
        } else {
            count = count + 1;
            i = i + 1;
            j = j + 1;
        }
    }

    col_counts[col] = count;
}
"#
    .to_string()
}

/// Generate WGSL shader for CSC add compute
pub fn generate_csc_add_compute_shader(dtype: DType) -> Result<String> {
    generate_csc_binary_compute_shader(dtype, "add", "a_val + b_val", "a_val", "b_val")
}

/// Generate WGSL shader for CSC sub compute
pub fn generate_csc_sub_compute_shader(dtype: DType) -> Result<String> {
    generate_csc_binary_compute_shader(dtype, "sub", "a_val - b_val", "a_val", "-b_val")
}

/// Generate WGSL shader for CSC mul compute
pub fn generate_csc_mul_compute_shader(dtype: DType) -> Result<String> {
    generate_csc_intersection_compute_shader(dtype, "mul", "a_val * b_val")
}

/// Generate WGSL shader for CSC div compute
pub fn generate_csc_div_compute_shader(dtype: DType) -> Result<String> {
    generate_csc_intersection_compute_shader(dtype, "div", "a_val / b_val")
}

/// Internal helper for CSC add/sub compute (union semantics)
fn generate_csc_binary_compute_shader(
    dtype: DType,
    op_name: &str,
    both_expr: &str,
    a_only_expr: &str,
    b_only_expr: &str,
) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// CSC {op_name} compute kernel (union semantics)

const WORKGROUP_SIZE: u32 = 256u;

struct ComputeParams {{
    ncols: u32,
}}

@group(0) @binding(0) var<storage, read> a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> a_values: array<{t}>;
@group(0) @binding(3) var<storage, read> b_col_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> b_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read> b_values: array<{t}>;
@group(0) @binding(6) var<storage, read> out_col_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> out_row_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> out_values: array<{t}>;
@group(0) @binding(9) var<uniform> params: ComputeParams;

@compute @workgroup_size(256)
fn csc_{op_name}_compute_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    if (col >= params.ncols) {{
        return;
    }}

    let a_start = a_col_ptrs[col];
    let a_end = a_col_ptrs[col + 1u];
    let b_start = b_col_ptrs[col];
    let b_end = b_col_ptrs[col + 1u];

    var out_idx = out_col_ptrs[col];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {{
        let a_row = a_row_indices[i];
        let b_row = b_row_indices[j];
        let a_val = a_values[i];
        let b_val = b_values[j];

        if (a_row < b_row) {{
            out_row_indices[out_idx] = a_row;
            out_values[out_idx] = {a_only_expr};
            out_idx = out_idx + 1;
            i = i + 1;
        }} else if (a_row > b_row) {{
            out_row_indices[out_idx] = b_row;
            out_values[out_idx] = {b_only_expr};
            out_idx = out_idx + 1;
            j = j + 1;
        }} else {{
            out_row_indices[out_idx] = a_row;
            out_values[out_idx] = {both_expr};
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }}
    }}

    while (i < a_end) {{
        out_row_indices[out_idx] = a_row_indices[i];
        out_values[out_idx] = a_values[i];
        out_idx = out_idx + 1;
        i = i + 1;
    }}

    while (j < b_end) {{
        out_row_indices[out_idx] = b_row_indices[j];
        out_values[out_idx] = {b_only_expr_for_b};
        out_idx = out_idx + 1;
        j = j + 1;
    }}
}}
"#,
        t = t,
        op_name = op_name,
        suffix = suffix,
        both_expr = both_expr,
        a_only_expr = a_only_expr,
        b_only_expr = b_only_expr,
        b_only_expr_for_b = if op_name == "sub" {
            "-b_values[j]"
        } else {
            "b_values[j]"
        },
    ))
}

/// Internal helper for CSC mul/div compute (intersection semantics)
fn generate_csc_intersection_compute_shader(
    dtype: DType,
    op_name: &str,
    expr: &str,
) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// CSC {op_name} compute kernel (intersection semantics)

const WORKGROUP_SIZE: u32 = 256u;

struct ComputeParams {{
    ncols: u32,
}}

@group(0) @binding(0) var<storage, read> a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> a_values: array<{t}>;
@group(0) @binding(3) var<storage, read> b_col_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> b_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read> b_values: array<{t}>;
@group(0) @binding(6) var<storage, read> out_col_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> out_row_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> out_values: array<{t}>;
@group(0) @binding(9) var<uniform> params: ComputeParams;

@compute @workgroup_size(256)
fn csc_{op_name}_compute_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    if (col >= params.ncols) {{
        return;
    }}

    let a_start = a_col_ptrs[col];
    let a_end = a_col_ptrs[col + 1u];
    let b_start = b_col_ptrs[col];
    let b_end = b_col_ptrs[col + 1u];

    var out_idx = out_col_ptrs[col];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {{
        let a_row = a_row_indices[i];
        let b_row = b_row_indices[j];

        if (a_row < b_row) {{
            i = i + 1;
        }} else if (a_row > b_row) {{
            j = j + 1;
        }} else {{
            let a_val = a_values[i];
            let b_val = b_values[j];
            out_row_indices[out_idx] = a_row;
            out_values[out_idx] = {expr};
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }}
    }}
}}
"#,
        t = t,
        op_name = op_name,
        suffix = suffix,
        expr = expr,
    ))
}

// ============================================================================
// COO Format Shaders
// ============================================================================

// COO merge is more complex since entries aren't sorted by row/col.
// For simplicity, we convert COO to CSR, perform the merge, then optionally convert back.
// This is the standard approach since COO doesn't have efficient merge algorithms.

// ============================================================================
// Exclusive Scan (Prefix Sum) Shader
// ============================================================================

/// Generate WGSL shader for sequential exclusive scan (for small arrays)
///
/// This is a simple sequential scan that works for the row_ptrs/col_ptrs arrays
/// which are typically small (O(nrows) or O(ncols)).
pub fn generate_exclusive_scan_shader() -> String {
    r#"// Sequential exclusive scan for small arrays

const WORKGROUP_SIZE: u32 = 256u;

struct ScanParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<uniform> params: ScanParams;

// Sequential exclusive scan - only first thread does work
// For parallel scan on larger arrays, use work-efficient parallel scan
@compute @workgroup_size(1)
fn exclusive_scan_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    var sum: i32 = 0;
    for (var i: u32 = 0u; i < params.n; i = i + 1u) {
        let val = input[i];
        output[i] = sum;
        sum = sum + val;
    }
    // Final element is total sum
    output[params.n] = sum;
}
"#
    .to_string()
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
    fn test_csr_merge_count_shader_syntax() {
        let shader = generate_csr_merge_count_shader();
        validate_wgsl_syntax(&shader).expect("CSR merge count shader should be valid WGSL");
    }

    #[test]
    fn test_csr_mul_count_shader_syntax() {
        let shader = generate_csr_mul_count_shader();
        validate_wgsl_syntax(&shader).expect("CSR mul count shader should be valid WGSL");
    }

    #[test]
    fn test_csr_add_compute_shader_syntax_f32() {
        let shader = generate_csr_add_compute_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("CSR add compute shader should be valid WGSL");
    }

    #[test]
    fn test_csr_sub_compute_shader_syntax_f32() {
        let shader = generate_csr_sub_compute_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("CSR sub compute shader should be valid WGSL");
    }

    #[test]
    fn test_csr_mul_compute_shader_syntax_f32() {
        let shader = generate_csr_mul_compute_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("CSR mul compute shader should be valid WGSL");
    }

    #[test]
    fn test_csr_div_compute_shader_syntax_f32() {
        let shader = generate_csr_div_compute_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("CSR div compute shader should be valid WGSL");
    }

    #[test]
    fn test_csc_merge_count_shader_syntax() {
        let shader = generate_csc_merge_count_shader();
        validate_wgsl_syntax(&shader).expect("CSC merge count shader should be valid WGSL");
    }

    #[test]
    fn test_csc_mul_count_shader_syntax() {
        let shader = generate_csc_mul_count_shader();
        validate_wgsl_syntax(&shader).expect("CSC mul count shader should be valid WGSL");
    }

    #[test]
    fn test_csc_add_compute_shader_syntax_f32() {
        let shader = generate_csc_add_compute_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("CSC add compute shader should be valid WGSL");
    }

    #[test]
    fn test_exclusive_scan_shader_syntax() {
        let shader = generate_exclusive_scan_shader();
        validate_wgsl_syntax(&shader).expect("Exclusive scan shader should be valid WGSL");
    }
}
