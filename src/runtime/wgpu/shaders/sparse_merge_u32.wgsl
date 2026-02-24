// Sparse merge compute shaders - U32
//
// CSR: csr_add_compute_u32, csr_sub_compute_u32, csr_mul_compute_u32, csr_div_compute_u32
// CSC: csc_add_compute_u32, csc_sub_compute_u32, csc_mul_compute_u32, csc_div_compute_u32
//
// Note: U32 subtraction uses wrapping arithmetic. Sub b-only case emits 0u - b_val.

// ============================================================================
// csr_add_compute_u32  (union semantics)
// ============================================================================

struct CsrAddU32Params {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> csr_add_u32_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csr_add_u32_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csr_add_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csr_add_u32_b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csr_add_u32_b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csr_add_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csr_add_u32_out_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csr_add_u32_out_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csr_add_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csr_add_u32_params: CsrAddU32Params;

@compute @workgroup_size(256)
fn csr_add_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= csr_add_u32_params.nrows) {
        return;
    }

    let a_start = csr_add_u32_a_row_ptrs[row];
    let a_end = csr_add_u32_a_row_ptrs[row + 1u];
    let b_start = csr_add_u32_b_row_ptrs[row];
    let b_end = csr_add_u32_b_row_ptrs[row + 1u];

    var out_idx = csr_add_u32_out_row_ptrs[row];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_col = csr_add_u32_a_col_indices[i];
        let b_col = csr_add_u32_b_col_indices[j];
        let a_val = csr_add_u32_a_values[i];
        let b_val = csr_add_u32_b_values[j];

        if (a_col < b_col) {
            csr_add_u32_out_col_indices[out_idx] = a_col;
            csr_add_u32_out_values[out_idx] = a_val;
            out_idx = out_idx + 1;
            i = i + 1;
        } else if (a_col > b_col) {
            csr_add_u32_out_col_indices[out_idx] = b_col;
            csr_add_u32_out_values[out_idx] = b_val;
            out_idx = out_idx + 1;
            j = j + 1;
        } else {
            csr_add_u32_out_col_indices[out_idx] = a_col;
            csr_add_u32_out_values[out_idx] = a_val + b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }

    while (i < a_end) {
        csr_add_u32_out_col_indices[out_idx] = csr_add_u32_a_col_indices[i];
        csr_add_u32_out_values[out_idx] = csr_add_u32_a_values[i];
        out_idx = out_idx + 1;
        i = i + 1;
    }

    while (j < b_end) {
        csr_add_u32_out_col_indices[out_idx] = csr_add_u32_b_col_indices[j];
        csr_add_u32_out_values[out_idx] = csr_add_u32_b_values[j];
        out_idx = out_idx + 1;
        j = j + 1;
    }
}

// ============================================================================
// csr_sub_compute_u32  (union semantics, wrapping subtraction)
// ============================================================================

struct CsrSubU32Params {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> csr_sub_u32_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csr_sub_u32_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csr_sub_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csr_sub_u32_b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csr_sub_u32_b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csr_sub_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csr_sub_u32_out_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csr_sub_u32_out_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csr_sub_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csr_sub_u32_params: CsrSubU32Params;

@compute @workgroup_size(256)
fn csr_sub_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= csr_sub_u32_params.nrows) {
        return;
    }

    let a_start = csr_sub_u32_a_row_ptrs[row];
    let a_end = csr_sub_u32_a_row_ptrs[row + 1u];
    let b_start = csr_sub_u32_b_row_ptrs[row];
    let b_end = csr_sub_u32_b_row_ptrs[row + 1u];

    var out_idx = csr_sub_u32_out_row_ptrs[row];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_col = csr_sub_u32_a_col_indices[i];
        let b_col = csr_sub_u32_b_col_indices[j];
        let a_val = csr_sub_u32_a_values[i];
        let b_val = csr_sub_u32_b_values[j];

        if (a_col < b_col) {
            csr_sub_u32_out_col_indices[out_idx] = a_col;
            csr_sub_u32_out_values[out_idx] = a_val;
            out_idx = out_idx + 1;
            i = i + 1;
        } else if (a_col > b_col) {
            csr_sub_u32_out_col_indices[out_idx] = b_col;
            csr_sub_u32_out_values[out_idx] = 0u - b_val;
            out_idx = out_idx + 1;
            j = j + 1;
        } else {
            csr_sub_u32_out_col_indices[out_idx] = a_col;
            csr_sub_u32_out_values[out_idx] = a_val - b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }

    while (i < a_end) {
        csr_sub_u32_out_col_indices[out_idx] = csr_sub_u32_a_col_indices[i];
        csr_sub_u32_out_values[out_idx] = csr_sub_u32_a_values[i];
        out_idx = out_idx + 1;
        i = i + 1;
    }

    while (j < b_end) {
        csr_sub_u32_out_col_indices[out_idx] = csr_sub_u32_b_col_indices[j];
        csr_sub_u32_out_values[out_idx] = 0u - csr_sub_u32_b_values[j];
        out_idx = out_idx + 1;
        j = j + 1;
    }
}

// ============================================================================
// csr_mul_compute_u32  (intersection semantics)
// ============================================================================

struct CsrMulU32Params {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> csr_mul_u32_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csr_mul_u32_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csr_mul_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csr_mul_u32_b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csr_mul_u32_b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csr_mul_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csr_mul_u32_out_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csr_mul_u32_out_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csr_mul_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csr_mul_u32_params: CsrMulU32Params;

@compute @workgroup_size(256)
fn csr_mul_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= csr_mul_u32_params.nrows) {
        return;
    }

    let a_start = csr_mul_u32_a_row_ptrs[row];
    let a_end = csr_mul_u32_a_row_ptrs[row + 1u];
    let b_start = csr_mul_u32_b_row_ptrs[row];
    let b_end = csr_mul_u32_b_row_ptrs[row + 1u];

    var out_idx = csr_mul_u32_out_row_ptrs[row];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_col = csr_mul_u32_a_col_indices[i];
        let b_col = csr_mul_u32_b_col_indices[j];

        if (a_col < b_col) {
            i = i + 1;
        } else if (a_col > b_col) {
            j = j + 1;
        } else {
            let a_val = csr_mul_u32_a_values[i];
            let b_val = csr_mul_u32_b_values[j];
            csr_mul_u32_out_col_indices[out_idx] = a_col;
            csr_mul_u32_out_values[out_idx] = a_val * b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }
}

// ============================================================================
// csr_div_compute_u32  (intersection semantics)
// ============================================================================

struct CsrDivU32Params {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> csr_div_u32_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csr_div_u32_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csr_div_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csr_div_u32_b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csr_div_u32_b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csr_div_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csr_div_u32_out_row_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csr_div_u32_out_col_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csr_div_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csr_div_u32_params: CsrDivU32Params;

@compute @workgroup_size(256)
fn csr_div_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= csr_div_u32_params.nrows) {
        return;
    }

    let a_start = csr_div_u32_a_row_ptrs[row];
    let a_end = csr_div_u32_a_row_ptrs[row + 1u];
    let b_start = csr_div_u32_b_row_ptrs[row];
    let b_end = csr_div_u32_b_row_ptrs[row + 1u];

    var out_idx = csr_div_u32_out_row_ptrs[row];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_col = csr_div_u32_a_col_indices[i];
        let b_col = csr_div_u32_b_col_indices[j];

        if (a_col < b_col) {
            i = i + 1;
        } else if (a_col > b_col) {
            j = j + 1;
        } else {
            let a_val = csr_div_u32_a_values[i];
            let b_val = csr_div_u32_b_values[j];
            csr_div_u32_out_col_indices[out_idx] = a_col;
            csr_div_u32_out_values[out_idx] = a_val / b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }
}

// ============================================================================
// csc_add_compute_u32  (union semantics)
// ============================================================================

struct CscAddU32Params {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> csc_add_u32_a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csc_add_u32_a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csc_add_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csc_add_u32_b_col_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csc_add_u32_b_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csc_add_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csc_add_u32_out_col_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csc_add_u32_out_row_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csc_add_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csc_add_u32_params: CscAddU32Params;

@compute @workgroup_size(256)
fn csc_add_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= csc_add_u32_params.ncols) {
        return;
    }

    let a_start = csc_add_u32_a_col_ptrs[col];
    let a_end = csc_add_u32_a_col_ptrs[col + 1u];
    let b_start = csc_add_u32_b_col_ptrs[col];
    let b_end = csc_add_u32_b_col_ptrs[col + 1u];

    var out_idx = csc_add_u32_out_col_ptrs[col];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = csc_add_u32_a_row_indices[i];
        let b_row = csc_add_u32_b_row_indices[j];
        let a_val = csc_add_u32_a_values[i];
        let b_val = csc_add_u32_b_values[j];

        if (a_row < b_row) {
            csc_add_u32_out_row_indices[out_idx] = a_row;
            csc_add_u32_out_values[out_idx] = a_val;
            out_idx = out_idx + 1;
            i = i + 1;
        } else if (a_row > b_row) {
            csc_add_u32_out_row_indices[out_idx] = b_row;
            csc_add_u32_out_values[out_idx] = b_val;
            out_idx = out_idx + 1;
            j = j + 1;
        } else {
            csc_add_u32_out_row_indices[out_idx] = a_row;
            csc_add_u32_out_values[out_idx] = a_val + b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }

    while (i < a_end) {
        csc_add_u32_out_row_indices[out_idx] = csc_add_u32_a_row_indices[i];
        csc_add_u32_out_values[out_idx] = csc_add_u32_a_values[i];
        out_idx = out_idx + 1;
        i = i + 1;
    }

    while (j < b_end) {
        csc_add_u32_out_row_indices[out_idx] = csc_add_u32_b_row_indices[j];
        csc_add_u32_out_values[out_idx] = csc_add_u32_b_values[j];
        out_idx = out_idx + 1;
        j = j + 1;
    }
}

// ============================================================================
// csc_sub_compute_u32  (union semantics, wrapping subtraction)
// ============================================================================

struct CscSubU32Params {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> csc_sub_u32_a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csc_sub_u32_a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csc_sub_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csc_sub_u32_b_col_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csc_sub_u32_b_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csc_sub_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csc_sub_u32_out_col_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csc_sub_u32_out_row_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csc_sub_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csc_sub_u32_params: CscSubU32Params;

@compute @workgroup_size(256)
fn csc_sub_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= csc_sub_u32_params.ncols) {
        return;
    }

    let a_start = csc_sub_u32_a_col_ptrs[col];
    let a_end = csc_sub_u32_a_col_ptrs[col + 1u];
    let b_start = csc_sub_u32_b_col_ptrs[col];
    let b_end = csc_sub_u32_b_col_ptrs[col + 1u];

    var out_idx = csc_sub_u32_out_col_ptrs[col];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = csc_sub_u32_a_row_indices[i];
        let b_row = csc_sub_u32_b_row_indices[j];
        let a_val = csc_sub_u32_a_values[i];
        let b_val = csc_sub_u32_b_values[j];

        if (a_row < b_row) {
            csc_sub_u32_out_row_indices[out_idx] = a_row;
            csc_sub_u32_out_values[out_idx] = a_val;
            out_idx = out_idx + 1;
            i = i + 1;
        } else if (a_row > b_row) {
            csc_sub_u32_out_row_indices[out_idx] = b_row;
            csc_sub_u32_out_values[out_idx] = 0u - b_val;
            out_idx = out_idx + 1;
            j = j + 1;
        } else {
            csc_sub_u32_out_row_indices[out_idx] = a_row;
            csc_sub_u32_out_values[out_idx] = a_val - b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }

    while (i < a_end) {
        csc_sub_u32_out_row_indices[out_idx] = csc_sub_u32_a_row_indices[i];
        csc_sub_u32_out_values[out_idx] = csc_sub_u32_a_values[i];
        out_idx = out_idx + 1;
        i = i + 1;
    }

    while (j < b_end) {
        csc_sub_u32_out_row_indices[out_idx] = csc_sub_u32_b_row_indices[j];
        csc_sub_u32_out_values[out_idx] = 0u - csc_sub_u32_b_values[j];
        out_idx = out_idx + 1;
        j = j + 1;
    }
}

// ============================================================================
// csc_mul_compute_u32  (intersection semantics)
// ============================================================================

struct CscMulU32Params {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> csc_mul_u32_a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csc_mul_u32_a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csc_mul_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csc_mul_u32_b_col_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csc_mul_u32_b_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csc_mul_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csc_mul_u32_out_col_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csc_mul_u32_out_row_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csc_mul_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csc_mul_u32_params: CscMulU32Params;

@compute @workgroup_size(256)
fn csc_mul_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= csc_mul_u32_params.ncols) {
        return;
    }

    let a_start = csc_mul_u32_a_col_ptrs[col];
    let a_end = csc_mul_u32_a_col_ptrs[col + 1u];
    let b_start = csc_mul_u32_b_col_ptrs[col];
    let b_end = csc_mul_u32_b_col_ptrs[col + 1u];

    var out_idx = csc_mul_u32_out_col_ptrs[col];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = csc_mul_u32_a_row_indices[i];
        let b_row = csc_mul_u32_b_row_indices[j];

        if (a_row < b_row) {
            i = i + 1;
        } else if (a_row > b_row) {
            j = j + 1;
        } else {
            let a_val = csc_mul_u32_a_values[i];
            let b_val = csc_mul_u32_b_values[j];
            csc_mul_u32_out_row_indices[out_idx] = a_row;
            csc_mul_u32_out_values[out_idx] = a_val * b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }
}

// ============================================================================
// csc_div_compute_u32  (intersection semantics)
// ============================================================================

struct CscDivU32Params {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> csc_div_u32_a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csc_div_u32_a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csc_div_u32_a_values: array<u32>;
@group(0) @binding(3) var<storage, read> csc_div_u32_b_col_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> csc_div_u32_b_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read> csc_div_u32_b_values: array<u32>;
@group(0) @binding(6) var<storage, read> csc_div_u32_out_col_ptrs: array<i32>;
@group(0) @binding(7) var<storage, read_write> csc_div_u32_out_row_indices: array<i32>;
@group(0) @binding(8) var<storage, read_write> csc_div_u32_out_values: array<u32>;
@group(0) @binding(9) var<uniform> csc_div_u32_params: CscDivU32Params;

@compute @workgroup_size(256)
fn csc_div_compute_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= csc_div_u32_params.ncols) {
        return;
    }

    let a_start = csc_div_u32_a_col_ptrs[col];
    let a_end = csc_div_u32_a_col_ptrs[col + 1u];
    let b_start = csc_div_u32_b_col_ptrs[col];
    let b_end = csc_div_u32_b_col_ptrs[col + 1u];

    var out_idx = csc_div_u32_out_col_ptrs[col];
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = csc_div_u32_a_row_indices[i];
        let b_row = csc_div_u32_b_row_indices[j];

        if (a_row < b_row) {
            i = i + 1;
        } else if (a_row > b_row) {
            j = j + 1;
        } else {
            let a_val = csc_div_u32_a_values[i];
            let b_val = csc_div_u32_b_values[j];
            csc_div_u32_out_row_indices[out_idx] = a_row;
            csc_div_u32_out_values[out_idx] = a_val / b_val;
            out_idx = out_idx + 1;
            i = i + 1;
            j = j + 1;
        }
    }
}
