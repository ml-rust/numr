// CSR Sparse Matrix-Vector Multiplication: y = A * x
// Row-parallel implementation: one thread per row

const WORKGROUP_SIZE: u32 = 256u;

struct SpmvParams {
    nrows: u32,
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
}

// CSR format
@group(0) @binding(0) var<storage, read> spmv_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> spmv_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> spmv_values: array<f32>;
// Dense vector x
@group(0) @binding(3) var<storage, read> spmv_x: array<f32>;
// Output vector y
@group(0) @binding(4) var<storage, read_write> spmv_y: array<f32>;
// Parameters
@group(0) @binding(5) var<uniform> spmv_params: SpmvParams;

@compute @workgroup_size(256)
fn csr_spmv_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= spmv_params.nrows) {
        return;
    }

    let row_start = spmv_row_ptrs[row];
    let row_end = spmv_row_ptrs[row + 1u];

    var sum: f32 = 0.0;
    for (var j: i32 = row_start; j < row_end; j = j + 1) {
        let col = spmv_col_indices[j];
        sum = sum + spmv_values[j] * spmv_x[col];
    }

    spmv_y[row] = sum;
}

// CSR Sparse Matrix-Dense Matrix Multiplication: C = A * B
// Each thread computes one output element C[row, col]

struct SpmmParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

// CSR format for A
@group(0) @binding(0) var<storage, read> spmm_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> spmm_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> spmm_a_values: array<f32>;
// Dense matrix B (k x n, row-major)
@group(0) @binding(3) var<storage, read> spmm_b: array<f32>;
// Output matrix C (m x n, row-major)
@group(0) @binding(4) var<storage, read_write> spmm_c: array<f32>;
// Parameters
@group(0) @binding(5) var<uniform> spmm_params: SpmmParams;

@compute @workgroup_size(256)
fn csr_spmm_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = spmm_params.m * spmm_params.n;
    if (idx >= total) {
        return;
    }

    let row = idx / spmm_params.n;
    let col = idx % spmm_params.n;

    let row_start = spmm_row_ptrs[row];
    let row_end = spmm_row_ptrs[row + 1u];

    var sum: f32 = 0.0;
    for (var j: i32 = row_start; j < row_end; j = j + 1) {
        let a_col = spmm_col_indices[j];
        let a_val = spmm_a_values[j];
        let b_idx = u32(a_col) * spmm_params.n + col;
        sum = sum + a_val * spmm_b[b_idx];
    }

    spmm_c[idx] = sum;
}

// CSR Extract Diagonal: diag[i] = A[i,i]
// Thread-per-row: each thread scans one row for col_index == row_index

struct DiagParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> diag_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> diag_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> diag_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> diag_out: array<f32>;
@group(0) @binding(4) var<uniform> diag_params: DiagParams;

@compute @workgroup_size(256)
fn csr_extract_diagonal_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= diag_params.n) {
        return;
    }

    let row_start = diag_row_ptrs[row];
    let row_end = diag_row_ptrs[row + 1u];

    var val: f32 = 0.0;
    for (var j: i32 = row_start; j < row_end; j = j + 1) {
        if (diag_col_indices[j] == i32(row)) {
            val = diag_values[j];
            break;
        }
    }

    diag_out[row] = val;
}
