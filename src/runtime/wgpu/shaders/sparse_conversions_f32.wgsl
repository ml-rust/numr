// Sparse format conversion shaders - F32 typed operations

// ============================================================================
// coo_to_csr_scatter
// ============================================================================

struct ScatterParams {
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> c2r_in_row_indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> c2r_in_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> c2r_in_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> c2r_row_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> c2r_out_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> c2r_out_values: array<f32>;
@group(0) @binding(6) var<uniform> c2r_params: ScatterParams;

@compute @workgroup_size(256)
fn coo_to_csr_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= c2r_params.nnz) {
        return;
    }

    let row = c2r_in_row_indices[idx];
    let col = c2r_in_col_indices[idx];
    let val = c2r_in_values[idx];

    let pos = atomicAdd(&c2r_row_ptrs_atomic[row], 1);

    c2r_out_col_indices[pos] = col;
    c2r_out_values[pos] = val;
}

// ============================================================================
// coo_to_csc_scatter
// ============================================================================

@group(0) @binding(0) var<storage, read_write> c2c_in_row_indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> c2c_in_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> c2c_in_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> c2c_col_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> c2c_out_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> c2c_out_values: array<f32>;
@group(0) @binding(6) var<uniform> c2c_params: ScatterParams;

@compute @workgroup_size(256)
fn coo_to_csc_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= c2c_params.nnz) {
        return;
    }

    let row = c2c_in_row_indices[idx];
    let col = c2c_in_col_indices[idx];
    let val = c2c_in_values[idx];

    let pos = atomicAdd(&c2c_col_ptrs_atomic[col], 1);

    c2c_out_row_indices[pos] = row;
    c2c_out_values[pos] = val;
}

// ============================================================================
// csr_to_csc_scatter (transpose)
// ============================================================================

struct TransposeRowParams {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> r2c_in_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> r2c_in_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> r2c_in_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> r2c_col_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> r2c_out_row_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> r2c_out_values: array<f32>;
@group(0) @binding(6) var<uniform> r2c_params: TransposeRowParams;

@compute @workgroup_size(256)
fn csr_to_csc_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= r2c_params.nrows) {
        return;
    }

    let start = r2c_in_row_ptrs[row];
    let end = r2c_in_row_ptrs[row + 1u];

    for (var i = start; i < end; i = i + 1) {
        let col = r2c_in_col_indices[i];
        let val = r2c_in_values[i];

        let pos = atomicAdd(&r2c_col_ptrs_atomic[col], 1);

        r2c_out_row_indices[pos] = i32(row);
        r2c_out_values[pos] = val;
    }
}

// ============================================================================
// csc_to_csr_scatter (transpose)
// ============================================================================

struct TransposeColParams {
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> c2r2_in_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> c2r2_in_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> c2r2_in_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> c2r2_row_ptrs_atomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> c2r2_out_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> c2r2_out_values: array<f32>;
@group(0) @binding(6) var<uniform> c2r2_params: TransposeColParams;

@compute @workgroup_size(256)
fn csc_to_csr_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= c2r2_params.ncols) {
        return;
    }

    let start = c2r2_in_col_ptrs[col];
    let end = c2r2_in_col_ptrs[col + 1u];

    for (var i = start; i < end; i = i + 1) {
        let row = c2r2_in_row_indices[i];
        let val = c2r2_in_values[i];

        let pos = atomicAdd(&c2r2_row_ptrs_atomic[row], 1);

        c2r2_out_col_indices[pos] = i32(col);
        c2r2_out_values[pos] = val;
    }
}

// ============================================================================
// csr_to_dense
// ============================================================================

struct CsrToDenseParams {
    nrows: u32,
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> ctd_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> ctd_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> ctd_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> ctd_dense: array<f32>;
@group(0) @binding(4) var<uniform> ctd_params: CsrToDenseParams;

@compute @workgroup_size(256)
fn csr_to_dense(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= ctd_params.nrows) {
        return;
    }

    let start = ctd_row_ptrs[row];
    let end = ctd_row_ptrs[row + 1u];
    let ncols = ctd_params.ncols;

    for (var i = start; i < end; i = i + 1) {
        let col = u32(ctd_col_indices[i]);
        ctd_dense[row * ncols + col] = ctd_values[i];
    }
}

// ============================================================================
// count_nonzeros
// ============================================================================

struct CountNzParams {
    total_elems: u32,
    threshold_bits: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> cnz_dense: array<f32>;
@group(0) @binding(1) var<storage, read_write> cnz_count: atomic<u32>;
@group(0) @binding(2) var<uniform> cnz_params: CountNzParams;

@compute @workgroup_size(256)
fn count_nonzeros(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cnz_params.total_elems) {
        return;
    }

    let val = cnz_dense[idx];
    let threshold = bitcast<f32>(cnz_params.threshold_bits);
    let zero_val = f32(0);

    if (abs(val) >= threshold) {
        atomicAdd(&cnz_count, 1u);
    }
}

// ============================================================================
// dense_to_coo_scatter
// ============================================================================

struct DenseToCooParams {
    nrows: u32,
    ncols: u32,
    threshold_bits: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> dtc_dense: array<f32>;
@group(0) @binding(1) var<storage, read_write> dtc_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> dtc_col_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> dtc_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> dtc_write_pos: atomic<u32>;
@group(0) @binding(5) var<uniform> dtc_params: DenseToCooParams;

@compute @workgroup_size(256)
fn dense_to_coo_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dtc_params.nrows * dtc_params.ncols;
    if (idx >= total) {
        return;
    }

    let val = dtc_dense[idx];
    let threshold = bitcast<f32>(dtc_params.threshold_bits);

    if (abs(val) >= threshold) {
        let row = idx / dtc_params.ncols;
        let col = idx % dtc_params.ncols;

        let pos = atomicAdd(&dtc_write_pos, 1u);

        dtc_row_indices[pos] = i32(row);
        dtc_col_indices[pos] = i32(col);
        dtc_values[pos] = val;
    }
}
