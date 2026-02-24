// Sparse format conversion shaders - index-only (type-independent)
//
// expand_row_ptrs: CSR row pointers -> explicit row indices
// expand_col_ptrs: CSC col pointers -> explicit col indices
// histogram:       count elements per bucket
// copy_ptrs:       copy a pointer array

// ============================================================================
// expand_row_ptrs
// ============================================================================

struct ExpandRowParams {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> erp_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> erp_row_indices: array<i32>;
@group(0) @binding(2) var<uniform> erp_params: ExpandRowParams;

@compute @workgroup_size(256)
fn expand_row_ptrs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= erp_params.nrows) {
        return;
    }

    let start = erp_row_ptrs[row];
    let end = erp_row_ptrs[row + 1u];

    for (var i = start; i < end; i = i + 1) {
        erp_row_indices[i] = i32(row);
    }
}

// ============================================================================
// expand_col_ptrs
// ============================================================================

struct ExpandColParams {
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> ecp_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> ecp_col_indices: array<i32>;
@group(0) @binding(2) var<uniform> ecp_params: ExpandColParams;

@compute @workgroup_size(256)
fn expand_col_ptrs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= ecp_params.ncols) {
        return;
    }

    let start = ecp_col_ptrs[col];
    let end = ecp_col_ptrs[col + 1u];

    for (var i = start; i < end; i = i + 1) {
        ecp_col_indices[i] = i32(col);
    }
}

// ============================================================================
// histogram
// ============================================================================

struct HistogramParams {
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> hist_indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> hist_counts: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> hist_params: HistogramParams;

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= hist_params.nnz) {
        return;
    }

    let bucket = hist_indices[idx];
    atomicAdd(&hist_counts[bucket], 1);
}

// ============================================================================
// copy_ptrs
// ============================================================================

struct CopyPtrsParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> cp_src: array<i32>;
@group(0) @binding(1) var<storage, read_write> cp_dst: array<i32>;
@group(0) @binding(2) var<uniform> cp_params: CopyPtrsParams;

@compute @workgroup_size(256)
fn copy_ptrs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cp_params.n) {
        return;
    }
    cp_dst[idx] = cp_src[idx];
}
