// Sparse LU split and lower triangle extraction shaders - F32
//
// split_lu_count:          Count L and U non-zeros per row
// split_lu_scatter_l_f32:  Scatter values into L matrix (lower triangle)
// split_lu_scatter_u_f32:  Scatter values into U matrix (upper triangle + diagonal)
// extract_lower_count:     Count lower triangle non-zeros per row
// extract_lower_scatter_f32: Scatter lower triangle values

// ============================================================================
// split_lu_count
// ============================================================================

struct SplitLuCountParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> slc_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> slc_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> slc_l_counts: array<i32>;
@group(0) @binding(3) var<storage, read_write> slc_u_counts: array<i32>;
@group(0) @binding(4) var<uniform> slc_params: SplitLuCountParams;

@compute @workgroup_size(256)
fn split_lu_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= slc_params.n) {
        return;
    }

    let start = slc_row_ptrs[row];
    let end = slc_row_ptrs[row + 1];

    var l_count = 0i;
    var u_count = 0i;

    for (var idx = start; idx < end; idx = idx + 1) {
        let col = slc_col_indices[idx];
        if (col < row) {
            l_count = l_count + 1;
        } else {
            u_count = u_count + 1;
        }
    }

    slc_l_counts[row] = l_count;
    slc_u_counts[row] = u_count;
}

// ============================================================================
// split_lu_scatter_l_f32
// ============================================================================

struct SplitLuScatterLParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> sll_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> sll_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> sll_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> sll_l_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read_write> sll_l_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> sll_l_values: array<f32>;
@group(0) @binding(6) var<uniform> sll_params: SplitLuScatterLParams;

@compute @workgroup_size(256)
fn split_lu_scatter_l_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= sll_params.n) {
        return;
    }

    let src_start = sll_row_ptrs[row];
    let src_end = sll_row_ptrs[row + 1];
    var l_write_pos = sll_l_row_ptrs[row];

    for (var idx = src_start; idx < src_end; idx = idx + 1) {
        let col = sll_col_indices[idx];
        if (col < row) {
            sll_l_col_indices[l_write_pos] = col;
            sll_l_values[l_write_pos] = sll_values[idx];
            l_write_pos = l_write_pos + 1;
        }
    }
}

// ============================================================================
// split_lu_scatter_u_f32
// ============================================================================

struct SplitLuScatterUParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> slu_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> slu_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> slu_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> slu_u_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read_write> slu_u_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> slu_u_values: array<f32>;
@group(0) @binding(6) var<uniform> slu_params: SplitLuScatterUParams;

@compute @workgroup_size(256)
fn split_lu_scatter_u_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= slu_params.n) {
        return;
    }

    let src_start = slu_row_ptrs[row];
    let src_end = slu_row_ptrs[row + 1];
    var u_write_pos = slu_u_row_ptrs[row];

    for (var idx = src_start; idx < src_end; idx = idx + 1) {
        let col = slu_col_indices[idx];
        if (col >= row) {
            slu_u_col_indices[u_write_pos] = col;
            slu_u_values[u_write_pos] = slu_values[idx];
            u_write_pos = u_write_pos + 1;
        }
    }
}

// ============================================================================
// extract_lower_count
// ============================================================================

struct ExtractLowerCountParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> elc_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> elc_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> elc_l_counts: array<i32>;
@group(0) @binding(3) var<uniform> elc_params: ExtractLowerCountParams;

@compute @workgroup_size(256)
fn extract_lower_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= elc_params.n) {
        return;
    }

    let start = elc_row_ptrs[row];
    let end = elc_row_ptrs[row + 1];

    var count = 0i;

    for (var idx = start; idx < end; idx = idx + 1) {
        let col = elc_col_indices[idx];
        if (col <= row) {
            count = count + 1;
        }
    }

    elc_l_counts[row] = count;
}

// ============================================================================
// extract_lower_scatter_f32
// ============================================================================

struct ExtractLowerScatterParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Note: All buffers use read_write due to LayoutKey-based pipeline layout
@group(0) @binding(0) var<storage, read_write> els_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> els_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> els_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> els_l_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read_write> els_l_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read_write> els_l_values: array<f32>;
@group(0) @binding(6) var<uniform> els_params: ExtractLowerScatterParams;

@compute @workgroup_size(256)
fn extract_lower_scatter_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= els_params.n) {
        return;
    }

    let src_start = els_row_ptrs[row];
    let src_end = els_row_ptrs[row + 1];

    var write_pos = els_l_row_ptrs[row];

    for (var idx = src_start; idx < src_end; idx = idx + 1) {
        let col = els_col_indices[idx];
        if (col <= row) {
            els_l_col_indices[write_pos] = col;
            els_l_values[write_pos] = els_values[idx];
            write_pos = write_pos + 1;
        }
    }
}
