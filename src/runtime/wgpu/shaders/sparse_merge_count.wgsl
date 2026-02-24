// Sparse merge count shaders - type-independent
//
// csr_merge_count:   Count output NNZ per row for CSR add/sub (union semantics)
// csr_mul_count:     Count output NNZ per row for CSR mul/div (intersection semantics)
// csc_merge_count:   Count output NNZ per col for CSC add/sub (union semantics)
// csc_mul_count:     Count output NNZ per col for CSC mul/div (intersection semantics)
// exclusive_scan_i32: Sequential exclusive prefix sum

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// csr_merge_count
// ============================================================================

struct CsrMergeCountParams {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> cmc_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> cmc_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> cmc_b_row_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> cmc_b_col_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> cmc_row_counts: array<i32>;
@group(0) @binding(5) var<uniform> cmc_params: CsrMergeCountParams;

@compute @workgroup_size(256)
fn csr_merge_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= cmc_params.nrows) {
        return;
    }

    let a_start = cmc_a_row_ptrs[row];
    let a_end = cmc_a_row_ptrs[row + 1u];
    let b_start = cmc_b_row_ptrs[row];
    let b_end = cmc_b_row_ptrs[row + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    // Merge sorted column indices, count unique columns
    while (i < a_end && j < b_end) {
        let a_col = cmc_a_col_indices[i];
        let b_col = cmc_b_col_indices[j];

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

    cmc_row_counts[row] = count;
}

// ============================================================================
// csr_mul_count
// ============================================================================

struct CsrMulCountParams {
    nrows: u32,
}

@group(0) @binding(0) var<storage, read> cmmc_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> cmmc_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> cmmc_b_row_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> cmmc_b_col_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> cmmc_row_counts: array<i32>;
@group(0) @binding(5) var<uniform> cmmc_params: CsrMulCountParams;

@compute @workgroup_size(256)
fn csr_mul_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= cmmc_params.nrows) {
        return;
    }

    let a_start = cmmc_a_row_ptrs[row];
    let a_end = cmmc_a_row_ptrs[row + 1u];
    let b_start = cmmc_b_row_ptrs[row];
    let b_end = cmmc_b_row_ptrs[row + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    // Count matching column indices only (intersection)
    while (i < a_end && j < b_end) {
        let a_col = cmmc_a_col_indices[i];
        let b_col = cmmc_b_col_indices[j];

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

    cmmc_row_counts[row] = count;
}

// ============================================================================
// csc_merge_count
// ============================================================================

struct CscMergeCountParams {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> csmc_a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csmc_a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csmc_b_col_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> csmc_b_row_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> csmc_col_counts: array<i32>;
@group(0) @binding(5) var<uniform> csmc_params: CscMergeCountParams;

@compute @workgroup_size(256)
fn csc_merge_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= csmc_params.ncols) {
        return;
    }

    let a_start = csmc_a_col_ptrs[col];
    let a_end = csmc_a_col_ptrs[col + 1u];
    let b_start = csmc_b_col_ptrs[col];
    let b_end = csmc_b_col_ptrs[col + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = csmc_a_row_indices[i];
        let b_row = csmc_b_row_indices[j];

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

    csmc_col_counts[col] = count;
}

// ============================================================================
// csc_mul_count
// ============================================================================

struct CscMulCountParams {
    ncols: u32,
}

@group(0) @binding(0) var<storage, read> csmmc_a_col_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> csmmc_a_row_indices: array<i32>;
@group(0) @binding(2) var<storage, read> csmmc_b_col_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> csmmc_b_row_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> csmmc_col_counts: array<i32>;
@group(0) @binding(5) var<uniform> csmmc_params: CscMulCountParams;

@compute @workgroup_size(256)
fn csc_mul_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= csmmc_params.ncols) {
        return;
    }

    let a_start = csmmc_a_col_ptrs[col];
    let a_end = csmmc_a_col_ptrs[col + 1u];
    let b_start = csmmc_b_col_ptrs[col];
    let b_end = csmmc_b_col_ptrs[col + 1u];

    var count: i32 = 0;
    var i: i32 = a_start;
    var j: i32 = b_start;

    while (i < a_end && j < b_end) {
        let a_row = csmmc_a_row_indices[i];
        let b_row = csmmc_b_row_indices[j];

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

    csmmc_col_counts[col] = count;
}

// ============================================================================
// exclusive_scan_i32
// ============================================================================

struct ScanParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> scan_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> scan_output: array<i32>;
@group(0) @binding(2) var<uniform> scan_params: ScanParams;

// Sequential exclusive scan - only first thread does work
@compute @workgroup_size(1)
fn exclusive_scan_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    var sum: i32 = 0;
    for (var i: u32 = 0u; i < scan_params.n; i = i + 1u) {
        let val = scan_input[i];
        scan_output[i] = sum;
        sum = sum + val;
    }
    // Final element is total sum
    scan_output[scan_params.n] = sum;
}
