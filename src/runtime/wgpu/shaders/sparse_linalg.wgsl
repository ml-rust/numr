// Sparse linear algebra WGSL shaders
//
// Primitive operations for Gilbert-Peierls sparse LU factorization:
// - scatter: Copy sparse column into dense work vector
// - sparse_axpy: work[indices] -= scale * values
// - gather_clear: Extract values from work vector and clear
// - divide_pivot: Divide work elements by pivot value
// - clear: Zero out specific indices in work vector
//
// All kernels support offset+count for processing subsets of pre-allocated buffers.

// ============================================================================
// Scatter Kernel - f32 (with offset support)
// work[row_indices[offset + i]] = values[offset + i] for i in [0, count)
// ============================================================================

struct ScatterParams {
    offset: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> scatter_params: ScatterParams;
@group(0) @binding(1) var<storage, read> scatter_values_f32: array<f32>;
@group(0) @binding(2) var<storage, read> scatter_row_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> scatter_work_f32: array<f32>;

@compute @workgroup_size(256)
fn sparse_scatter_offset_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= scatter_params.count) {
        return;
    }

    let global_idx = scatter_params.offset + idx;
    let row = scatter_row_indices[global_idx];
    scatter_work_f32[row] = scatter_values_f32[global_idx];
}

// Legacy scatter without offset (for backwards compatibility)
@group(0) @binding(0) var<storage, read> scatter_values_legacy: array<f32>;
@group(0) @binding(1) var<storage, read> scatter_indices_legacy: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_work_legacy: array<f32>;

@compute @workgroup_size(256)
fn sparse_scatter_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nnz = arrayLength(&scatter_values_legacy);
    if (idx >= nnz) {
        return;
    }

    let row = scatter_indices_legacy[idx];
    scatter_work_legacy[row] = scatter_values_legacy[idx];
}

// ============================================================================
// Sparse AXPY Kernel - f32 (with offset support)
// work[row_indices[offset + i]] -= scale * values[offset + i]
// ============================================================================

struct AxpyParams {
    scale: f32,
    offset: u32,
    count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> axpy_params: AxpyParams;
@group(0) @binding(1) var<storage, read> axpy_values_f32: array<f32>;
@group(0) @binding(2) var<storage, read> axpy_row_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> axpy_work_f32: array<f32>;

@compute @workgroup_size(256)
fn sparse_axpy_offset_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= axpy_params.count) {
        return;
    }

    let global_idx = axpy_params.offset + idx;
    let row = axpy_row_indices[global_idx];
    // Note: This is NOT atomic - for single-column processing only
    axpy_work_f32[row] = axpy_work_f32[row] - axpy_params.scale * axpy_values_f32[global_idx];
}

// Legacy AXPY without offset
struct AxpyParamsLegacy {
    scale: f32,
    nnz: u32,
}

@group(0) @binding(0) var<uniform> axpy_params_legacy: AxpyParamsLegacy;
@group(0) @binding(1) var<storage, read> axpy_values_legacy: array<f32>;
@group(0) @binding(2) var<storage, read> axpy_indices_legacy: array<i32>;
@group(0) @binding(3) var<storage, read_write> axpy_work_legacy: array<f32>;

@compute @workgroup_size(256)
fn sparse_axpy_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= axpy_params_legacy.nnz) {
        return;
    }

    let row = axpy_indices_legacy[idx];
    axpy_work_legacy[row] = axpy_work_legacy[row] - axpy_params_legacy.scale * axpy_values_legacy[idx];
}

// ============================================================================
// Gather and Clear Kernel - f32 (with offset support)
// output[offset + i] = work[row_indices[offset + i]], then work[...] = 0
// ============================================================================

struct GatherParams {
    offset: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> gather_params: GatherParams;
@group(0) @binding(1) var<storage, read_write> gather_work_f32: array<f32>;
@group(0) @binding(2) var<storage, read> gather_row_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> gather_output_f32: array<f32>;

@compute @workgroup_size(256)
fn sparse_gather_clear_offset_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= gather_params.count) {
        return;
    }

    let global_idx = gather_params.offset + idx;
    let row = gather_row_indices[global_idx];
    gather_output_f32[global_idx] = gather_work_f32[row];
    gather_work_f32[row] = 0.0;
}

// Legacy gather_clear without offset
@group(0) @binding(0) var<storage, read_write> gather_work_legacy: array<f32>;
@group(0) @binding(1) var<storage, read> gather_indices_legacy: array<i32>;
@group(0) @binding(2) var<storage, read_write> gather_output_legacy: array<f32>;

@compute @workgroup_size(256)
fn sparse_gather_clear_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nnz = arrayLength(&gather_indices_legacy);
    if (idx >= nnz) {
        return;
    }

    let row = gather_indices_legacy[idx];
    gather_output_legacy[idx] = gather_work_legacy[row];
    gather_work_legacy[row] = 0.0;
}

// ============================================================================
// Divide by Pivot Kernel - f32 (with offset support)
// output[offset + i] *= inv_pivot
// ============================================================================

struct DivideParams {
    inv_pivot: f32,
    offset: u32,
    count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> divide_params: DivideParams;
@group(0) @binding(1) var<storage, read_write> divide_output_f32: array<f32>;

@compute @workgroup_size(256)
fn sparse_divide_pivot_offset_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= divide_params.count) {
        return;
    }

    let global_idx = divide_params.offset + idx;
    divide_output_f32[global_idx] = divide_output_f32[global_idx] * divide_params.inv_pivot;
}

// Legacy divide_pivot without offset
struct DivideParamsLegacy {
    inv_pivot: f32,
    nnz: u32,
}

@group(0) @binding(0) var<uniform> divide_params_legacy: DivideParamsLegacy;
@group(0) @binding(1) var<storage, read_write> divide_work_legacy: array<f32>;
@group(0) @binding(2) var<storage, read> divide_indices_legacy: array<i32>;

@compute @workgroup_size(256)
fn sparse_divide_pivot_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= divide_params_legacy.nnz) {
        return;
    }

    let row = divide_indices_legacy[idx];
    divide_work_legacy[row] = divide_work_legacy[row] * divide_params_legacy.inv_pivot;
}

// ============================================================================
// Clear Kernel - f32 (with offset support)
// work[row_indices[offset + i]] = 0
// ============================================================================

struct ClearParams {
    offset: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> clear_params: ClearParams;
@group(0) @binding(1) var<storage, read_write> clear_work_f32: array<f32>;
@group(0) @binding(2) var<storage, read> clear_row_indices: array<i32>;

@compute @workgroup_size(256)
fn sparse_clear_offset_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= clear_params.count) {
        return;
    }

    let global_idx = clear_params.offset + idx;
    let row = clear_row_indices[global_idx];
    clear_work_f32[row] = 0.0;
}

// Legacy clear without offset
@group(0) @binding(0) var<storage, read_write> clear_work_legacy: array<f32>;
@group(0) @binding(1) var<storage, read> clear_indices_legacy: array<i32>;

@compute @workgroup_size(256)
fn sparse_clear_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nnz = arrayLength(&clear_indices_legacy);
    if (idx >= nnz) {
        return;
    }

    let row = clear_indices_legacy[idx];
    clear_work_legacy[row] = 0.0;
}

// ============================================================================
// CSC Triangular Solve Kernels (for LU solve with CSC factors)
// ============================================================================

// CSC Lower Triangular Solve - Forward Substitution
// For each column j in the level:
// x[j] = b[j] / L[j,j] (if not unit diagonal), then b[i] -= L[i,j] * x[j] for i > j
struct TrsvCscLowerParams {
    level_offset: u32,  // Offset into level_cols array
    level_size: u32,    // Number of columns in this level
    n: u32,
    unit_diagonal: u32, // 1 = unit diagonal, 0 = use diagonal from matrix
}

@group(0) @binding(0) var<uniform> trsv_lower_params: TrsvCscLowerParams;
@group(0) @binding(1) var<storage, read> trsv_lower_level_cols: array<i32>;
@group(0) @binding(2) var<storage, read> trsv_lower_col_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> trsv_lower_row_indices: array<i32>;
@group(0) @binding(4) var<storage, read> trsv_lower_values: array<f32>;
@group(0) @binding(5) var<storage, read> trsv_lower_diag_ptr: array<i32>;
@group(0) @binding(6) var<storage, read_write> trsv_lower_b: array<f32>;

@compute @workgroup_size(256)
fn sparse_trsv_csc_lower_level_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= trsv_lower_params.level_size) {
        return;
    }

    let col_idx = trsv_lower_params.level_offset + tid;
    let col = trsv_lower_level_cols[col_idx];
    let col_start = trsv_lower_col_ptrs[col];
    let col_end = trsv_lower_col_ptrs[col + 1];

    // Get diagonal value
    var diag: f32 = 1.0;
    if (trsv_lower_params.unit_diagonal == 0u) {
        let diag_idx = trsv_lower_diag_ptr[col];
        if (diag_idx >= 0) {
            diag = trsv_lower_values[diag_idx];
        }
    }

    // x[col] = b[col] / L[col,col]
    var x_col = trsv_lower_b[col];
    if (trsv_lower_params.unit_diagonal == 0u && abs(diag) > 1e-15) {
        x_col = x_col / diag;
    }
    trsv_lower_b[col] = x_col;

    // Update b[row] for rows below diagonal
    // Note: This uses non-atomic operations which is safe because
    // columns in the same level don't affect each other's diagonal rows
    for (var idx = col_start; idx < col_end; idx = idx + 1) {
        let row = trsv_lower_row_indices[idx];
        if (row > col) {
            // Atomic subtraction for thread safety
            // WGSL doesn't have atomicAdd for f32, so we use a workaround
            // For correctness, we rely on level scheduling ensuring no conflicts
            let old_val = trsv_lower_b[row];
            trsv_lower_b[row] = old_val - trsv_lower_values[idx] * x_col;
        }
    }
}

// CSC Upper Triangular Solve - Backward Substitution
// For each column j in the level:
// x[j] = b[j] / U[j,j], then b[i] -= U[i,j] * x[j] for i < j
struct TrsvCscUpperParams {
    level_offset: u32,
    level_size: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> trsv_upper_params: TrsvCscUpperParams;
@group(0) @binding(1) var<storage, read> trsv_upper_level_cols: array<i32>;
@group(0) @binding(2) var<storage, read> trsv_upper_col_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> trsv_upper_row_indices: array<i32>;
@group(0) @binding(4) var<storage, read> trsv_upper_values: array<f32>;
@group(0) @binding(5) var<storage, read> trsv_upper_diag_ptr: array<i32>;
@group(0) @binding(6) var<storage, read_write> trsv_upper_b: array<f32>;

@compute @workgroup_size(256)
fn sparse_trsv_csc_upper_level_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= trsv_upper_params.level_size) {
        return;
    }

    let col_idx = trsv_upper_params.level_offset + tid;
    let col = trsv_upper_level_cols[col_idx];
    let col_start = trsv_upper_col_ptrs[col];
    let col_end = trsv_upper_col_ptrs[col + 1];

    // Get diagonal value
    let diag_idx = trsv_upper_diag_ptr[col];
    var diag: f32 = 1.0;
    if (diag_idx >= 0) {
        diag = trsv_upper_values[diag_idx];
    }

    // x[col] = b[col] / U[col,col]
    var x_col = trsv_upper_b[col];
    if (abs(diag) > 1e-15) {
        x_col = x_col / diag;
    }
    trsv_upper_b[col] = x_col;

    // Update b[row] for rows above diagonal
    for (var idx = col_start; idx < col_end; idx = idx + 1) {
        let row = trsv_upper_row_indices[idx];
        if (row < col) {
            let old_val = trsv_upper_b[row];
            trsv_upper_b[row] = old_val - trsv_upper_values[idx] * x_col;
        }
    }
}

// Find diagonal indices in CSC matrix
struct FindDiagCscParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> find_diag_csc_params: FindDiagCscParams;
@group(0) @binding(1) var<storage, read> find_diag_csc_col_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> find_diag_csc_row_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> find_diag_csc_diag_ptr: array<i32>;

@compute @workgroup_size(256)
fn find_diag_indices_csc_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= find_diag_csc_params.n) {
        return;
    }

    let start = find_diag_csc_col_ptrs[col];
    let end = find_diag_csc_col_ptrs[col + 1u];

    find_diag_csc_diag_ptr[col] = -1; // Default: no diagonal found

    for (var idx = start; idx < end; idx = idx + 1) {
        if (find_diag_csc_row_indices[idx] == i32(col)) {
            find_diag_csc_diag_ptr[col] = idx;
            break;
        }
    }
}

// Apply row permutation: y[i] = b[perm[i]]
struct ApplyPermParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> apply_perm_params: ApplyPermParams;
@group(0) @binding(1) var<storage, read> apply_perm_b: array<f32>;
@group(0) @binding(2) var<storage, read> apply_perm_perm: array<i32>;
@group(0) @binding(3) var<storage, read_write> apply_perm_y: array<f32>;

@compute @workgroup_size(256)
fn apply_row_perm_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= apply_perm_params.n) {
        return;
    }

    let src_idx = apply_perm_perm[i];
    apply_perm_y[i] = apply_perm_b[src_idx];
}

// ============================================================================
// Find Pivot Kernel - f32
// Parallel reduction to find max |work[start..end]|
// Uses two-phase approach: first find local maxes, then reduce
// ============================================================================

struct PivotParams {
    start: u32,
    end: u32,
}

@group(0) @binding(0) var<uniform> pivot_params: PivotParams;
@group(0) @binding(1) var<storage, read> pivot_work_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> pivot_max_vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> pivot_max_indices: array<u32>;

var<workgroup> shared_vals: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;

@compute @workgroup_size(256)
fn sparse_find_pivot_f32(@builtin(global_invocation_id) gid: vec3<u32>,
                         @builtin(local_invocation_id) lid: vec3<u32>,
                         @builtin(workgroup_id) wid: vec3<u32>) {
    let global_idx = pivot_params.start + gid.x;
    let local_idx = lid.x;

    // Load into shared memory
    if (global_idx < pivot_params.end) {
        shared_vals[local_idx] = abs(pivot_work_f32[global_idx]);
        shared_indices[local_idx] = global_idx;
    } else {
        shared_vals[local_idx] = 0.0;
        shared_indices[local_idx] = pivot_params.start;
    }
    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (local_idx < s) {
            if (shared_vals[local_idx + s] > shared_vals[local_idx]) {
                shared_vals[local_idx] = shared_vals[local_idx + s];
                shared_indices[local_idx] = shared_indices[local_idx + s];
            }
        }
        workgroupBarrier();
    }

    // Write workgroup result
    if (local_idx == 0u) {
        pivot_max_vals[wid.x] = shared_vals[0];
        pivot_max_indices[wid.x] = shared_indices[0];
    }
}

// ============================================================================
// Swap Rows Kernel - f32
// Swaps work[row_a] <-> work[row_b] and perm[row_a] <-> perm[row_b]
// Single-threaded operation (small data)
// ============================================================================

struct SwapParams {
    row_a: u32,
    row_b: u32,
}

@group(0) @binding(0) var<uniform> swap_params: SwapParams;
@group(0) @binding(1) var<storage, read_write> swap_work_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> swap_perm: array<u32>;

@compute @workgroup_size(1)
fn sparse_swap_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (swap_params.row_a != swap_params.row_b) {
        let tmp_work = swap_work_f32[swap_params.row_a];
        swap_work_f32[swap_params.row_a] = swap_work_f32[swap_params.row_b];
        swap_work_f32[swap_params.row_b] = tmp_work;

        let tmp_perm = swap_perm[swap_params.row_a];
        swap_perm[swap_params.row_a] = swap_perm[swap_params.row_b];
        swap_perm[swap_params.row_b] = tmp_perm;
    }
}
