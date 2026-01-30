//! Matrix utilities: determinant, permutation, column ops, rank helpers, copy
//!
//! F32 only - WGSL doesn't support F64.

/// Utilities shader: det, permutation, scatter/extract column, max_abs, count, copy
#[allow(dead_code)]
pub const UTILITIES_SHADER: &str = r#"
const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Determinant from LU
// ============================================================================

struct DetParams {
    n: u32,
    num_swaps: i32,
}

@group(0) @binding(0) var<storage, read_write> det_lu: array<f32>;
@group(0) @binding(1) var<storage, read_write> det_output: array<f32>;
@group(0) @binding(2) var<uniform> det_params: DetParams;

@compute @workgroup_size(1)
fn det_from_lu_f32() {
    let n = det_params.n;
    let num_swaps = det_params.num_swaps;

    var det: f32 = 1.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        det = det * det_lu[i * n + i];
    }

    // Apply sign based on number of row swaps
    if ((num_swaps % 2) != 0) {
        det = -det;
    }

    det_output[0] = det;
}

// ============================================================================
// Apply LU Permutation
// ============================================================================

struct PermParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> perm_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> perm_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> perm_pivots: array<i32>;
@group(0) @binding(3) var<uniform> perm_params: PermParams;

@compute @workgroup_size(1)
fn apply_lu_permutation_f32() {
    let n = perm_params.n;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        perm_output[i] = perm_input[i];
    }

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let pivot = u32(perm_pivots[i]);
        if (pivot != i) {
            let tmp = perm_output[i];
            perm_output[i] = perm_output[pivot];
            perm_output[pivot] = tmp;
        }
    }
}

// ============================================================================
// Scatter Column
// ============================================================================

struct ScatterParams {
    n: u32,
    col: u32,
}

@group(0) @binding(0) var<storage, read_write> scatter_vec: array<f32>;
@group(0) @binding(1) var<storage, read_write> scatter_matrix: array<f32>;
@group(0) @binding(2) var<uniform> scatter_params: ScatterParams;

@compute @workgroup_size(256)
fn scatter_column_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let n = scatter_params.n;
    let col = scatter_params.col;

    if (gid < n) {
        scatter_matrix[gid * n + col] = scatter_vec[gid];
    }
}

// ============================================================================
// Extract Column
// ============================================================================

struct ExtractParams {
    m: u32,
    n_cols: u32,
    col: u32,
}

@group(0) @binding(0) var<storage, read_write> extract_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> extract_vec: array<f32>;
@group(0) @binding(2) var<uniform> extract_params: ExtractParams;

@compute @workgroup_size(256)
fn extract_column_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let m = extract_params.m;
    let n_cols = extract_params.n_cols;
    let col = extract_params.col;

    if (gid < m) {
        extract_vec[gid] = extract_matrix[gid * n_cols + col];
    }
}

// ============================================================================
// Max Absolute Value (for matrix_rank)
// ============================================================================

struct MaxAbsParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> maxabs_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> maxabs_output: array<f32>;
@group(0) @binding(2) var<uniform> maxabs_params: MaxAbsParams;

var<workgroup> maxabs_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn max_abs_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = maxabs_params.n;

    var val: f32 = 0.0;
    if (gid < n) {
        val = abs(maxabs_values[gid]);
    }
    maxabs_shared[tid] = val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            maxabs_shared[tid] = max(maxabs_shared[tid], maxabs_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        maxabs_output[0] = max(maxabs_output[0], maxabs_shared[0]);
    }
}

// ============================================================================
// Count Above Threshold (for matrix_rank)
// ============================================================================

struct CountParams {
    n: u32,
    threshold: f32,
}

@group(0) @binding(0) var<storage, read_write> count_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> count_output: atomic<u32>;
@group(0) @binding(2) var<uniform> count_params: CountParams;

var<workgroup> count_shared: array<u32, 256>;

@compute @workgroup_size(256)
fn count_above_threshold_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = count_params.n;
    let threshold = count_params.threshold;

    var count: u32 = 0u;
    if (gid < n && abs(count_values[gid]) > threshold) {
        count = 1u;
    }
    count_shared[tid] = count;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            count_shared[tid] = count_shared[tid] + count_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        atomicAdd(&count_output, count_shared[0]);
    }
}

// ============================================================================
// Matrix Copy
// ============================================================================

struct CopyParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> copy_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> copy_dst: array<f32>;
@group(0) @binding(2) var<uniform> copy_params: CopyParams;

@compute @workgroup_size(256)
fn matrix_copy_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let n = copy_params.n;

    if (gid < n) {
        copy_dst[gid] = copy_src[gid];
    }
}
"#;
