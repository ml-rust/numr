//! WebGPU sparse level computation shaders
//!
//! Computes level schedule for level-synchronous sparse factorization.
//! Iteratively computes dependency levels until convergence.

// ============================================================================
// Cast i64 to i32
// ============================================================================
// WGSL has no i64 support. CSR tensors store indices as i64 (two u32 values).
// This shader reads the low 32 bits of each i64 value (since CSR indices fit in i32).

@group(0) @binding(0) var<storage, read> input_i64: array<u32>;   // Pairs of u32
@group(0) @binding(1) var<storage, read_write> output_i32: array<i32>;

@compute @workgroup_size(256)
fn cast_i64_to_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output_i32)) { return; }
    // i64 stored as two u32: low word at 2*idx, high word at 2*idx+1
    // We only care about low 32 bits (all indices fit in i32 range)
    output_i32[idx] = i32(input_i64[2u * idx]);
}

// ============================================================================
// Compute levels for lower triangular (forward dependencies)
// ============================================================================
// For lower triangular: level[i] = max(level[j] + 1) for all j < i
// where A[i,j] is nonzero

@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;        // [n+1]
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;     // [nnz]
@group(0) @binding(2) var<storage, read_write> levels: array<atomic<i32>>;  // [n]
@group(0) @binding(3) var<storage, read_write> changed: array<atomic<u32>>;  // [1] = flag

struct Params {
    n: u32,
    iteration: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_levels_lower_iter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    var max_level: i32 = -1;

    // Scan all nonzeros in row i
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1) {
        let j = col_indices[idx];
        if (j < i32(i)) {  // j < i (lower triangle)
            let j_level = atomicLoad(&levels[u32(j)]);
            if (j_level + 1 > max_level) {
                max_level = j_level + 1;
            }
        }
    }

    // Update level[i] if it increased
    if (max_level > 0) {
        let old_level = atomicExchange(&levels[i], max_level);
        if (max_level > old_level) {
            atomicStore(&changed[0], 1u);
        }
    }
}

// ============================================================================
// Compute levels for upper triangular (backward dependencies)
// ============================================================================
// For upper triangular: level[i] = max(level[j] + 1) for all j > i
// where A[i,j] is nonzero

@compute @workgroup_size(256)
fn compute_levels_upper_iter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    var max_level: i32 = -1;

    // Scan all nonzeros in row i
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1) {
        let j = col_indices[idx];
        if (j > i32(i)) {  // j > i (upper triangle)
            let j_level = atomicLoad(&levels[u32(j)]);
            if (j_level + 1 > max_level) {
                max_level = j_level + 1;
            }
        }
    }

    // Update level[i] if it increased
    if (max_level > 0) {
        let old_level = atomicExchange(&levels[i], max_level);
        if (max_level > old_level) {
            atomicStore(&changed[0], 1u);
        }
    }
}

// ============================================================================
// Compute levels for ILU (all dependencies)
// ============================================================================
// For ILU: level[i] = max(level[j] + 1) for all j < i where A[i,j] is nonzero

@compute @workgroup_size(256)
fn compute_levels_ilu_iter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    var max_level: i32 = -1;

    // Scan all nonzeros in row i
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1) {
        let j = col_indices[idx];
        if (j < i32(i)) {  // j < i (strict lower part)
            let j_level = atomicLoad(&levels[u32(j)]);
            if (j_level + 1 > max_level) {
                max_level = j_level + 1;
            }
        }
    }

    // Update level[i] if it increased
    if (max_level > 0) {
        let old_level = atomicExchange(&levels[i], max_level);
        if (max_level > old_level) {
            atomicStore(&changed[0], 1u);
        }
    }
}

// ============================================================================
// Histogram levels (count rows per level)
// ============================================================================

@group(0) @binding(0) var<storage, read> levels: array<i32>;          // [n]
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;  // [max_level+1]

struct HistogramParams {
    n: u32,
    max_level: u32,
}
@group(0) @binding(2) var<uniform> hist_params: HistogramParams;

@compute @workgroup_size(256)
fn histogram_levels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= hist_params.n) { return; }

    let level = levels[i];
    if (level >= 0 && u32(level) <= hist_params.max_level) {
        atomicAdd(&histogram[u32(level)], 1u);
    }
}

// ============================================================================
// Scatter rows into level_rows array
// ============================================================================

@group(0) @binding(0) var<storage, read> levels: array<i32>;          // [n]
@group(0) @binding(1) var<storage, read> level_ptrs: array<u32>;      // [num_levels+1] prefix sum
@group(0) @binding(2) var<storage, read_write> level_offsets: array<atomic<u32>>;  // [num_levels]
@group(0) @binding(3) var<storage, read_write> level_rows: array<u32>; // [n] output

struct ScatterParams {
    n: u32,
    num_levels: u32,
}
@group(0) @binding(4) var<uniform> scatter_params: ScatterParams;

@compute @workgroup_size(256)
fn scatter_by_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= scatter_params.n) { return; }

    let level = levels[i];
    if (level >= 0 && u32(level) < scatter_params.num_levels) {
        let pos = atomicAdd(&level_offsets[u32(level)], 1u);
        let row_start = level_ptrs[u32(level)];
        level_rows[row_start + pos] = i;
    }
}
