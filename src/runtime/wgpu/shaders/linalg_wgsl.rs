//! WGSL shader source code for linear algebra operations
//!
//! All shaders follow the same algorithm as CPU/CUDA implementations
//! to ensure backend parity.

/// Linear algebra shader module source (F32 only - WGSL doesn't support F64)
pub const LINALG_SHADER: &str = r#"
// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Trace - Sum of diagonal elements
// ============================================================================

struct TraceParams {
    n: u32,
    stride: u32,
}

@group(0) @binding(0) var<storage, read_write> trace_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> trace_output: array<f32>;
@group(0) @binding(2) var<uniform> trace_params: TraceParams;

var<workgroup> trace_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn trace_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>,
             @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = trace_params.n;
    let stride = trace_params.stride;

    // Each thread loads one diagonal element
    var val: f32 = 0.0;
    if (gid < n) {
        let idx = gid * stride + gid;
        val = trace_input[idx];
    }
    trace_shared[tid] = val;
    workgroupBarrier();

    // Parallel reduction
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            trace_shared[tid] = trace_shared[tid] + trace_shared[tid + s];
        }
        workgroupBarrier();
    }

    // First thread writes result (atomic add for multi-workgroup)
    if (tid == 0u) {
        // Use atomicAdd for f32 - requires WGSL atomics extension
        // For now, assume single workgroup for small matrices
        trace_output[0] = trace_output[0] + trace_shared[0];
    }
}

// ============================================================================
// Diag - Extract diagonal elements
// ============================================================================

struct DiagParams {
    min_dim: u32,
    n_cols: u32,
}

@group(0) @binding(0) var<storage, read_write> diag_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> diag_output: array<f32>;
@group(0) @binding(2) var<uniform> diag_params: DiagParams;

@compute @workgroup_size(256)
fn diag_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let min_dim = diag_params.min_dim;
    let n_cols = diag_params.n_cols;

    if (gid < min_dim) {
        let idx = gid * n_cols + gid;
        diag_output[gid] = diag_input[idx];
    }
}

// ============================================================================
// Diagflat - Create diagonal matrix from vector
// ============================================================================

struct DiagflatParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> diagflat_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> diagflat_output: array<f32>;
@group(0) @binding(2) var<uniform> diagflat_params: DiagflatParams;

@compute @workgroup_size(256)
fn diagflat_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let n = diagflat_params.n;
    let total = n * n;

    if (gid < total) {
        let row = gid / n;
        let col = gid % n;
        if (row == col) {
            diagflat_output[gid] = diagflat_input[row];
        } else {
            diagflat_output[gid] = 0.0;
        }
    }
}

// ============================================================================
// Create Identity Matrix
// ============================================================================

struct IdentityParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> identity_output: array<f32>;
@group(0) @binding(1) var<uniform> identity_params: IdentityParams;

@compute @workgroup_size(256)
fn create_identity_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let n = identity_params.n;
    let total = n * n;

    if (gid < total) {
        let row = gid / n;
        let col = gid % n;
        if (row == col) {
            identity_output[gid] = 1.0;
        } else {
            identity_output[gid] = 0.0;
        }
    }
}

// ============================================================================
// Forward Substitution - Solve Lx = b
// ============================================================================

struct ForwardSubParams {
    n: u32,
    unit_diagonal: u32,
}

@group(0) @binding(0) var<storage, read_write> forward_l: array<f32>;
@group(0) @binding(1) var<storage, read_write> forward_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> forward_x: array<f32>;
@group(0) @binding(3) var<uniform> forward_params: ForwardSubParams;

@compute @workgroup_size(1)
fn forward_sub_f32() {
    let n = forward_params.n;
    let unit_diag = forward_params.unit_diagonal != 0u;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        var sum: f32 = forward_b[i];

        for (var j: u32 = 0u; j < i; j = j + 1u) {
            let l_idx = i * n + j;
            sum = sum - forward_l[l_idx] * forward_x[j];
        }

        if (unit_diag) {
            forward_x[i] = sum;
        } else {
            let diag_idx = i * n + i;
            forward_x[i] = sum / forward_l[diag_idx];
        }
    }
}

// ============================================================================
// Backward Substitution - Solve Ux = b
// ============================================================================

struct BackwardSubParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> backward_u: array<f32>;
@group(0) @binding(1) var<storage, read_write> backward_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> backward_x: array<f32>;
@group(0) @binding(3) var<uniform> backward_params: BackwardSubParams;

@compute @workgroup_size(1)
fn backward_sub_f32() {
    let n = backward_params.n;

    // Start from last row
    for (var ii: u32 = 0u; ii < n; ii = ii + 1u) {
        let i = n - 1u - ii;

        var sum: f32 = backward_b[i];

        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            let u_idx = i * n + j;
            sum = sum - backward_u[u_idx] * backward_x[j];
        }

        let diag_idx = i * n + i;
        backward_x[i] = sum / backward_u[diag_idx];
    }
}

// ============================================================================
// LU Decomposition with Partial Pivoting
// ============================================================================

struct LuParams {
    m: u32,
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> lu_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> lu_pivots: array<i32>;
@group(0) @binding(2) var<storage, read_write> lu_num_swaps: atomic<i32>;
@group(0) @binding(3) var<storage, read_write> lu_singular_flag: atomic<i32>;
@group(0) @binding(4) var<uniform> lu_params: LuParams;

@compute @workgroup_size(1)
fn lu_decompose_f32() {
    let m = lu_params.m;
    let n = lu_params.n;
    let k = min(m, n);

    var num_swaps: i32 = 0;

    for (var col: u32 = 0u; col < k; col = col + 1u) {
        // Find pivot (max absolute value in column)
        var max_val: f32 = abs(lu_matrix[col * n + col]);
        var max_row: u32 = col;

        for (var row: u32 = col + 1u; row < m; row = row + 1u) {
            let val = abs(lu_matrix[row * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Store pivot
        lu_pivots[col] = i32(max_row);

        // Swap rows if needed
        if (max_row != col) {
            num_swaps = num_swaps + 1;
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                let tmp = lu_matrix[col * n + j];
                lu_matrix[col * n + j] = lu_matrix[max_row * n + j];
                lu_matrix[max_row * n + j] = tmp;
            }
        }

        // Check for singularity
        let pivot = lu_matrix[col * n + col];
        if (abs(pivot) < 1e-10) {
            atomicStore(&lu_singular_flag, 1);
            return;
        }

        // Eliminate below pivot
        for (var row: u32 = col + 1u; row < m; row = row + 1u) {
            let factor = lu_matrix[row * n + col] / pivot;
            lu_matrix[row * n + col] = factor; // Store L factor

            for (var j: u32 = col + 1u; j < n; j = j + 1u) {
                lu_matrix[row * n + j] = lu_matrix[row * n + j] - factor * lu_matrix[col * n + j];
            }
        }
    }

    atomicStore(&lu_num_swaps, num_swaps);
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

struct CholeskyParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> chol_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> chol_not_pd_flag: atomic<i32>;
@group(0) @binding(2) var<uniform> chol_params: CholeskyParams;

@compute @workgroup_size(1)
fn cholesky_decompose_f32() {
    let n = chol_params.n;

    for (var j: u32 = 0u; j < n; j = j + 1u) {
        // Compute L[j,j]
        var sum: f32 = chol_matrix[j * n + j];
        for (var k: u32 = 0u; k < j; k = k + 1u) {
            let ljk = chol_matrix[j * n + k];
            sum = sum - ljk * ljk;
        }

        if (sum <= 0.0) {
            atomicStore(&chol_not_pd_flag, 1);
            return;
        }

        let ljj = sqrt(sum);
        chol_matrix[j * n + j] = ljj;

        // Compute L[i,j] for i > j
        for (var i: u32 = j + 1u; i < n; i = i + 1u) {
            var s: f32 = chol_matrix[i * n + j];
            for (var k: u32 = 0u; k < j; k = k + 1u) {
                s = s - chol_matrix[i * n + k] * chol_matrix[j * n + k];
            }
            chol_matrix[i * n + j] = s / ljj;
        }

        // Zero out upper triangle
        for (var i: u32 = 0u; i < j; i = i + 1u) {
            chol_matrix[i * n + j] = 0.0;
        }
    }
}

// ============================================================================
// QR Decomposition (Householder)
// ============================================================================

struct QrParams {
    m: u32,
    n: u32,
    thin: u32,
}

@group(0) @binding(0) var<storage, read_write> qr_q: array<f32>;
@group(0) @binding(1) var<storage, read_write> qr_r: array<f32>;
@group(0) @binding(2) var<storage, read_write> qr_workspace: array<f32>;
@group(0) @binding(3) var<uniform> qr_params: QrParams;

@compute @workgroup_size(1)
fn qr_decompose_f32() {
    let m = qr_params.m;
    let n = qr_params.n;
    let thin = qr_params.thin != 0u;
    let k = min(m, n);
    let q_cols = select(m, k, thin);

    // Initialize Q to identity
    for (var i: u32 = 0u; i < m; i = i + 1u) {
        for (var j: u32 = 0u; j < q_cols; j = j + 1u) {
            if (i == j) {
                qr_q[i * q_cols + j] = 1.0;
            } else {
                qr_q[i * q_cols + j] = 0.0;
            }
        }
    }

    // Householder reflections
    // Only iterate min(m-1, n) times - no need to reflect the last 1x1 block
    let num_reflections = min(m - 1u, n);
    for (var col: u32 = 0u; col < num_reflections; col = col + 1u) {
        // Compute norm of column below diagonal
        var norm_sq: f32 = 0.0;
        for (var i: u32 = col; i < m; i = i + 1u) {
            let val = qr_r[i * n + col];
            norm_sq = norm_sq + val * val;
        }
        let norm = sqrt(norm_sq);

        if (norm < 1e-10) {
            continue;
        }

        // Compute Householder vector
        let x0 = qr_r[col * n + col];
        let sign_x0 = select(-1.0, 1.0, x0 >= 0.0);
        let alpha = -sign_x0 * norm;

        // v = x - alpha * e1 (stored in workspace)
        for (var i: u32 = 0u; i < col; i = i + 1u) {
            qr_workspace[i] = 0.0;
        }
        qr_workspace[col] = x0 - alpha;
        for (var i: u32 = col + 1u; i < m; i = i + 1u) {
            qr_workspace[i] = qr_r[i * n + col];
        }

        // Normalize v
        var v_norm_sq: f32 = 0.0;
        for (var i: u32 = col; i < m; i = i + 1u) {
            v_norm_sq = v_norm_sq + qr_workspace[i] * qr_workspace[i];
        }

        if (v_norm_sq < 1e-20) {
            continue;
        }

        let v_norm = sqrt(v_norm_sq);
        for (var i: u32 = col; i < m; i = i + 1u) {
            qr_workspace[i] = qr_workspace[i] / v_norm;
        }

        // Apply H = I - 2vv^T to R
        for (var j: u32 = col; j < n; j = j + 1u) {
            var dot: f32 = 0.0;
            for (var i: u32 = col; i < m; i = i + 1u) {
                dot = dot + qr_workspace[i] * qr_r[i * n + j];
            }
            for (var i: u32 = col; i < m; i = i + 1u) {
                qr_r[i * n + j] = qr_r[i * n + j] - 2.0 * qr_workspace[i] * dot;
            }
        }

        // Apply H to Q
        for (var j: u32 = 0u; j < q_cols; j = j + 1u) {
            var dot: f32 = 0.0;
            for (var i: u32 = col; i < m; i = i + 1u) {
                dot = dot + qr_workspace[i] * qr_q[i * q_cols + j];
            }
            for (var i: u32 = col; i < m; i = i + 1u) {
                qr_q[i * q_cols + j] = qr_q[i * q_cols + j] - 2.0 * qr_workspace[i] * dot;
            }
        }
    }

    // Zero out below diagonal in R
    for (var i: u32 = 1u; i < m; i = i + 1u) {
        for (var j: u32 = 0u; j < min(i, n); j = j + 1u) {
            qr_r[i * n + j] = 0.0;
        }
    }
}

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

    // Copy input to output first
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        perm_output[i] = perm_input[i];
    }

    // Apply swaps in order
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

    // Parallel reduction for max
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            maxabs_shared[tid] = max(maxabs_shared[tid], maxabs_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        // For multi-workgroup, this would need atomicMax
        // For now, take max with existing value
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

    // Parallel reduction for sum
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

// ============================================================================
// SVD Decomposition (One-Sided Jacobi Algorithm)
// ============================================================================
//
// Computes thin SVD: A = U * S * V^T
// Uses One-Sided Jacobi algorithm for backend parity with CPU/CUDA.
//
// Algorithm:
// 1. Initialize B = A (working copy), V = I_n
// 2. Repeat sweeps (max 30):
//    - For each pair (p, q) where p < q:
//      - Compute Gram elements: a_pp = B[:,p]·B[:,p], a_qq = B[:,q]·B[:,q], a_pq = B[:,p]·B[:,q]
//      - If |a_pq| > tol: compute Jacobi rotation (c,s), apply to B and V columns
//    - Check convergence: off-diagonal Frobenius norm < tolerance
// 3. Extract: S[j] = ||B[:,j]||, U[:,j] = B[:,j]/S[j]
// 4. Sort singular values descending, reorder U and V columns
//
// Buffers:
//   b: Working matrix [m * n], becomes U columns after normalization
//   v: Right singular vectors [n * n], initialized to identity
//   s: Singular values [n]
//   converged_flag: 0 if converged, 1 if not converged after max iterations

struct SvdParams {
    work_m: u32,
    work_n: u32,
}

@group(0) @binding(0) var<storage, read_write> svd_b: array<f32>;
@group(0) @binding(1) var<storage, read_write> svd_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> svd_s: array<f32>;
@group(0) @binding(3) var<storage, read_write> svd_converged_flag: atomic<i32>;
@group(0) @binding(4) var<uniform> svd_params: SvdParams;

@compute @workgroup_size(1)
fn svd_jacobi_f32() {
    let m = svd_params.work_m;
    let n = svd_params.work_n;

    let tol: f32 = f32(n) * 1.1920929e-7; // n * epsilon for F32
    let max_sweeps: u32 = 30u;

    // Initialize V to identity
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            if (i == j) {
                svd_v[i * n + j] = 1.0;
            } else {
                svd_v[i * n + j] = 0.0;
            }
        }
    }

    // Jacobi iterations
    var converged: bool = false;
    for (var sweep: u32 = 0u; sweep < max_sweeps; sweep = sweep + 1u) {
        // Check convergence: sum of squared off-diagonal elements
        var off_diag_sum: f32 = 0.0;
        for (var p: u32 = 0u; p < n; p = p + 1u) {
            for (var q: u32 = p + 1u; q < n; q = q + 1u) {
                // Compute B[:,p] · B[:,q]
                var a_pq: f32 = 0.0;
                for (var i: u32 = 0u; i < m; i = i + 1u) {
                    a_pq = a_pq + svd_b[i * n + p] * svd_b[i * n + q];
                }
                off_diag_sum = off_diag_sum + a_pq * a_pq;
            }
        }

        if (sqrt(off_diag_sum) < tol) {
            converged = true;
            break;
        }

        // One sweep: process all pairs (p, q) where p < q
        for (var p: u32 = 0u; p < n; p = p + 1u) {
            for (var q: u32 = p + 1u; q < n; q = q + 1u) {
                // Compute Gram matrix elements
                var a_pp: f32 = 0.0;
                var a_qq: f32 = 0.0;
                var a_pq: f32 = 0.0;

                for (var i: u32 = 0u; i < m; i = i + 1u) {
                    let bp = svd_b[i * n + p];
                    let bq = svd_b[i * n + q];
                    a_pp = a_pp + bp * bp;
                    a_qq = a_qq + bq * bq;
                    a_pq = a_pq + bp * bq;
                }

                // Skip if already orthogonal
                if (abs(a_pq) < 1e-15) {
                    continue;
                }

                // Compute Jacobi rotation parameters (c, s)
                // Using stable formula from LAPACK
                var c: f32;
                var s: f32;

                let tau = (a_qq - a_pp) / (2.0 * a_pq);
                var t: f32;
                if (tau >= 0.0) {
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                } else {
                    t = 1.0 / (tau - sqrt(1.0 + tau * tau));
                }
                c = 1.0 / sqrt(1.0 + t * t);
                s = t * c;

                // Apply rotation to columns p and q of B
                for (var i: u32 = 0u; i < m; i = i + 1u) {
                    let bp = svd_b[i * n + p];
                    let bq = svd_b[i * n + q];
                    svd_b[i * n + p] = c * bp - s * bq;
                    svd_b[i * n + q] = s * bp + c * bq;
                }

                // Apply rotation to columns p and q of V
                for (var i: u32 = 0u; i < n; i = i + 1u) {
                    let vp = svd_v[i * n + p];
                    let vq = svd_v[i * n + q];
                    svd_v[i * n + p] = c * vp - s * vq;
                    svd_v[i * n + q] = s * vp + c * vq;
                }
            }
        }
    }

    // Set convergence flag
    if (!converged) {
        atomicStore(&svd_converged_flag, 1);
    } else {
        atomicStore(&svd_converged_flag, 0);
    }

    // Extract singular values and normalize U columns
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        // Compute column norm
        var norm_sq: f32 = 0.0;
        for (var i: u32 = 0u; i < m; i = i + 1u) {
            let val = svd_b[i * n + j];
            norm_sq = norm_sq + val * val;
        }
        let norm = sqrt(norm_sq);
        svd_s[j] = norm;

        // Normalize column to get U
        if (norm > 1e-15) {
            for (var i: u32 = 0u; i < m; i = i + 1u) {
                svd_b[i * n + j] = svd_b[i * n + j] / norm;
            }
        }
    }

    // Sort singular values descending (bubble sort for simplicity - n is typically small)
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            if (svd_s[j] > svd_s[i]) {
                // Swap singular values
                let tmp_s = svd_s[i];
                svd_s[i] = svd_s[j];
                svd_s[j] = tmp_s;

                // Swap U columns
                for (var row: u32 = 0u; row < m; row = row + 1u) {
                    let tmp_b = svd_b[row * n + i];
                    svd_b[row * n + i] = svd_b[row * n + j];
                    svd_b[row * n + j] = tmp_b;
                }

                // Swap V columns
                for (var row: u32 = 0u; row < n; row = row + 1u) {
                    let tmp_v = svd_v[row * n + i];
                    svd_v[row * n + i] = svd_v[row * n + j];
                    svd_v[row * n + j] = tmp_v;
                }
            }
        }
    }
}

// ============================================================================
// Eigendecomposition for Symmetric Matrices (Jacobi Algorithm)
// ============================================================================
//
// Computes eigendecomposition: A = V * diag(eigenvalues) * V^T
// Uses Jacobi eigenvalue algorithm for backend parity with CPU/CUDA.
//
// Algorithm:
// 1. Initialize V = I_n (eigenvector matrix)
// 2. Repeat sweeps (max 30):
//    - For each pair (p, q) where p < q:
//      - If |A[p,q]| > tol: compute Jacobi rotation (c,s)
//      - Apply rotation: A' = J^T @ A @ J (zeros out A[p,q])
//      - Update V: V = V @ J
//    - Check convergence: max off-diagonal < tolerance
// 3. eigenvalues = diag(A)
// 4. Sort eigenvalues by magnitude descending, reorder V columns
//
// Buffers:
//   work: Working matrix [n * n], modified in-place to become diagonal
//   eigenvectors: Eigenvector matrix [n * n], initialized to identity
//   eigenvalues: Eigenvalues [n]
//   converged_flag: 0 if converged, 1 if not converged after max iterations

struct EigParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> eig_work: array<f32>;
@group(0) @binding(1) var<storage, read_write> eig_vectors: array<f32>;
@group(0) @binding(2) var<storage, read_write> eig_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> eig_converged_flag: atomic<i32>;
@group(0) @binding(4) var<uniform> eig_params: EigParams;

@compute @workgroup_size(1)
fn eig_jacobi_symmetric_f32() {
    let n = eig_params.n;

    let tol: f32 = f32(n) * 1.1920929e-7; // n * epsilon for F32
    let max_sweeps: u32 = 30u;

    // Initialize eigenvector matrix V to identity
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            if (i == j) {
                eig_vectors[i * n + j] = 1.0;
            } else {
                eig_vectors[i * n + j] = 0.0;
            }
        }
    }

    // Symmetrize input (use lower triangle)
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < i; j = j + 1u) {
            let val = eig_work[i * n + j];
            eig_work[j * n + i] = val;
        }
    }

    // Jacobi iterations
    var converged: bool = false;
    for (var sweep: u32 = 0u; sweep < max_sweeps; sweep = sweep + 1u) {
        // Find maximum off-diagonal element
        var max_off_diag: f32 = 0.0;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            for (var j: u32 = i + 1u; j < n; j = j + 1u) {
                let val = abs(eig_work[i * n + j]);
                if (val > max_off_diag) {
                    max_off_diag = val;
                }
            }
        }

        // Check convergence
        if (max_off_diag < tol) {
            converged = true;
            break;
        }

        // One sweep: process all pairs (p, q) where p < q
        for (var p: u32 = 0u; p < n; p = p + 1u) {
            for (var q: u32 = p + 1u; q < n; q = q + 1u) {
                let a_pq = eig_work[p * n + q];

                // Skip if already essentially zero
                if (abs(a_pq) < tol) {
                    continue;
                }

                let a_pp = eig_work[p * n + p];
                let a_qq = eig_work[q * n + q];

                // Compute Jacobi rotation parameters (c, s)
                var c: f32;
                var s: f32;

                let tau_num = a_qq - a_pp;
                let tau_den = 2.0 * a_pq;

                if (abs(tau_den) < 1e-30) {
                    c = 1.0;
                    s = 0.0;
                } else {
                    let tau = tau_num / tau_den;
                    var t: f32;
                    if (tau >= 0.0) {
                        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                    } else {
                        t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
                    }
                    c = 1.0 / sqrt(1.0 + t * t);
                    s = t * c;
                }

                // Apply Jacobi rotation: A' = J^T @ A @ J
                // Update rows and columns p and q
                for (var k: u32 = 0u; k < n; k = k + 1u) {
                    if (k != p && k != q) {
                        let a_kp = eig_work[k * n + p];
                        let a_kq = eig_work[k * n + q];

                        let new_kp = c * a_kp - s * a_kq;
                        let new_kq = s * a_kp + c * a_kq;

                        eig_work[k * n + p] = new_kp;
                        eig_work[p * n + k] = new_kp;
                        eig_work[k * n + q] = new_kq;
                        eig_work[q * n + k] = new_kq;
                    }
                }

                // Update diagonal elements
                let c2 = c * c;
                let s2 = s * s;
                let cs2 = 2.0 * c * s;

                let new_pp = c2 * a_pp - cs2 * a_pq + s2 * a_qq;
                let new_qq = s2 * a_pp + cs2 * a_pq + c2 * a_qq;

                eig_work[p * n + p] = new_pp;
                eig_work[q * n + q] = new_qq;
                eig_work[p * n + q] = 0.0;
                eig_work[q * n + p] = 0.0;

                // Update eigenvector matrix: V = V @ J
                for (var i: u32 = 0u; i < n; i = i + 1u) {
                    let v_ip = eig_vectors[i * n + p];
                    let v_iq = eig_vectors[i * n + q];

                    eig_vectors[i * n + p] = c * v_ip - s * v_iq;
                    eig_vectors[i * n + q] = s * v_ip + c * v_iq;
                }
            }
        }
    }

    // Set convergence flag
    if (!converged) {
        atomicStore(&eig_converged_flag, 1);
    } else {
        atomicStore(&eig_converged_flag, 0);
    }

    // Extract eigenvalues (diagonal elements)
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        eig_values[i] = eig_work[i * n + i];
    }

    // Sort eigenvalues by magnitude descending (selection sort)
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            if (abs(eig_values[j]) > abs(eig_values[i])) {
                // Swap eigenvalues
                let tmp_e = eig_values[i];
                eig_values[i] = eig_values[j];
                eig_values[j] = tmp_e;

                // Swap eigenvector columns
                for (var row: u32 = 0u; row < n; row = row + 1u) {
                    let tmp_v = eig_vectors[row * n + i];
                    eig_vectors[row * n + i] = eig_vectors[row * n + j];
                    eig_vectors[row * n + j] = tmp_v;
                }
            }
        }
    }
}

// ============================================================================
// Schur Decomposition - Hessenberg reduction + QR iteration
// For general (non-symmetric) matrices: A = Z @ T @ Z^T
// T is quasi-upper-triangular (real Schur form), Z is orthogonal
// ============================================================================
//
// Algorithm:
// 1. Hessenberg reduction using Householder reflections
// 2. QR iteration with Wilkinson shift until convergence
// 3. Clean up small subdiagonals and strictly lower triangular elements
//
// Note: Uses F32 internally since WGSL doesn't support F64

struct SchurParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> schur_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> schur_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> schur_converged_flag: atomic<i32>;
@group(0) @binding(3) var<uniform> schur_params: SchurParams;

@compute @workgroup_size(1)
fn schur_decompose_f32() {
    let n = schur_params.n;
    let eps: f32 = 1.1920929e-7;  // F32 epsilon
    let max_sweeps: u32 = 30u * n;

    // Initialize Z as identity
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            if (i == j) {
                schur_z[i * n + j] = 1.0;
            } else {
                schur_z[i * n + j] = 0.0;
            }
        }
    }

    // Step 1: Hessenberg reduction using Householder reflections
    if (n > 2u) {
        for (var k: u32 = 0u; k < n - 2u; k = k + 1u) {
            // Compute norm of column k, rows k+1 to n-1
            var norm_sq: f32 = 0.0;
            for (var i: u32 = k + 1u; i < n; i = i + 1u) {
                let val = schur_t[i * n + k];
                norm_sq = norm_sq + val * val;
            }

            if (norm_sq < eps) {
                continue;
            }

            let norm = sqrt(norm_sq);
            let x0 = schur_t[(k + 1u) * n + k];
            var alpha: f32;
            if (x0 >= 0.0) {
                alpha = -norm;
            } else {
                alpha = norm;
            }

            // v = x - alpha * e1, then normalize
            let v0 = x0 - alpha;
            var v_norm_sq: f32 = v0 * v0;
            for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                let val = schur_t[i * n + k];
                v_norm_sq = v_norm_sq + val * val;
            }

            if (v_norm_sq < eps) {
                continue;
            }

            let v_norm = sqrt(v_norm_sq);

            // Left multiply: T = (I - 2vv^T) @ T
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                var dot: f32 = 0.0;
                dot = dot + (v0 / v_norm) * schur_t[(k + 1u) * n + j];
                for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                    dot = dot + (schur_t[i * n + k] / v_norm) * schur_t[i * n + j];
                }
                schur_t[(k + 1u) * n + j] = schur_t[(k + 1u) * n + j] - 2.0 * (v0 / v_norm) * dot;
                for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                    schur_t[i * n + j] = schur_t[i * n + j] - 2.0 * (schur_t[i * n + k] / v_norm) * dot;
                }
            }

            // Right multiply: T = T @ (I - 2vv^T)
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                var dot: f32 = 0.0;
                dot = dot + schur_t[i * n + (k + 1u)] * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    dot = dot + schur_t[i * n + jj] * (schur_t[jj * n + k] / v_norm);
                }
                schur_t[i * n + (k + 1u)] = schur_t[i * n + (k + 1u)] - 2.0 * dot * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    schur_t[i * n + jj] = schur_t[i * n + jj] - 2.0 * dot * (schur_t[jj * n + k] / v_norm);
                }
            }

            // Accumulate Z: Z = Z @ (I - 2vv^T)
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                var dot: f32 = 0.0;
                dot = dot + schur_z[i * n + (k + 1u)] * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    dot = dot + schur_z[i * n + jj] * (schur_t[jj * n + k] / v_norm);
                }
                schur_z[i * n + (k + 1u)] = schur_z[i * n + (k + 1u)] - 2.0 * dot * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    schur_z[i * n + jj] = schur_z[i * n + jj] - 2.0 * dot * (schur_t[jj * n + k] / v_norm);
                }
            }
        }
    }

    // Step 2: QR iteration with Wilkinson shift
    var converged: bool = false;
    for (var iter: u32 = 0u; iter < max_sweeps; iter = iter + 1u) {
        // Check convergence (all subdiagonals essentially zero)
        var is_converged: bool = true;
        if (n > 1u) {
            for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
                let h_ii = abs(schur_t[i * n + i]);
                let h_ip1 = abs(schur_t[(i + 1u) * n + (i + 1u)]);
                let threshold = eps * max(h_ii + h_ip1, 1.0);
                if (abs(schur_t[(i + 1u) * n + i]) > threshold) {
                    is_converged = false;
                    break;
                }
            }
        }

        if (is_converged) {
            converged = true;
            break;
        }

        // Compute Wilkinson shift from bottom 2x2 block
        let a_val = schur_t[(n - 2u) * n + (n - 2u)];
        let b_val = schur_t[(n - 2u) * n + (n - 1u)];
        let c_val = schur_t[(n - 1u) * n + (n - 2u)];
        let d_val = schur_t[(n - 1u) * n + (n - 1u)];

        let trace = a_val + d_val;
        let det = a_val * d_val - b_val * c_val;
        let disc = trace * trace - 4.0 * det;

        var shift: f32;
        if (disc >= 0.0) {
            let sqrt_disc = sqrt(disc);
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;
            if (abs(lambda1 - d_val) < abs(lambda2 - d_val)) {
                shift = lambda1;
            } else {
                shift = lambda2;
            }
        } else {
            shift = trace / 2.0;
        }

        // Apply shift
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            schur_t[i * n + i] = schur_t[i * n + i] - shift;
        }

        // QR step using Givens rotations
        if (n > 1u) {
            for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
                let a_ii = schur_t[i * n + i];
                let b_ii = schur_t[(i + 1u) * n + i];

                if (abs(b_ii) < eps) {
                    continue;
                }

                let r = sqrt(a_ii * a_ii + b_ii * b_ii);
                let cs = a_ii / r;
                let sn = -b_ii / r;

                // Left multiply (Q^T @ T)
                for (var j: u32 = 0u; j < n; j = j + 1u) {
                    let t1 = schur_t[i * n + j];
                    let t2 = schur_t[(i + 1u) * n + j];
                    schur_t[i * n + j] = cs * t1 - sn * t2;
                    schur_t[(i + 1u) * n + j] = sn * t1 + cs * t2;
                }

                // Right multiply (T @ Q)
                for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
                    let t1 = schur_t[kk * n + i];
                    let t2 = schur_t[kk * n + (i + 1u)];
                    schur_t[kk * n + i] = cs * t1 - sn * t2;
                    schur_t[kk * n + (i + 1u)] = sn * t1 + cs * t2;
                }

                // Accumulate Z
                for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
                    let z1 = schur_z[kk * n + i];
                    let z2 = schur_z[kk * n + (i + 1u)];
                    schur_z[kk * n + i] = cs * z1 - sn * z2;
                    schur_z[kk * n + (i + 1u)] = sn * z1 + cs * z2;
                }
            }
        }

        // Remove shift
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            schur_t[i * n + i] = schur_t[i * n + i] + shift;
        }
    }

    // Set convergence flag
    if (converged) {
        atomicStore(&schur_converged_flag, 0);
    } else {
        atomicStore(&schur_converged_flag, 1);
    }

    // Clean up small subdiagonals
    if (n > 1u) {
        for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
            let h_ii = abs(schur_t[i * n + i]);
            let h_ip1 = abs(schur_t[(i + 1u) * n + (i + 1u)]);
            let threshold = eps * max(h_ii + h_ip1, 1.0);
            if (abs(schur_t[(i + 1u) * n + i]) <= threshold) {
                schur_t[(i + 1u) * n + i] = 0.0;
            }
        }
    }

    // Clear strictly lower triangular (except first subdiagonal for 2x2 blocks)
    if (n > 2u) {
        for (var i: u32 = 2u; i < n; i = i + 1u) {
            for (var j: u32 = 0u; j < i - 1u; j = j + 1u) {
                schur_t[i * n + j] = 0.0;
            }
        }
    }
}

// ============================================================================
// General Eigenvalue Decomposition - for non-symmetric matrices
// Uses Schur decomposition + back-substitution for eigenvectors
// Returns real and imaginary parts of eigenvalues and eigenvectors
// ============================================================================

struct EigGeneralParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> eigg_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> eigg_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> eigg_eval_real: array<f32>;
@group(0) @binding(3) var<storage, read_write> eigg_eval_imag: array<f32>;
@group(0) @binding(4) var<storage, read_write> eigg_evec_real: array<f32>;
@group(0) @binding(5) var<storage, read_write> eigg_evec_imag: array<f32>;
@group(0) @binding(6) var<storage, read_write> eigg_converged_flag: atomic<i32>;
@group(0) @binding(7) var<uniform> eigg_params: EigGeneralParams;

@compute @workgroup_size(1)
fn eig_general_f32() {
    let n = eigg_params.n;
    let eps: f32 = 1.1920929e-7;
    let max_sweeps: u32 = 30u * n;

    // Initialize Z as identity
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            if (i == j) {
                eigg_z[i * n + j] = 1.0;
            } else {
                eigg_z[i * n + j] = 0.0;
            }
        }
    }

    // === Schur decomposition (inline) ===

    // Hessenberg reduction
    if (n > 2u) {
        for (var k: u32 = 0u; k < n - 2u; k = k + 1u) {
            var norm_sq: f32 = 0.0;
            for (var i: u32 = k + 1u; i < n; i = i + 1u) {
                let val = eigg_t[i * n + k];
                norm_sq = norm_sq + val * val;
            }

            if (norm_sq < eps) {
                continue;
            }

            let norm = sqrt(norm_sq);
            let x0 = eigg_t[(k + 1u) * n + k];
            var alpha: f32;
            if (x0 >= 0.0) {
                alpha = -norm;
            } else {
                alpha = norm;
            }

            let v0 = x0 - alpha;
            var v_norm_sq: f32 = v0 * v0;
            for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                let val = eigg_t[i * n + k];
                v_norm_sq = v_norm_sq + val * val;
            }

            if (v_norm_sq < eps) {
                continue;
            }

            let v_norm = sqrt(v_norm_sq);

            for (var j: u32 = 0u; j < n; j = j + 1u) {
                var dot: f32 = 0.0;
                dot = dot + (v0 / v_norm) * eigg_t[(k + 1u) * n + j];
                for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                    dot = dot + (eigg_t[i * n + k] / v_norm) * eigg_t[i * n + j];
                }
                eigg_t[(k + 1u) * n + j] = eigg_t[(k + 1u) * n + j] - 2.0 * (v0 / v_norm) * dot;
                for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                    eigg_t[i * n + j] = eigg_t[i * n + j] - 2.0 * (eigg_t[i * n + k] / v_norm) * dot;
                }
            }

            for (var i: u32 = 0u; i < n; i = i + 1u) {
                var dot: f32 = 0.0;
                dot = dot + eigg_t[i * n + (k + 1u)] * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    dot = dot + eigg_t[i * n + jj] * (eigg_t[jj * n + k] / v_norm);
                }
                eigg_t[i * n + (k + 1u)] = eigg_t[i * n + (k + 1u)] - 2.0 * dot * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    eigg_t[i * n + jj] = eigg_t[i * n + jj] - 2.0 * dot * (eigg_t[jj * n + k] / v_norm);
                }
            }

            for (var i: u32 = 0u; i < n; i = i + 1u) {
                var dot: f32 = 0.0;
                dot = dot + eigg_z[i * n + (k + 1u)] * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    dot = dot + eigg_z[i * n + jj] * (eigg_t[jj * n + k] / v_norm);
                }
                eigg_z[i * n + (k + 1u)] = eigg_z[i * n + (k + 1u)] - 2.0 * dot * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    eigg_z[i * n + jj] = eigg_z[i * n + jj] - 2.0 * dot * (eigg_t[jj * n + k] / v_norm);
                }
            }
        }
    }

    // QR iteration
    var converged: bool = false;
    for (var iter: u32 = 0u; iter < max_sweeps; iter = iter + 1u) {
        var is_converged: bool = true;
        if (n > 1u) {
            for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
                let h_ii = abs(eigg_t[i * n + i]);
                let h_ip1 = abs(eigg_t[(i + 1u) * n + (i + 1u)]);
                let threshold = eps * max(h_ii + h_ip1, 1.0);
                if (abs(eigg_t[(i + 1u) * n + i]) > threshold) {
                    is_converged = false;
                    break;
                }
            }
        }

        if (is_converged) {
            converged = true;
            break;
        }

        let a_val = eigg_t[(n - 2u) * n + (n - 2u)];
        let b_val = eigg_t[(n - 2u) * n + (n - 1u)];
        let c_val = eigg_t[(n - 1u) * n + (n - 2u)];
        let d_val = eigg_t[(n - 1u) * n + (n - 1u)];

        let trace = a_val + d_val;
        let det = a_val * d_val - b_val * c_val;
        let disc = trace * trace - 4.0 * det;

        var shift: f32;
        if (disc >= 0.0) {
            let sqrt_disc = sqrt(disc);
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;
            if (abs(lambda1 - d_val) < abs(lambda2 - d_val)) {
                shift = lambda1;
            } else {
                shift = lambda2;
            }
        } else {
            shift = trace / 2.0;
        }

        for (var i: u32 = 0u; i < n; i = i + 1u) {
            eigg_t[i * n + i] = eigg_t[i * n + i] - shift;
        }

        if (n > 1u) {
            for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
                let a_ii = eigg_t[i * n + i];
                let b_ii = eigg_t[(i + 1u) * n + i];

                if (abs(b_ii) < eps) {
                    continue;
                }

                let r = sqrt(a_ii * a_ii + b_ii * b_ii);
                let cs = a_ii / r;
                let sn = -b_ii / r;

                for (var j: u32 = 0u; j < n; j = j + 1u) {
                    let t1 = eigg_t[i * n + j];
                    let t2 = eigg_t[(i + 1u) * n + j];
                    eigg_t[i * n + j] = cs * t1 - sn * t2;
                    eigg_t[(i + 1u) * n + j] = sn * t1 + cs * t2;
                }

                for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
                    let t1 = eigg_t[kk * n + i];
                    let t2 = eigg_t[kk * n + (i + 1u)];
                    eigg_t[kk * n + i] = cs * t1 - sn * t2;
                    eigg_t[kk * n + (i + 1u)] = sn * t1 + cs * t2;
                }

                for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
                    let z1 = eigg_z[kk * n + i];
                    let z2 = eigg_z[kk * n + (i + 1u)];
                    eigg_z[kk * n + i] = cs * z1 - sn * z2;
                    eigg_z[kk * n + (i + 1u)] = sn * z1 + cs * z2;
                }
            }
        }

        for (var i: u32 = 0u; i < n; i = i + 1u) {
            eigg_t[i * n + i] = eigg_t[i * n + i] + shift;
        }
    }

    // Clean up
    if (n > 1u) {
        for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
            let h_ii = abs(eigg_t[i * n + i]);
            let h_ip1 = abs(eigg_t[(i + 1u) * n + (i + 1u)]);
            let threshold = eps * max(h_ii + h_ip1, 1.0);
            if (abs(eigg_t[(i + 1u) * n + i]) <= threshold) {
                eigg_t[(i + 1u) * n + i] = 0.0;
            }
        }
    }

    if (n > 2u) {
        for (var ii: u32 = 2u; ii < n; ii = ii + 1u) {
            for (var jj: u32 = 0u; jj < ii - 1u; jj = jj + 1u) {
                eigg_t[ii * n + jj] = 0.0;
            }
        }
    }

    // === Extract eigenvalues from Schur form ===
    var i: u32 = 0u;
    loop {
        if (i >= n) {
            break;
        }

        if (i == n - 1u) {
            eigg_eval_real[i] = eigg_t[i * n + i];
            eigg_eval_imag[i] = 0.0;
            i = i + 1u;
        } else {
            let subdiag = abs(eigg_t[(i + 1u) * n + i]);
            let diag_scale = abs(eigg_t[i * n + i]) + abs(eigg_t[(i + 1u) * n + (i + 1u)]);
            let threshold = eps * max(diag_scale, 1.0);

            if (subdiag > threshold) {
                // 2x2 block - complex conjugate pair
                let a_blk = eigg_t[i * n + i];
                let b_blk = eigg_t[i * n + (i + 1u)];
                let c_blk = eigg_t[(i + 1u) * n + i];
                let d_blk = eigg_t[(i + 1u) * n + (i + 1u)];

                let tr = a_blk + d_blk;
                let dc = (a_blk - d_blk) * (a_blk - d_blk) / 4.0 + b_blk * c_blk;

                if (dc < 0.0) {
                    let real_part = tr / 2.0;
                    let imag_part = sqrt(-dc);
                    eigg_eval_real[i] = real_part;
                    eigg_eval_imag[i] = imag_part;
                    eigg_eval_real[i + 1u] = real_part;
                    eigg_eval_imag[i + 1u] = -imag_part;
                } else {
                    let sqrt_dc = sqrt(dc);
                    eigg_eval_real[i] = tr / 2.0 + sqrt_dc;
                    eigg_eval_imag[i] = 0.0;
                    eigg_eval_real[i + 1u] = tr / 2.0 - sqrt_dc;
                    eigg_eval_imag[i + 1u] = 0.0;
                }
                i = i + 2u;
            } else {
                eigg_eval_real[i] = eigg_t[i * n + i];
                eigg_eval_imag[i] = 0.0;
                i = i + 1u;
            }
        }
    }

    // === Compute eigenvectors via back-substitution ===
    i = 0u;
    loop {
        if (i >= n) {
            break;
        }

        let imag = eigg_eval_imag[i];

        if (abs(imag) < eps) {
            // Real eigenvalue
            let lambda = eigg_eval_real[i];

            // Initialize y to zero, set y[i] = 1
            for (var k: u32 = 0u; k < n; k = k + 1u) {
                eigg_evec_real[k * n + i] = 0.0;
                eigg_evec_imag[k * n + i] = 0.0;
            }
            var y_real: f32 = 1.0;
            var y_idx: u32 = i;

            // Back-substitution for (T - λI)y = 0
            // Store y temporarily in eigg_evec_real column i (will be overwritten)
            eigg_evec_real[i * n + i] = 1.0;

            if (i > 0u) {
                for (var kk: u32 = 0u; kk < i; kk = kk + 1u) {
                    let k = i - 1u - kk;
                    let diag = eigg_t[k * n + k] - lambda;
                    var rhs: f32 = 0.0;
                    for (var j: u32 = k + 1u; j < n; j = j + 1u) {
                        rhs = rhs - eigg_t[k * n + j] * eigg_evec_real[j * n + i];
                    }
                    if (abs(diag) > eps) {
                        eigg_evec_real[k * n + i] = rhs / diag;
                    } else {
                        eigg_evec_real[k * n + i] = 0.0;
                    }
                }
            }

            // Normalize
            var norm_sq: f32 = 0.0;
            for (var k: u32 = 0u; k < n; k = k + 1u) {
                norm_sq = norm_sq + eigg_evec_real[k * n + i] * eigg_evec_real[k * n + i];
            }
            let norm = sqrt(norm_sq);
            if (norm > eps) {
                for (var k: u32 = 0u; k < n; k = k + 1u) {
                    eigg_evec_real[k * n + i] = eigg_evec_real[k * n + i] / norm;
                }
            }

            // Transform by Z: evec = Z @ y (store in temp, then copy back)
            // We need to use a different approach since we can't have local arrays
            // Transform in-place by computing each row of result
            for (var row: u32 = 0u; row < n; row = row + 1u) {
                var sum: f32 = 0.0;
                for (var k: u32 = 0u; k < n; k = k + 1u) {
                    sum = sum + eigg_z[row * n + k] * eigg_evec_real[k * n + i];
                }
                eigg_evec_imag[row * n + i] = sum; // Temporarily store in imag
            }
            for (var row: u32 = 0u; row < n; row = row + 1u) {
                eigg_evec_real[row * n + i] = eigg_evec_imag[row * n + i];
                eigg_evec_imag[row * n + i] = 0.0;
            }

            i = i + 1u;
        } else {
            // Complex eigenvalue - simplified handling
            // For complex eigenvalues, set eigenvector to a simple approximation
            let lambda_real = eigg_eval_real[i];
            let lambda_imag = eigg_eval_imag[i];

            // Initialize
            for (var k: u32 = 0u; k < n; k = k + 1u) {
                eigg_evec_real[k * n + i] = 0.0;
                eigg_evec_imag[k * n + i] = 0.0;
                eigg_evec_real[k * n + (i + 1u)] = 0.0;
                eigg_evec_imag[k * n + (i + 1u)] = 0.0;
            }

            // Use the 2x2 block to initialize
            let a_blk = eigg_t[i * n + i];
            let b_blk = eigg_t[i * n + (i + 1u)];

            eigg_evec_real[i * n + i] = b_blk;
            eigg_evec_imag[i * n + i] = 0.0;
            eigg_evec_real[(i + 1u) * n + i] = lambda_real - a_blk;
            eigg_evec_imag[(i + 1u) * n + i] = lambda_imag;

            // Back-substitute (simplified)
            if (i > 0u) {
                for (var kk: u32 = 0u; kk < i; kk = kk + 1u) {
                    let k = i - 1u - kk;
                    let diag_real = eigg_t[k * n + k] - lambda_real;
                    let diag_imag = -lambda_imag;

                    var rhs_real: f32 = 0.0;
                    var rhs_imag: f32 = 0.0;
                    for (var j: u32 = k + 1u; j < n; j = j + 1u) {
                        let t_kj = eigg_t[k * n + j];
                        rhs_real = rhs_real - t_kj * eigg_evec_real[j * n + i];
                        rhs_imag = rhs_imag - t_kj * eigg_evec_imag[j * n + i];
                    }

                    let denom = diag_real * diag_real + diag_imag * diag_imag;
                    if (denom > eps * eps) {
                        eigg_evec_real[k * n + i] = (rhs_real * diag_real + rhs_imag * diag_imag) / denom;
                        eigg_evec_imag[k * n + i] = (rhs_imag * diag_real - rhs_real * diag_imag) / denom;
                    }
                }
            }

            // Normalize
            var norm_sq: f32 = 0.0;
            for (var k: u32 = 0u; k < n; k = k + 1u) {
                norm_sq = norm_sq + eigg_evec_real[k * n + i] * eigg_evec_real[k * n + i]
                        + eigg_evec_imag[k * n + i] * eigg_evec_imag[k * n + i];
            }
            let norm = sqrt(norm_sq);
            if (norm > eps) {
                for (var k: u32 = 0u; k < n; k = k + 1u) {
                    eigg_evec_real[k * n + i] = eigg_evec_real[k * n + i] / norm;
                    eigg_evec_imag[k * n + i] = eigg_evec_imag[k * n + i] / norm;
                }
            }

            // Transform by Z and store conjugate pair
            for (var row: u32 = 0u; row < n; row = row + 1u) {
                var sum_real: f32 = 0.0;
                var sum_imag: f32 = 0.0;
                for (var k: u32 = 0u; k < n; k = k + 1u) {
                    let z_val = eigg_z[row * n + k];
                    sum_real = sum_real + z_val * eigg_evec_real[k * n + i];
                    sum_imag = sum_imag + z_val * eigg_evec_imag[k * n + i];
                }
                // Store for both conjugate columns
                eigg_evec_real[row * n + i] = sum_real;
                eigg_evec_imag[row * n + i] = sum_imag;
                eigg_evec_real[row * n + (i + 1u)] = sum_real;
                eigg_evec_imag[row * n + (i + 1u)] = -sum_imag;
            }

            i = i + 2u;
        }
    }

    // Set convergence flag
    if (converged) {
        atomicStore(&eigg_converged_flag, 0);
    } else {
        atomicStore(&eigg_converged_flag, 1);
    }
}
"#;
