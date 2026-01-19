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
"#;
