// Combined linear algebra WGSL shaders
// Generated from split modules in shaders/linalg/

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// From basic_ops.rs
// ============================================================================
// ============================================================================
// Workgroup Configuration
// ============================================================================


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
// From solvers.rs
// ============================================================================
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
// From decompositions.rs
// ============================================================================
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
// From utilities.rs
// ============================================================================

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

// ============================================================================
// From svd.rs
// ============================================================================
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
//      - Compute Gram elements: a_pp, a_qq, a_pq
//      - If |a_pq| > tol: compute Jacobi rotation (c,s), apply to B and V
//    - Check convergence
// 3. Extract: S[j] = ||B[:,j]||, U[:,j] = B[:,j]/S[j]
// 4. Sort singular values descending

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

    let tol: f32 = f32(n) * 1.1920929e-7;
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
        var off_diag_sum: f32 = 0.0;
        for (var p: u32 = 0u; p < n; p = p + 1u) {
            for (var q: u32 = p + 1u; q < n; q = q + 1u) {
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

        for (var p: u32 = 0u; p < n; p = p + 1u) {
            for (var q: u32 = p + 1u; q < n; q = q + 1u) {
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

                if (abs(a_pq) < 1e-15) {
                    continue;
                }

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

                for (var i: u32 = 0u; i < m; i = i + 1u) {
                    let bp = svd_b[i * n + p];
                    let bq = svd_b[i * n + q];
                    svd_b[i * n + p] = c * bp - s * bq;
                    svd_b[i * n + q] = s * bp + c * bq;
                }

                for (var i: u32 = 0u; i < n; i = i + 1u) {
                    let vp = svd_v[i * n + p];
                    let vq = svd_v[i * n + q];
                    svd_v[i * n + p] = c * vp - s * vq;
                    svd_v[i * n + q] = s * vp + c * vq;
                }
            }
        }
    }

    if (!converged) {
        atomicStore(&svd_converged_flag, 1);
    } else {
        atomicStore(&svd_converged_flag, 0);
    }

    for (var j: u32 = 0u; j < n; j = j + 1u) {
        var norm_sq: f32 = 0.0;
        for (var i: u32 = 0u; i < m; i = i + 1u) {
            let val = svd_b[i * n + j];
            norm_sq = norm_sq + val * val;
        }
        let norm = sqrt(norm_sq);
        svd_s[j] = norm;

        if (norm > 1e-15) {
            for (var i: u32 = 0u; i < m; i = i + 1u) {
                svd_b[i * n + j] = svd_b[i * n + j] / norm;
            }
        }
    }

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            if (svd_s[j] > svd_s[i]) {
                let tmp_s = svd_s[i];
                svd_s[i] = svd_s[j];
                svd_s[j] = tmp_s;

                for (var row: u32 = 0u; row < m; row = row + 1u) {
                    let tmp_b = svd_b[row * n + i];
                    svd_b[row * n + i] = svd_b[row * n + j];
                    svd_b[row * n + j] = tmp_b;
                }

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
// From eig_symmetric.rs
// ============================================================================
// ============================================================================
// Eigendecomposition for Symmetric Matrices (Jacobi Algorithm)
// ============================================================================
//
// Computes eigendecomposition: A = V * diag(eigenvalues) * V^T
// Uses Jacobi eigenvalue algorithm for backend parity with CPU/CUDA.

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
    let tol: f32 = f32(n) * 1.1920929e-7;
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
        var max_off_diag: f32 = 0.0;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            for (var j: u32 = i + 1u; j < n; j = j + 1u) {
                let val = abs(eig_work[i * n + j]);
                if (val > max_off_diag) {
                    max_off_diag = val;
                }
            }
        }

        if (max_off_diag < tol) {
            converged = true;
            break;
        }

        for (var p: u32 = 0u; p < n; p = p + 1u) {
            for (var q: u32 = p + 1u; q < n; q = q + 1u) {
                let a_pq = eig_work[p * n + q];
                if (abs(a_pq) < tol) {
                    continue;
                }

                let a_pp = eig_work[p * n + p];
                let a_qq = eig_work[q * n + q];

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

                // Apply Jacobi rotation
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

                let c2 = c * c;
                let s2 = s * s;
                let cs2 = 2.0 * c * s;
                eig_work[p * n + p] = c2 * a_pp - cs2 * a_pq + s2 * a_qq;
                eig_work[q * n + q] = s2 * a_pp + cs2 * a_pq + c2 * a_qq;
                eig_work[p * n + q] = 0.0;
                eig_work[q * n + p] = 0.0;

                for (var i: u32 = 0u; i < n; i = i + 1u) {
                    let v_ip = eig_vectors[i * n + p];
                    let v_iq = eig_vectors[i * n + q];
                    eig_vectors[i * n + p] = c * v_ip - s * v_iq;
                    eig_vectors[i * n + q] = s * v_ip + c * v_iq;
                }
            }
        }
    }

    if (!converged) {
        atomicStore(&eig_converged_flag, 1);
    } else {
        atomicStore(&eig_converged_flag, 0);
    }

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        eig_values[i] = eig_work[i * n + i];
    }

    // Sort by magnitude descending
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            if (abs(eig_values[j]) > abs(eig_values[i])) {
                let tmp_e = eig_values[i];
                eig_values[i] = eig_values[j];
                eig_values[j] = tmp_e;

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
// From schur.rs
// ============================================================================
// ============================================================================
// Schur Decomposition - Hessenberg reduction + QR iteration
// For general (non-symmetric) matrices: A = Z @ T @ Z^T
// T is quasi-upper-triangular (real Schur form), Z is orthogonal
// ============================================================================

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
    let eps: f32 = 1.1920929e-7;
    let max_sweeps: u32 = 30u * n;

    // Initialize Z as identity
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            if (i == j) { schur_z[i * n + j] = 1.0; }
            else { schur_z[i * n + j] = 0.0; }
        }
    }

    // Step 1: Hessenberg reduction
    if (n > 2u) {
        for (var k: u32 = 0u; k < n - 2u; k = k + 1u) {
            var norm_sq: f32 = 0.0;
            for (var i: u32 = k + 1u; i < n; i = i + 1u) {
                let val = schur_t[i * n + k];
                norm_sq = norm_sq + val * val;
            }

            if (norm_sq < eps) { continue; }

            let norm = sqrt(norm_sq);
            let x0 = schur_t[(k + 1u) * n + k];
            var alpha: f32;
            if (x0 >= 0.0) { alpha = -norm; } else { alpha = norm; }

            let v0 = x0 - alpha;
            var v_norm_sq: f32 = v0 * v0;
            for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                let val = schur_t[i * n + k];
                v_norm_sq = v_norm_sq + val * val;
            }

            if (v_norm_sq < eps) { continue; }

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

            // Accumulate Z
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

        if (is_converged) { converged = true; break; }

        // Wilkinson shift
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
            if (abs(lambda1 - d_val) < abs(lambda2 - d_val)) { shift = lambda1; }
            else { shift = lambda2; }
        } else { shift = trace / 2.0; }

        for (var i: u32 = 0u; i < n; i = i + 1u) { schur_t[i * n + i] = schur_t[i * n + i] - shift; }

        // QR step using Givens rotations
        if (n > 1u) {
            for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
                let a_ii = schur_t[i * n + i];
                let b_ii = schur_t[(i + 1u) * n + i];
                if (abs(b_ii) < eps) { continue; }

                let r = sqrt(a_ii * a_ii + b_ii * b_ii);
                let cs = a_ii / r;
                let sn = -b_ii / r;

                for (var j: u32 = 0u; j < n; j = j + 1u) {
                    let t1 = schur_t[i * n + j];
                    let t2 = schur_t[(i + 1u) * n + j];
                    schur_t[i * n + j] = cs * t1 - sn * t2;
                    schur_t[(i + 1u) * n + j] = sn * t1 + cs * t2;
                }

                for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
                    let t1 = schur_t[kk * n + i];
                    let t2 = schur_t[kk * n + (i + 1u)];
                    schur_t[kk * n + i] = cs * t1 - sn * t2;
                    schur_t[kk * n + (i + 1u)] = sn * t1 + cs * t2;
                }

                for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
                    let z1 = schur_z[kk * n + i];
                    let z2 = schur_z[kk * n + (i + 1u)];
                    schur_z[kk * n + i] = cs * z1 - sn * z2;
                    schur_z[kk * n + (i + 1u)] = sn * z1 + cs * z2;
                }
            }
        }

        for (var i: u32 = 0u; i < n; i = i + 1u) { schur_t[i * n + i] = schur_t[i * n + i] + shift; }
    }

    if (converged) { atomicStore(&schur_converged_flag, 0); }
    else { atomicStore(&schur_converged_flag, 1); }

    // Clean up
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

    if (n > 2u) {
        for (var ii: u32 = 2u; ii < n; ii = ii + 1u) {
            for (var jj: u32 = 0u; jj < ii - 1u; jj = jj + 1u) {
                schur_t[ii * n + jj] = 0.0;
            }
        }
    }
}

// ============================================================================
// From eig_general.rs
// ============================================================================
// ============================================================================
// General Eigenvalue Decomposition - for non-symmetric matrices
// Uses Schur decomposition + back-substitution for eigenvectors
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
            if (i == j) { eigg_z[i * n + j] = 1.0; }
            else { eigg_z[i * n + j] = 0.0; }
        }
    }

    // Hessenberg reduction (inline)
    if (n > 2u) {
        for (var k: u32 = 0u; k < n - 2u; k = k + 1u) {
            var norm_sq: f32 = 0.0;
            for (var i: u32 = k + 1u; i < n; i = i + 1u) {
                let val = eigg_t[i * n + k];
                norm_sq = norm_sq + val * val;
            }
            if (norm_sq < eps) { continue; }

            let norm = sqrt(norm_sq);
            let x0 = eigg_t[(k + 1u) * n + k];
            var alpha: f32;
            if (x0 >= 0.0) { alpha = -norm; } else { alpha = norm; }

            let v0 = x0 - alpha;
            var v_norm_sq: f32 = v0 * v0;
            for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                v_norm_sq = v_norm_sq + eigg_t[i * n + k] * eigg_t[i * n + k];
            }
            if (v_norm_sq < eps) { continue; }
            let v_norm = sqrt(v_norm_sq);

            for (var j: u32 = 0u; j < n; j = j + 1u) {
                var dot: f32 = (v0 / v_norm) * eigg_t[(k + 1u) * n + j];
                for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                    dot = dot + (eigg_t[i * n + k] / v_norm) * eigg_t[i * n + j];
                }
                eigg_t[(k + 1u) * n + j] = eigg_t[(k + 1u) * n + j] - 2.0 * (v0 / v_norm) * dot;
                for (var i: u32 = k + 2u; i < n; i = i + 1u) {
                    eigg_t[i * n + j] = eigg_t[i * n + j] - 2.0 * (eigg_t[i * n + k] / v_norm) * dot;
                }
            }

            for (var i: u32 = 0u; i < n; i = i + 1u) {
                var dot: f32 = eigg_t[i * n + (k + 1u)] * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    dot = dot + eigg_t[i * n + jj] * (eigg_t[jj * n + k] / v_norm);
                }
                eigg_t[i * n + (k + 1u)] = eigg_t[i * n + (k + 1u)] - 2.0 * dot * (v0 / v_norm);
                for (var jj: u32 = k + 2u; jj < n; jj = jj + 1u) {
                    eigg_t[i * n + jj] = eigg_t[i * n + jj] - 2.0 * dot * (eigg_t[jj * n + k] / v_norm);
                }
            }

            for (var i: u32 = 0u; i < n; i = i + 1u) {
                var dot: f32 = eigg_z[i * n + (k + 1u)] * (v0 / v_norm);
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
                let threshold = eps * max(abs(eigg_t[i * n + i]) + abs(eigg_t[(i + 1u) * n + (i + 1u)]), 1.0);
                if (abs(eigg_t[(i + 1u) * n + i]) > threshold) {
                    is_converged = false;
                    break;
                }
            }
        }
        if (is_converged) { converged = true; break; }

        // Wilkinson shift
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
            let l1 = (trace + sqrt_disc) / 2.0;
            let l2 = (trace - sqrt_disc) / 2.0;
            if (abs(l1 - d_val) < abs(l2 - d_val)) { shift = l1; } else { shift = l2; }
        } else { shift = trace / 2.0; }

        for (var i: u32 = 0u; i < n; i = i + 1u) { eigg_t[i * n + i] = eigg_t[i * n + i] - shift; }

        if (n > 1u) {
            for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
                let a_ii = eigg_t[i * n + i];
                let b_ii = eigg_t[(i + 1u) * n + i];
                if (abs(b_ii) < eps) { continue; }
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

        for (var i: u32 = 0u; i < n; i = i + 1u) { eigg_t[i * n + i] = eigg_t[i * n + i] + shift; }
    }

    // Clean up
    if (n > 1u) {
        for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
            let threshold = eps * max(abs(eigg_t[i * n + i]) + abs(eigg_t[(i + 1u) * n + (i + 1u)]), 1.0);
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

    // Extract eigenvalues
    var i: u32 = 0u;
    loop {
        if (i >= n) { break; }
        if (i == n - 1u) {
            eigg_eval_real[i] = eigg_t[i * n + i];
            eigg_eval_imag[i] = 0.0;
            i = i + 1u;
        } else {
            let subdiag = abs(eigg_t[(i + 1u) * n + i]);
            let threshold = eps * max(abs(eigg_t[i * n + i]) + abs(eigg_t[(i + 1u) * n + (i + 1u)]), 1.0);
            if (subdiag > threshold) {
                let a_blk = eigg_t[i * n + i];
                let b_blk = eigg_t[i * n + (i + 1u)];
                let c_blk = eigg_t[(i + 1u) * n + i];
                let d_blk = eigg_t[(i + 1u) * n + (i + 1u)];
                let tr = a_blk + d_blk;
                let dc = (a_blk - d_blk) * (a_blk - d_blk) / 4.0 + b_blk * c_blk;
                if (dc < 0.0) {
                    eigg_eval_real[i] = tr / 2.0;
                    eigg_eval_imag[i] = sqrt(-dc);
                    eigg_eval_real[i + 1u] = tr / 2.0;
                    eigg_eval_imag[i + 1u] = -sqrt(-dc);
                } else {
                    eigg_eval_real[i] = tr / 2.0 + sqrt(dc);
                    eigg_eval_imag[i] = 0.0;
                    eigg_eval_real[i + 1u] = tr / 2.0 - sqrt(dc);
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

    // Simplified eigenvector computation
    for (var idx: u32 = 0u; idx < n; idx = idx + 1u) {
        for (var k: u32 = 0u; k < n; k = k + 1u) {
            eigg_evec_real[k * n + idx] = eigg_z[k * n + idx];
            eigg_evec_imag[k * n + idx] = 0.0;
        }
    }

    if (converged) { atomicStore(&eigg_converged_flag, 0); }
    else { atomicStore(&eigg_converged_flag, 1); }
}

// ============================================================================
// From matrix_functions.rs
// ============================================================================
// ============================================================================
// Matrix Exponential - exp(T) for quasi-triangular T (Parlett recurrence)
// ============================================================================

struct MatfunParams {
    n: u32,
    max_iter: u32,
}

@group(0) @binding(0) var<storage, read_write> expm_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> expm_result: array<f32>;
@group(0) @binding(2) var<uniform> expm_params: MatfunParams;

fn exp_2x2_block(a: f32, b: f32, c: f32) -> vec4<f32> {
    let bc = b * c;
    let exp_a = exp(a);
    if (bc < 0.0) {
        let omega = sqrt(-bc);
        let cos_omega = cos(omega);
        let sin_omega = sin(omega);
        return vec4<f32>(exp_a * cos_omega, exp_a * sin_omega * b / omega,
                         exp_a * sin_omega * c / omega, exp_a * cos_omega);
    } else if (bc > 0.0) {
        let delta = sqrt(bc);
        let sinh_delta = sinh(delta);
        let cosh_delta = cosh(delta);
        return vec4<f32>(exp_a * cosh_delta, exp_a * sinh_delta * b / delta,
                         exp_a * sinh_delta * c / delta, exp_a * cosh_delta);
    } else {
        return vec4<f32>(exp_a, exp_a * b, exp_a * c, exp_a);
    }
}

@compute @workgroup_size(1)
fn exp_quasi_triangular_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = expm_params.n;
    let eps: f32 = 1.1920929e-7;

    for (var i: u32 = 0u; i < n * n; i = i + 1u) {
        expm_result[i] = 0.0;
    }

    // Diagonal blocks
    var i: u32 = 0u;
    while (i < n) {
        if (i + 1u < n && abs(expm_t[(i + 1u) * n + i]) > eps) {
            let a = expm_t[i * n + i];
            let b = expm_t[i * n + (i + 1u)];
            let c = expm_t[(i + 1u) * n + i];
            let block = exp_2x2_block(a, b, c);
            expm_result[i * n + i] = block.x;
            expm_result[i * n + (i + 1u)] = block.y;
            expm_result[(i + 1u) * n + i] = block.z;
            expm_result[(i + 1u) * n + (i + 1u)] = block.w;
            i = i + 2u;
        } else {
            expm_result[i * n + i] = exp(expm_t[i * n + i]);
            i = i + 1u;
        }
    }

    // Superdiagonals via Parlett recurrence
    for (var d: u32 = 1u; d < n; d = d + 1u) {
        for (var ii: u32 = 0u; ii < n - d; ii = ii + 1u) {
            let j = ii + d;
            if (ii + 1u < n && abs(expm_t[(ii + 1u) * n + ii]) > eps && d == 1u) { continue; }
            if (j > 0u && abs(expm_t[j * n + (j - 1u)]) > eps && d == 1u) { continue; }

            let t_ii = expm_t[ii * n + ii];
            let t_jj = expm_t[j * n + j];
            let t_ij = expm_t[ii * n + j];
            var sum: f32 = 0.0;
            for (var k: u32 = ii + 1u; k < j; k = k + 1u) {
                sum = sum + expm_result[ii * n + k] * expm_t[k * n + j];
                sum = sum - expm_t[ii * n + k] * expm_result[k * n + j];
            }
            let f_ii = expm_result[ii * n + ii];
            let f_jj = expm_result[j * n + j];
            let diff = t_ii - t_jj;
            if (abs(diff) > eps) {
                expm_result[ii * n + j] = ((f_ii - f_jj) * t_ij + sum) / diff;
            } else {
                expm_result[ii * n + j] = f_ii * t_ij + sum;
            }
        }
    }
}

// ============================================================================
// Matrix Square Root - sqrt(T) using Denman-Beavers iteration
// ============================================================================

@group(0) @binding(0) var<storage, read_write> sqrtm_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> sqrtm_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> sqrtm_z: array<f32>;
@group(0) @binding(3) var<storage, read_write> sqrtm_work1: array<f32>;
@group(0) @binding(4) var<storage, read_write> sqrtm_work2: array<f32>;
@group(0) @binding(5) var<uniform> sqrtm_params: MatfunParams;

fn lu_factor_inplace(lu: ptr<function, array<f32, 256>>, perm: ptr<function, array<u32, 16>>, n: u32) -> bool {
    let eps: f32 = 1e-10;
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        var max_val: f32 = abs((*lu)[k * n + k]);
        var max_row: u32 = k;
        for (var i: u32 = k + 1u; i < n; i = i + 1u) {
            let val = abs((*lu)[i * n + k]);
            if (val > max_val) { max_val = val; max_row = i; }
        }
        if (max_val < eps) { return false; }
        if (max_row != k) {
            let tmp_perm = (*perm)[k];
            (*perm)[k] = (*perm)[max_row];
            (*perm)[max_row] = tmp_perm;
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                let tmp = (*lu)[k * n + j];
                (*lu)[k * n + j] = (*lu)[max_row * n + j];
                (*lu)[max_row * n + j] = tmp;
            }
        }
        let pivot = (*lu)[k * n + k];
        for (var i: u32 = k + 1u; i < n; i = i + 1u) {
            let factor = (*lu)[i * n + k] / pivot;
            (*lu)[i * n + k] = factor;
            for (var j: u32 = k + 1u; j < n; j = j + 1u) {
                (*lu)[i * n + j] = (*lu)[i * n + j] - factor * (*lu)[k * n + j];
            }
        }
    }
    return true;
}

@compute @workgroup_size(1)
fn sqrt_quasi_triangular_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = sqrtm_params.n;
    let max_iter = sqrtm_params.max_iter;
    let eps: f32 = 1.1920929e-7;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            sqrtm_y[i * n + j] = sqrtm_input[i * n + j];
            if (i == j) { sqrtm_z[i * n + j] = 1.0; }
            else { sqrtm_z[i * n + j] = 0.0; }
        }
    }

    for (var iter: u32 = 0u; iter < max_iter; iter = iter + 1u) {
        var lu_y: array<f32, 256>;
        var perm_y: array<u32, 16>;
        var lu_z: array<f32, 256>;
        var perm_z: array<u32, 16>;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            perm_y[i] = i;
            perm_z[i] = i;
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                lu_y[i * n + j] = sqrtm_y[i * n + j];
                lu_z[i * n + j] = sqrtm_z[i * n + j];
            }
        }
        let y_ok = lu_factor_inplace(&lu_y, &perm_y, n);
        let z_ok = lu_factor_inplace(&lu_z, &perm_z, n);
        if (!y_ok || !z_ok) { return; }

        for (var col: u32 = 0u; col < n; col = col + 1u) {
            var b: array<f32, 16>;
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                if (perm_y[i] == col) { b[i] = 1.0; } else { b[i] = 0.0; }
            }
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                for (var j: u32 = 0u; j < i; j = j + 1u) { b[i] = b[i] - lu_y[i * n + j] * b[j]; }
            }
            for (var i_rev: u32 = 0u; i_rev < n; i_rev = i_rev + 1u) {
                let i = n - 1u - i_rev;
                for (var j: u32 = i + 1u; j < n; j = j + 1u) { b[i] = b[i] - lu_y[i * n + j] * b[j]; }
                b[i] = b[i] / lu_y[i * n + i];
            }
            for (var i: u32 = 0u; i < n; i = i + 1u) { sqrtm_work1[i * n + col] = b[i]; }
        }

        for (var col: u32 = 0u; col < n; col = col + 1u) {
            var b: array<f32, 16>;
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                if (perm_z[i] == col) { b[i] = 1.0; } else { b[i] = 0.0; }
            }
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                for (var j: u32 = 0u; j < i; j = j + 1u) { b[i] = b[i] - lu_z[i * n + j] * b[j]; }
            }
            for (var i_rev: u32 = 0u; i_rev < n; i_rev = i_rev + 1u) {
                let i = n - 1u - i_rev;
                for (var j: u32 = i + 1u; j < n; j = j + 1u) { b[i] = b[i] - lu_z[i * n + j] * b[j]; }
                b[i] = b[i] / lu_z[i * n + i];
            }
            for (var i: u32 = 0u; i < n; i = i + 1u) { sqrtm_work2[i * n + col] = b[i]; }
        }

        var diff: f32 = 0.0;
        for (var i: u32 = 0u; i < n * n; i = i + 1u) {
            let y_new = (sqrtm_y[i] + sqrtm_work2[i]) / 2.0;
            let z_new = (sqrtm_z[i] + sqrtm_work1[i]) / 2.0;
            diff = diff + abs(y_new - sqrtm_y[i]);
            sqrtm_y[i] = y_new;
            sqrtm_z[i] = z_new;
        }
        if (diff < eps * f32(n * n)) { break; }
    }
}

// ============================================================================
// Matrix Logarithm - log(T) using inverse scaling and squaring
// ============================================================================

@group(0) @binding(0) var<storage, read_write> logm_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> logm_work: array<f32>;
@group(0) @binding(2) var<storage, read_write> logm_result: array<f32>;
@group(0) @binding(3) var<storage, read_write> logm_temp: array<f32>;
@group(0) @binding(4) var<storage, read_write> logm_xpower: array<f32>;
@group(0) @binding(5) var<uniform> logm_params: MatfunParams;

@compute @workgroup_size(1)
fn log_quasi_triangular_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = logm_params.n;
    let max_sqrt = logm_params.max_iter;
    let eps: f32 = 1.1920929e-7;

    for (var i: u32 = 0u; i < n * n; i = i + 1u) {
        logm_work[i] = logm_input[i];
    }

    // Square root scaling
    var k: u32 = 0u;
    while (k < max_sqrt) {
        var norm_diff: f32 = 0.0;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                var expected: f32 = 0.0;
                if (i == j) { expected = 1.0; }
                let diff = logm_work[i * n + j] - expected;
                norm_diff = norm_diff + diff * diff;
            }
        }
        norm_diff = sqrt(norm_diff);
        if (norm_diff < 0.5) { break; }

        // Denman-Beavers sqrt
        for (var i: u32 = 0u; i < n * n; i = i + 1u) {
            logm_temp[i] = logm_work[i];
            logm_xpower[i] = 0.0;
        }
        for (var i: u32 = 0u; i < n; i = i + 1u) { logm_xpower[i * n + i] = 1.0; }

        for (var sq_iter: u32 = 0u; sq_iter < 15u; sq_iter = sq_iter + 1u) {
            var lu_y: array<f32, 256>;
            var perm_y: array<u32, 16>;
            var lu_z: array<f32, 256>;
            var perm_z: array<u32, 16>;
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                perm_y[i] = i;
                perm_z[i] = i;
                for (var j: u32 = 0u; j < n; j = j + 1u) {
                    lu_y[i * n + j] = logm_temp[i * n + j];
                    lu_z[i * n + j] = logm_xpower[i * n + j];
                }
            }
            let y_ok = lu_factor_inplace(&lu_y, &perm_y, n);
            let z_ok = lu_factor_inplace(&lu_z, &perm_z, n);
            if (!y_ok || !z_ok) { break; }

            for (var col: u32 = 0u; col < n; col = col + 1u) {
                var by: array<f32, 16>;
                var bz: array<f32, 16>;
                for (var i: u32 = 0u; i < n; i = i + 1u) {
                    if (perm_y[i] == col) { by[i] = 1.0; } else { by[i] = 0.0; }
                    if (perm_z[i] == col) { bz[i] = 1.0; } else { bz[i] = 0.0; }
                }
                for (var i: u32 = 0u; i < n; i = i + 1u) {
                    for (var j: u32 = 0u; j < i; j = j + 1u) {
                        by[i] = by[i] - lu_y[i * n + j] * by[j];
                        bz[i] = bz[i] - lu_z[i * n + j] * bz[j];
                    }
                }
                for (var i_rev: u32 = 0u; i_rev < n; i_rev = i_rev + 1u) {
                    let i = n - 1u - i_rev;
                    for (var j: u32 = i + 1u; j < n; j = j + 1u) {
                        by[i] = by[i] - lu_y[i * n + j] * by[j];
                        bz[i] = bz[i] - lu_z[i * n + j] * bz[j];
                    }
                    by[i] = by[i] / lu_y[i * n + i];
                    bz[i] = bz[i] / lu_z[i * n + i];
                }
                for (var i: u32 = 0u; i < n; i = i + 1u) {
                    logm_result[i * n + col] = by[i];
                    logm_work[i * n + col] = bz[i];
                }
            }

            var diff: f32 = 0.0;
            for (var i: u32 = 0u; i < n * n; i = i + 1u) {
                let y_new = (logm_temp[i] + logm_work[i]) / 2.0;
                let z_new = (logm_xpower[i] + logm_result[i]) / 2.0;
                diff = diff + abs(y_new - logm_temp[i]);
                logm_temp[i] = y_new;
                logm_xpower[i] = z_new;
            }
            if (diff < eps * f32(n * n)) { break; }
        }

        for (var i: u32 = 0u; i < n * n; i = i + 1u) { logm_work[i] = logm_temp[i]; }
        k = k + 1u;
    }

    // X = work - I
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            var expected: f32 = 0.0;
            if (i == j) { expected = 1.0; }
            logm_temp[i * n + j] = logm_work[i * n + j] - expected;
        }
    }

    // Taylor series: log(I+X) = X - X^2/2 + X^3/3 - ...
    for (var i: u32 = 0u; i < n * n; i = i + 1u) {
        logm_result[i] = logm_temp[i];
        logm_xpower[i] = logm_temp[i];
    }

    for (var term: u32 = 2u; term <= 20u; term = term + 1u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                var sum: f32 = 0.0;
                for (var m: u32 = 0u; m < n; m = m + 1u) {
                    sum = sum + logm_xpower[i * n + m] * logm_temp[m * n + j];
                }
                logm_work[i * n + j] = sum;
            }
        }
        for (var i: u32 = 0u; i < n * n; i = i + 1u) { logm_xpower[i] = logm_work[i]; }

        var sign: f32 = -1.0;
        if (term % 2u == 1u) { sign = 1.0; }
        let coeff = sign / f32(term);

        var max_term: f32 = 0.0;
        for (var i: u32 = 0u; i < n * n; i = i + 1u) {
            let t = logm_xpower[i] * coeff;
            logm_result[i] = logm_result[i] + t;
            max_term = max(max_term, abs(t));
        }
        if (max_term < eps) { break; }
    }

    // Scale back
    let scale = f32(1u << k);
    for (var i: u32 = 0u; i < n * n; i = i + 1u) {
        logm_result[i] = logm_result[i] * scale;
    }
}

