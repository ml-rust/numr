//! Matrix decompositions: LU, Cholesky, QR
//!
//! F32 only - WGSL doesn't support F64.

/// Decompositions shader: lu_decompose, cholesky_decompose, qr_decompose
#[allow(dead_code)]
pub const DECOMPOSITIONS_SHADER: &str = r#"
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
"#;
