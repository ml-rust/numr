//! Banded linear system solvers: Thomas algorithm and banded LU
//!
//! F32 only - WGSL doesn't support F64.

/// Banded solvers shader: thomas_solve, banded_lu_solve
pub const BANDED_SHADER: &str = r#"
// ============================================================================
// Thomas Algorithm - Solve tridiagonal system (kl=1, ku=1)
// ============================================================================

struct ThomasParams {
    n: u32,
    ku: u32,
}

@group(0) @binding(0) var<storage, read_write> thomas_ab: array<f32>;
@group(0) @binding(1) var<storage, read_write> thomas_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> thomas_x: array<f32>;
@group(0) @binding(3) var<uniform> thomas_params: ThomasParams;

@compute @workgroup_size(1)
fn thomas_solve_f32() {
    let n = thomas_params.n;
    let ku = thomas_params.ku;

    if (n == 0u) { return; }

    // Diagonal accessors for band storage: ab[row * n + col]
    // main_diag(j) = ab[ku * n + j]
    // upper_diag(j) = ab[(ku - 1) * n + j + 1]  (j < n-1)
    // lower_diag(j) = ab[(ku + 1) * n + j - 1]  (j > 0)

    let m0 = thomas_ab[ku * n + 0u];
    if (m0 == 0.0) { return; }  // singular
    let m0_inv = 1.0 / m0;

    // Forward sweep - store c[] in thomas_x temporarily, d[] in thomas_b (overwrite)
    // c[0] = upper(0) / main(0)
    if (n > 1u) {
        var upper0: f32 = 0.0;
        if (ku > 0u) {
            upper0 = thomas_ab[(ku - 1u) * n + 1u];
        }
        thomas_x[0] = upper0 * m0_inv;  // c[0]
    }
    thomas_b[0] = thomas_b[0] * m0_inv;  // d[0]

    for (var i: u32 = 1u; i < n; i = i + 1u) {
        let a_i = thomas_ab[(ku + 1u) * n + i - 1u];  // lower_diag(i)
        let b_i = thomas_ab[ku * n + i];  // main_diag(i)
        let denom = b_i - a_i * thomas_x[i - 1u];  // b_i - a_i * c[i-1]
        if (denom == 0.0) { return; }  // singular
        let denom_inv = 1.0 / denom;

        if (i < n - 1u) {
            var upper_i: f32 = 0.0;
            if (ku > 0u) {
                upper_i = thomas_ab[(ku - 1u) * n + i + 1u];
            }
            thomas_x[i] = upper_i * denom_inv;  // c[i]
        }
        thomas_b[i] = (thomas_b[i] - a_i * thomas_b[i - 1u]) * denom_inv;  // d[i]
    }

    // Back substitution
    thomas_x[n - 1u] = thomas_b[n - 1u];
    for (var ii: u32 = 1u; ii < n; ii = ii + 1u) {
        let i = n - 1u - ii;
        thomas_x[i] = thomas_b[i] - thomas_x[i] * thomas_x[i + 1u];
    }
}

// ============================================================================
// Banded LU Solve - General banded system with partial pivoting
// ============================================================================

struct BandedLuParams {
    n: u32,
    kl: u32,
    ku: u32,
    band_rows: u32,
}

@group(0) @binding(0) var<storage, read_write> banded_ab: array<f32>;
@group(0) @binding(1) var<storage, read_write> banded_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> banded_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> banded_work: array<f32>;
@group(0) @binding(4) var<uniform> banded_params: BandedLuParams;

@compute @workgroup_size(1)
fn banded_lu_solve_f32() {
    let n = banded_params.n;
    let kl = banded_params.kl;
    let ku = banded_params.ku;
    let band_rows = banded_params.band_rows;
    let work_rows = 2u * kl + ku + 1u;

    if (n == 0u) { return; }

    // Initialize work: work[(kl+r)*n+j] = ab[r*n+j]
    for (var r: u32 = 0u; r < band_rows; r = r + 1u) {
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            banded_work[(kl + r) * n + j] = banded_ab[r * n + j];
        }
    }

    // Copy b to x (will be used as rhs)
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        banded_x[i] = banded_b[i];
    }

    // LU factorization with partial pivoting
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        var max_row_limit = k + kl + 1u;
        if (max_row_limit > n) { max_row_limit = n; }

        var pivot_row = k;
        var pivot_val: f32 = 0.0;

        for (var i: u32 = k; i < max_row_limit; i = i + 1u) {
            let row_idx = kl + ku + i - k;
            let val = abs(banded_work[row_idx * n + k]);
            if (val > pivot_val) {
                pivot_val = val;
                pivot_row = i;
            }
        }

        if (pivot_val == 0.0) { return; }  // singular

        // Swap rows if needed
        if (pivot_row != k) {
            var j_start: u32 = 0u;
            if (k > ku) { j_start = k - ku; }
            var j_end = k + kl + ku + 1u;
            if (j_end > n) { j_end = n; }

            for (var j: u32 = j_start; j < j_end; j = j + 1u) {
                let idx_k_signed = i32(kl + ku + k) - i32(j);
                let idx_p_signed = i32(kl + ku + pivot_row) - i32(j);
                if (idx_k_signed >= 0 && u32(idx_k_signed) < work_rows &&
                    idx_p_signed >= 0 && u32(idx_p_signed) < work_rows) {
                    let a_idx = u32(idx_k_signed) * n + j;
                    let b_idx = u32(idx_p_signed) * n + j;
                    let tmp = banded_work[a_idx];
                    banded_work[a_idx] = banded_work[b_idx];
                    banded_work[b_idx] = tmp;
                }
            }
            // Swap rhs
            let tmp_rhs = banded_x[k];
            banded_x[k] = banded_x[pivot_row];
            banded_x[pivot_row] = tmp_rhs;
        }

        let diag_val = banded_work[(kl + ku) * n + k];

        for (var i: u32 = k + 1u; i < max_row_limit; i = i + 1u) {
            let sub_row = kl + ku + i - k;
            let factor = banded_work[sub_row * n + k] / diag_val;
            banded_work[sub_row * n + k] = factor;

            var col_end = k + ku + 1u;
            if (col_end > n) { col_end = n; }

            for (var j: u32 = k + 1u; j < col_end; j = j + 1u) {
                let row_i_j = kl + ku + i - j;
                let row_k_j = kl + ku + k - j;
                if (row_i_j < work_rows && row_k_j < work_rows) {
                    banded_work[row_i_j * n + j] = banded_work[row_i_j * n + j] - factor * banded_work[row_k_j * n + j];
                }
            }

            banded_x[i] = banded_x[i] - factor * banded_x[k];
        }
    }

    // Back substitution
    for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
        let k = n - 1u - kk;
        let diag_val = banded_work[(kl + ku) * n + k];
        var col_end = k + ku + 1u;
        if (col_end > n) { col_end = n; }

        for (var j: u32 = k + 1u; j < col_end; j = j + 1u) {
            let row_idx = kl + ku + k - j;
            if (row_idx < work_rows) {
                banded_x[k] = banded_x[k] - banded_work[row_idx * n + j] * banded_x[j];
            }
        }
        banded_x[k] = banded_x[k] / diag_val;
    }
}
"#;
