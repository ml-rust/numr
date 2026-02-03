//! Singular Value Decomposition (One-Sided Jacobi Algorithm)
//!
//! F32 only - WGSL doesn't support F64.

/// SVD shader: svd_jacobi
pub const SVD_SHADER: &str = r#"
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
"#;
