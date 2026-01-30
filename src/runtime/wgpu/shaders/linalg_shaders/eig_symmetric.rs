//! Symmetric Eigendecomposition (Jacobi Algorithm)
//!
//! F32 only - WGSL doesn't support F64.

/// Symmetric eigendecomposition shader: eig_jacobi_symmetric
#[allow(dead_code)]
pub const EIG_SYMMETRIC_SHADER: &str = r#"
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
"#;
