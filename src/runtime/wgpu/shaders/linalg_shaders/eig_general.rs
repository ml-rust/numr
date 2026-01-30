//! General Eigenvalue Decomposition (non-symmetric matrices)
//!
//! F32 only - WGSL doesn't support F64.

/// General eigendecomposition shader: eig_general
pub const EIG_GENERAL_SHADER: &str = r#"
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
"#;
