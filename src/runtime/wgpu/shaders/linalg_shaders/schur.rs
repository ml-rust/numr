//! Schur Decomposition (Hessenberg + QR iteration)
//!
//! F32 only - WGSL doesn't support F64.

/// Schur decomposition shader: schur_decompose
#[allow(dead_code)]
pub const SCHUR_SHADER: &str = r#"
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
"#;
