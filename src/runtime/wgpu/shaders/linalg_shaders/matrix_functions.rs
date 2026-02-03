//! Matrix functions: exponential, square root, logarithm
//!
//! F32 only - WGSL doesn't support F64.

/// Matrix functions shader: expm, sqrtm, logm
pub const MATRIX_FUNCTIONS_SHADER: &str = r#"
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
"#;
