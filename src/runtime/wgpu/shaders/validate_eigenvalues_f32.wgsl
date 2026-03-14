// Schur eigenvalue validation for f32

const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    n: u32,
    eps: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> matrix_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;  // [has_error, error_value]
@group(0) @binding(2) var<uniform> params: Params;

// Check if a real eigenvalue is non-positive
fn check_real_eigenvalue(val: f32, eps: f32) -> bool {
    return val <= eps;
}

// Check if a 2x2 block represents non-positive real eigenvalues
// For 2x2 block [[a, b], [c, d]], eigenvalues are (a+d)/2 ± sqrt((a-d)²/4 + bc)
// If discriminant < 0, eigenvalues are complex (ok)
// If discriminant >= 0, check if real part is non-positive
fn check_2x2_block(a: f32, b: f32, c: f32, d: f32, eps: f32) -> bool {
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc < 0.0 {
        // Complex eigenvalues - check real part
        let real_part = trace / 2.0;
        return real_part <= eps;
    } else {
        // Real eigenvalues
        let sqrt_disc = sqrt(disc);
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        return lambda1 <= eps || lambda2 <= eps;
    }
}

@compute @workgroup_size(1)
fn validate_eigenvalues_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let eps = f32(params.eps);

    // Initialize result to "no error"
    result[0] = 0.0;
    result[1] = 0.0;

    var i: u32 = 0u;
    while i < n {
        let diag_idx = i * n + i;

        // Check if this is a 2x2 block (non-zero sub-diagonal)
        if i + 1u < n {
            let sub_diag = abs(matrix_t[(i + 1u) * n + i]);
            if sub_diag > eps {
                // 2x2 block
                let a = matrix_t[i * n + i];
                let b = matrix_t[i * n + (i + 1u)];
                let c = matrix_t[(i + 1u) * n + i];
                let d = matrix_t[(i + 1u) * n + (i + 1u)];

                if check_2x2_block(a, b, c, d, eps) {
                    result[0] = 1.0;
                    result[1] = (a + d) / 2.0;  // Report real part
                    return;
                }
                i = i + 2u;
                continue;
            }
        }

        // 1x1 block (real eigenvalue)
        let eigenvalue = matrix_t[diag_idx];
        if check_real_eigenvalue(eigenvalue, eps) {
            result[0] = 1.0;
            result[1] = eigenvalue;
            return;
        }
        i = i + 1u;
    }
}
