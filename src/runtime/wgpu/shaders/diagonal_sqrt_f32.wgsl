// Diagonal block function application for f32 - sqrt

const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    n: u32,
    eps: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_f: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Apply sqrt to 2x2 block
fn apply_2x2_block(a: f32, b: f32, c: f32, d: f32,
                   f11: ptr<function, f32>, f12: ptr<function, f32>,
                   f21: ptr<function, f32>, f22: ptr<function, f32>) {
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        // Real eigenvalues
        let sqrt_disc = sqrt(disc);
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        let sqrt1 = sqrt(lambda1);
        let sqrt2 = sqrt(lambda2);

        *f11 = (sqrt1 + sqrt2) / 2.0;
        *f22 = (sqrt1 + sqrt2) / 2.0;
        let denom = sqrt1 + sqrt2;
        if abs(denom) > 1e-10 {
            *f12 = b / denom;
            *f21 = c / denom;
        } else {
            *f12 = 0.0;
            *f21 = 0.0;
        }
    } else {
        // Complex eigenvalues
        let r = sqrt(det);
        let theta = atan2(sqrt(-disc) / 2.0, trace / 2.0);
        let sqrt_r = sqrt(r);
        let half_theta = theta / 2.0;

        *f11 = sqrt_r * cos(half_theta);
        *f22 = sqrt_r * cos(half_theta);
        let imag_part = sqrt(-disc) / 2.0;
        let scale = sqrt_r * sin(half_theta) / imag_part;
        *f12 = scale * b;
        *f21 = scale * c;
    }
}

@compute @workgroup_size(1)
fn diagonal_sqrt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let eps = f32(params.eps);

    // Initialize output to zero
    for (var idx: u32 = 0u; idx < n * n; idx = idx + 1u) {
        output_f[idx] = 0.0;
    }

    var i: u32 = 0u;
    while i < n {
        // Check if this is a 2x2 block
        if i + 1u < n {
            let sub_diag = abs(input_t[(i + 1u) * n + i]);
            if sub_diag > eps {
                // 2x2 block
                let a = input_t[i * n + i];
                let b = input_t[i * n + (i + 1u)];
                let c = input_t[(i + 1u) * n + i];
                let d = input_t[(i + 1u) * n + (i + 1u)];

                var f11: f32;
                var f12: f32;
                var f21: f32;
                var f22: f32;
                apply_2x2_block(a, b, c, d, &f11, &f12, &f21, &f22);

                output_f[i * n + i] = f11;
                output_f[i * n + (i + 1u)] = f12;
                output_f[(i + 1u) * n + i] = f21;
                output_f[(i + 1u) * n + (i + 1u)] = f22;

                i = i + 2u;
                continue;
            }
        }

        // 1x1 block
        let x = input_t[i * n + i];
        output_f[i * n + i] = sqrt(x);
        i = i + 1u;
    }
}
