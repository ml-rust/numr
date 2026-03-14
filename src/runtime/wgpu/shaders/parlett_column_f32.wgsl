// Parlett recurrence for off-diagonal elements - f32

const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    n: u32,
    col: u32,  // Current column being processed
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> input_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_f: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WORKGROUP_SIZE)
fn parlett_column_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let j = params.col;
    let eps = f32(params.eps);

    // Each thread handles one row i < j
    let i = gid.x;
    if i >= j {
        return;
    }

    let t_ii = input_t[i * n + i];
    let t_jj = input_t[j * n + j];
    let t_ij = input_t[i * n + j];

    let denom = t_ii - t_jj;

    // Compute the sum term
    var sum: f32 = 0.0;
    for (var k: u32 = i + 1u; k < j; k = k + 1u) {
        let f_ik = output_f[i * n + k];
        let t_kj = input_t[k * n + j];
        let t_ik = input_t[i * n + k];
        let f_kj = output_f[k * n + j];
        sum = sum + f_ik * t_kj - t_ik * f_kj;
    }

    let f_ii = output_f[i * n + i];
    let f_jj = output_f[j * n + j];

    // F[i,j] = (T[i,j] * (F[i,i] - F[j,j]) + sum) / (T[i,i] - T[j,j])
    if abs(denom) > eps {
        output_f[i * n + j] = (t_ij * (f_ii - f_jj) + sum) / denom;
    } else {
        // Eigenvalues too close - use limit formula
        output_f[i * n + j] = t_ij * f_ii;  // Simplified fallback
    }
}
