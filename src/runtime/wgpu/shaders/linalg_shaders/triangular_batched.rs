//! Batched triangular substitution: one invocation per (matrix, column).
//!
//! F32 only - WGSL doesn't support F64.

/// Batched triangular solve shader: forward_sub_batched_f32, backward_sub_batched_f32
///
/// `a: [batch, n, n]`, `b` and `x: [batch, n, nrhs]`, row-major and
/// contiguous. Invocation `t` solves column `t % nrhs` of matrix
/// `t / nrhs`. Rows run in order inside the invocation, with the same
/// arithmetic as `forward_sub_f32` and `backward_sub_f32`. The dispatch is
/// 2-D so that more than 65535 workgroups fit.
pub const TRIANGULAR_BATCHED_SHADER: &str = r#"
const WORKGROUP_SIZE: u32 = 64u;

struct TriangularBatchedParams {
    n: u32,
    nrhs: u32,
    batch: u32,
    unit_diagonal: u32,
}

@group(0) @binding(0) var<storage, read> tri_a: array<f32>;
@group(0) @binding(1) var<storage, read> tri_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> tri_x: array<f32>;
@group(0) @binding(3) var<uniform> tri_params: TriangularBatchedParams;

fn tri_thread(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.y * (nwg.x * WORKGROUP_SIZE) + gid.x;
}

@compute @workgroup_size(64)
fn forward_sub_batched_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = tri_params.n;
    let nrhs = tri_params.nrhs;
    let t = tri_thread(gid, nwg);
    if (t >= tri_params.batch * nrhs) {
        return;
    }
    let col = t % nrhs;
    let mat_idx = t / nrhs;
    let a_base = mat_idx * n * n;
    let x_base = mat_idx * n * nrhs;
    let unit_diag = tri_params.unit_diagonal != 0u;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        var sum: f32 = tri_b[x_base + i * nrhs + col];
        for (var j: u32 = 0u; j < i; j = j + 1u) {
            sum = sum - tri_a[a_base + i * n + j] * tri_x[x_base + j * nrhs + col];
        }
        if (unit_diag) {
            tri_x[x_base + i * nrhs + col] = sum;
        } else {
            tri_x[x_base + i * nrhs + col] = sum / tri_a[a_base + i * n + i];
        }
    }
}

@compute @workgroup_size(64)
fn backward_sub_batched_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = tri_params.n;
    let nrhs = tri_params.nrhs;
    let t = tri_thread(gid, nwg);
    if (t >= tri_params.batch * nrhs) {
        return;
    }
    let col = t % nrhs;
    let mat_idx = t / nrhs;
    let a_base = mat_idx * n * n;
    let x_base = mat_idx * n * nrhs;

    for (var ii: u32 = 0u; ii < n; ii = ii + 1u) {
        let i = n - 1u - ii;
        var sum: f32 = tri_b[x_base + i * nrhs + col];
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            sum = sum - tri_a[a_base + i * n + j] * tri_x[x_base + j * nrhs + col];
        }
        tri_x[x_base + i * nrhs + col] = sum / tri_a[a_base + i * n + i];
    }
}
"#;
