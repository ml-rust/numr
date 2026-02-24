// Semiring matmul: min_max for f32
// C[i,j] = min_k( max(A[i,k], B[k,j]) )
// Entry points: semiring_matmul_min_max_f32, batched_semiring_matmul_min_max_f32

struct SemiringMatmulParams {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read_write> sr_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> sr_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> sr_c: array<f32>;
@group(0) @binding(3) var<uniform> sr_params: SemiringMatmulParams;

fn sr_combine(a: f32, b: f32) -> f32 {
    return max(a, b);
}

fn sr_reduce(acc: f32, val: f32) -> f32 {
    return min(acc, val);
}

@compute @workgroup_size(16, 16, 1)
fn semiring_matmul_min_max_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = sr_params.M;
    let K = sr_params.K;
    let N = sr_params.N;

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    var acc: f32 = bitcast<f32>(0x7f800000u);

    for (var kk: u32 = 0u; kk < K; kk = kk + 1u) {
        let a_val = sr_a[row * K + kk];
        let b_val = sr_b[kk * N + col];
        acc = sr_reduce(acc, sr_combine(a_val, b_val));
    }

    sr_c[row * N + col] = acc;
}

@compute @workgroup_size(16, 16, 1)
fn batched_semiring_matmul_min_max_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = sr_params.M;
    let K = sr_params.K;
    let N = sr_params.N;
    let batch_size = sr_params.batch_size;

    let batch = global_id.z;
    if (batch >= batch_size) {
        return;
    }

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    let a_offset = batch * M * K;
    let b_offset = batch * K * N;
    let c_offset = batch * M * N;

    var acc: f32 = bitcast<f32>(0x7f800000u);

    for (var kk: u32 = 0u; kk < K; kk = kk + 1u) {
        let a_val = sr_a[a_offset + row * K + kk];
        let b_val = sr_b[b_offset + kk * N + col];
        acc = sr_reduce(acc, sr_combine(a_val, b_val));
    }

    sr_c[c_offset + row * N + col] = acc;
}
