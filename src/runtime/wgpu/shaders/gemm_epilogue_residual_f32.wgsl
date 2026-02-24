// Fused GEMM + bias + residual. F32 only.
// C = A @ B + bias + residual

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

struct GemmResidualParams {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;
@group(0) @binding(5) var<uniform> params: GemmResidualParams;

@compute @workgroup_size(16, 16, 1)
fn gemm_bias_residual_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                           @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = params.M;
    let K = params.K;
    let N = params.N;
    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    var sum: f32 = 0.0;
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = a[row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }
        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = b[b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        let idx = row * N + col;
        c[idx] = sum + bias[col] + residual[idx];
    }
}

@compute @workgroup_size(16, 16, 1)
fn gemm_bias_residual_batched_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = params.M;
    let K = params.K;
    let N = params.N;
    let batch = group_id.z;
    if (batch >= params.batch_size) { return; }

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;
    let a_off = batch * M * K;
    let b_off = batch * K * N;
    let c_off = batch * M * N;

    var sum: f32 = 0.0;
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = a[a_off + row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }
        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = b[b_off + b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        let idx = c_off + row * N + col;
        c[idx] = sum + bias[col] + residual[idx];
    }
}
