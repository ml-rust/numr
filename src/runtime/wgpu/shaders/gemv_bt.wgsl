// GEMV-BT: C[M,N] = A[M,K] @ B^T where B is stored as [N,K] row-major.
//
// Each output C[m,n] = dot(A[m,:], B[n,:]) where both vectors are contiguous.
// This avoids copying transposed weight matrices to make them contiguous.
//
// Each workgroup computes one output element using parallel reduction.
// A dispatch dimension holds at most 65535 workgroups. Columns past that
// spill into further chunks of GEMV_GRID_X columns each.

struct GemvBtParams {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read_write> gemv_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> gemv_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> gemv_c: array<f32>;
@group(0) @binding(3) var<uniform> gemv_params: GemvBtParams;

var<workgroup> gemv_shared: array<f32, 256>;

const GEMV_GRID_X: u32 = 65535u;

// 2D GEMV-BT: one workgroup per output element.
// Dispatch: (min(N, GEMV_GRID_X), M, column chunks).
// workgroup_id.x + workgroup_id.z * GEMV_GRID_X = output column (n),
// workgroup_id.y = output row (m).
@compute @workgroup_size(256, 1, 1)
fn gemv_bt_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = gemv_params.M;
    let K = gemv_params.K;
    let N = gemv_params.N;
    let tid = local_id.x;
    let m = group_id.y;
    let n = group_id.x + group_id.z * GEMV_GRID_X;

    if (m >= M || n >= N) {
        return;
    }

    // A is [M, K] row-major, B is [N, K] row-major
    let a_offset = m * K;
    let b_offset = n * K;

    // Each thread computes partial dot product
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < K) {
        sum = sum + gemv_a[a_offset + i] * gemv_b[b_offset + i];
        i = i + 256u;
    }

    gemv_shared[tid] = sum;
    workgroupBarrier();

    // Parallel reduction
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gemv_shared[tid] = gemv_shared[tid] + gemv_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        gemv_c[m * N + n] = gemv_shared[0];
    }
}

// Batched GEMV-BT: workgroup_id.z = batch index.
// Dispatch: (min(N, GEMV_GRID_X), M * column chunks, batch_size).
// workgroup_id.y = chunk * M + m.
@compute @workgroup_size(256, 1, 1)
fn batched_gemv_bt_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = gemv_params.M;
    let K = gemv_params.K;
    let N = gemv_params.N;
    let batch_size = gemv_params.batch_size;
    let tid = local_id.x;
    let m = group_id.y % M;
    let n = group_id.x + (group_id.y / M) * GEMV_GRID_X;
    let batch = group_id.z;

    if (m >= M || n >= N || batch >= batch_size) {
        return;
    }

    let a_offset = batch * M * K + m * K;
    let b_offset = batch * N * K + n * K;

    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < K) {
        sum = sum + gemv_a[a_offset + i] * gemv_b[b_offset + i];
        i = i + 256u;
    }

    gemv_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gemv_shared[tid] = gemv_shared[tid] + gemv_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        gemv_c[batch * M * N + m * N + n] = gemv_shared[0];
    }
}
