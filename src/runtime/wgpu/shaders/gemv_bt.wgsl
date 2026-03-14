// GEMV-BT: C[M,N] = A[M,K] @ B^T where B is stored as [N,K] row-major.
//
// Each output C[m,n] = dot(A[m,:], B[n,:]) where both vectors are contiguous.
// This avoids copying transposed weight matrices to make them contiguous.
//
// Dispatch: workgroups(N, M, batch_size) with workgroup_size(256, 1, 1)
// Each workgroup computes one output element using parallel reduction.

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

// 2D GEMV-BT: one workgroup per output element
// workgroup_id.x = output column (n), workgroup_id.y = output row (m)
@compute @workgroup_size(256, 1, 1)
fn gemv_bt_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = gemv_params.M;
    let K = gemv_params.K;
    let N = gemv_params.N;
    let tid = local_id.x;
    let m = group_id.y;
    let n = group_id.x;

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

// Batched GEMV-BT: workgroup_id.z = batch index
@compute @workgroup_size(256, 1, 1)
fn batched_gemv_bt_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = gemv_params.M;
    let K = gemv_params.K;
    let N = gemv_params.N;
    let batch_size = gemv_params.batch_size;
    let tid = local_id.x;
    let m = group_id.y;
    let n = group_id.x;
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
