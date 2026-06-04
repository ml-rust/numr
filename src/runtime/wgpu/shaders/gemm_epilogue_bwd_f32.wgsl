// Backward pass for fused GEMM + bias + activation. F32 only.
//
// Given output = activation(A @ B + bias), with pre = A @ B + bias:
//   grad_pre = grad * activation'(pre)         [batch, M, N]
//   d_a      = grad_pre @ B^T                  [batch, M, K]
//   d_b      = A^T @ grad_pre  (sum over batch)[K, N]
//   d_bias   = sum(grad_pre, dim=0,batch)      [N]
//
// activation_type: 0=None, 1=ReLU, 2=GELU, 3=SiLU, 4=Sigmoid, 5=Tanh
//
// All four entry points share one bind-group layout (8 storage + 1 uniform);
// each entry point only references the bindings it needs. Runs as four
// sequential compute passes so later passes observe `grad_pre`.

const TILE_SIZE: u32 = 16u;

struct GemmEpilogueParams {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
    activation_type: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_pre: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_b: array<f32>;
@group(0) @binding(7) var<storage, read_write> d_bias: array<f32>;
@group(0) @binding(8) var<uniform> params: GemmEpilogueParams;

var<workgroup> tile_x: array<array<f32, 16>, 16>;
var<workgroup> tile_y: array<array<f32, 16>, 16>;

fn activation_derivative(x: f32, act_type: u32) -> f32 {
    switch act_type {
        case 1u: { // ReLU
            if (x > 0.0) { return 1.0; }
            return 0.0;
        }
        case 2u: { // GELU (tanh approximation)
            let s = 0.7978845608028654;
            let co = 0.044715;
            let inner = s * (x + co * x * x * x);
            let t = tanh(inner);
            let sech2 = 1.0 - t * t;
            let d_inner = s * (1.0 + 3.0 * co * x * x);
            return 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner;
        }
        case 3u: { // SiLU
            let sig = 1.0 / (1.0 + exp(-x));
            return sig + x * sig * (1.0 - sig);
        }
        case 4u: { // Sigmoid
            let sig = 1.0 / (1.0 + exp(-x));
            return sig * (1.0 - sig);
        }
        case 5u: { // Tanh
            let t = tanh(x);
            return 1.0 - t * t;
        }
        default: { // None (identity)
            return 1.0;
        }
    }
}

// grad_pre[b, m, n] = grad[b, m, n] * activation'(A[b] @ B[b] + bias)[m, n]
@compute @workgroup_size(16, 16, 1)
fn gemm_bwd_grad_pre_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                         @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = params.M;
    let K = params.K;
    let N = params.N;
    let batch = group_id.z;
    if (batch >= params.batch_size) { return; }

    let row = group_id.y * TILE_SIZE + local_id.y; // m
    let col = group_id.x * TILE_SIZE + local_id.x; // n
    let a_off = batch * M * K;
    let b_off = batch * K * N;
    let out_off = batch * M * N;

    var sum: f32 = 0.0;
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_x[local_id.y][local_id.x] = a[a_off + row * K + a_col];
        } else {
            tile_x[local_id.y][local_id.x] = 0.0;
        }
        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_y[local_id.y][local_id.x] = b[b_off + b_row * N + col];
        } else {
            tile_y[local_id.y][local_id.x] = 0.0;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < TILE_SIZE; kk = kk + 1u) {
            sum = sum + tile_x[local_id.y][kk] * tile_y[kk][local_id.x];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        let pre = sum + bias[col];
        var deriv = activation_derivative(pre, params.activation_type);
        // Guard against non-finite derivatives (NaN: self-compare fails; Inf: huge magnitude)
        if (deriv != deriv || abs(deriv) > 3.4e38) {
            deriv = 0.0;
        }
        grad_pre[out_off + row * N + col] = grad[out_off + row * N + col] * deriv;
    }
}

// d_a[b, m, k] = sum_n grad_pre[b, m, n] * B[b, k, n]
@compute @workgroup_size(16, 16, 1)
fn gemm_bwd_da_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = params.M;
    let K = params.K;
    let N = params.N;
    let batch = group_id.z;
    if (batch >= params.batch_size) { return; }

    let row = group_id.y * TILE_SIZE + local_id.y; // m
    let col = group_id.x * TILE_SIZE + local_id.x; // k
    let gp_off = batch * M * N;
    let b_off = batch * K * N;
    let out_off = batch * M * K;

    var sum: f32 = 0.0;
    let num_tiles = (N + TILE_SIZE - 1u) / TILE_SIZE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let gp_col = t * TILE_SIZE + local_id.x; // n
        if (row < M && gp_col < N) {
            tile_x[local_id.y][local_id.x] = grad_pre[gp_off + row * N + gp_col];
        } else {
            tile_x[local_id.y][local_id.x] = 0.0;
        }
        // tile_y[ly][lx] = B[k=col, n = t*T+ly]  (B^T access)
        let b_n = t * TILE_SIZE + local_id.y; // n
        if (col < K && b_n < N) {
            tile_y[local_id.y][local_id.x] = b[b_off + col * N + b_n];
        } else {
            tile_y[local_id.y][local_id.x] = 0.0;
        }
        workgroupBarrier();
        for (var nn: u32 = 0u; nn < TILE_SIZE; nn = nn + 1u) {
            sum = sum + tile_x[local_id.y][nn] * tile_y[nn][local_id.x];
        }
        workgroupBarrier();
    }

    if (row < M && col < K) {
        d_a[out_off + row * K + col] = sum;
    }
}

// d_b[k, n] = sum_batch sum_m A[b, m, k] * grad_pre[b, m, n]
@compute @workgroup_size(16, 16, 1)
fn gemm_bwd_db_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = params.M;
    let K = params.K;
    let N = params.N;

    let row = group_id.y * TILE_SIZE + local_id.y; // k
    let col = group_id.x * TILE_SIZE + local_id.x; // n

    var sum: f32 = 0.0;
    let num_tiles = (M + TILE_SIZE - 1u) / TILE_SIZE;
    for (var batch: u32 = 0u; batch < params.batch_size; batch = batch + 1u) {
        let a_off = batch * M * K;
        let gp_off = batch * M * N;
        for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
            // tile_x[ly][lx] = A[b, m = t*T+lx, k = row]  (A^T access)
            let a_m = t * TILE_SIZE + local_id.x; // m
            if (row < K && a_m < M) {
                tile_x[local_id.y][local_id.x] = a[a_off + a_m * K + row];
            } else {
                tile_x[local_id.y][local_id.x] = 0.0;
            }
            // tile_y[ly][lx] = grad_pre[b, m = t*T+ly, n = col]
            let gp_m = t * TILE_SIZE + local_id.y; // m
            if (gp_m < M && col < N) {
                tile_y[local_id.y][local_id.x] = grad_pre[gp_off + gp_m * N + col];
            } else {
                tile_y[local_id.y][local_id.x] = 0.0;
            }
            workgroupBarrier();
            for (var mm: u32 = 0u; mm < TILE_SIZE; mm = mm + 1u) {
                sum = sum + tile_x[local_id.y][mm] * tile_y[mm][local_id.x];
            }
            workgroupBarrier();
        }
    }

    if (row < K && col < N) {
        d_b[row * N + col] = sum;
    }
}

// d_bias[n] = sum_batch sum_m grad_pre[b, m, n]
@compute @workgroup_size(256, 1, 1)
fn gemm_bwd_dbias_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = params.M;
    let N = params.N;
    let n = gid.x;
    if (n >= N) { return; }

    var sum: f32 = 0.0;
    for (var batch: u32 = 0u; batch < params.batch_size; batch = batch + 1u) {
        let gp_off = batch * M * N;
        for (var m: u32 = 0u; m < M; m = m + 1u) {
            sum = sum + grad_pre[gp_off + m * N + n];
        }
    }
    d_bias[n] = sum;
}
