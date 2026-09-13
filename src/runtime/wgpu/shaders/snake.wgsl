// Snake activation WGSL shaders (F32 only)
// Forward: out = x + sin(alpha * x)^2 / (beta + eps)
// Backward: d_x elementwise, d_alpha / d_beta one workgroup per channel
//
// x, out and grad are contiguous [outer, channels, inner] buffers; alpha and
// beta hold one value per channel. The channel of flat element i is
// (i / inner) % channels. The per-channel reduction folds a fixed
// shared-memory tree, so a repeated dispatch returns identical bits.

const WORKGROUP_SIZE: u32 = 256u;

// Single-precision sine with proper argument reduction.
//
// The built-in sin() is a driver-supplied fast approximation whose absolute
// error grows with |x|. Snake's argument alpha * x is routinely tens of
// radians, where that approximation diverges from the CPU kernel. Cody-Waite
// reduction by pi in three parts (each part has few enough bits that q * part
// is exact for q below 2^12, so |x| below about 1.2e4 reduces exactly), then
// an odd minimax polynomial on [-pi/2, pi/2].
fn snake_sin(x: f32) -> f32 {
    let q = round(x * 0.318309886183790671538);
    var r = fma(q, -3.1414794921875, x);
    r = fma(q, -0.00011315941810607910156, r);
    r = fma(q, -1.9841872589410058936e-09, r);
    if ((i32(q) & 1) != 0) {
        r = -r;
    }
    let s = r * r;
    var u = 2.6083159809786593541503e-06;
    u = fma(u, s, -0.0001981069071916863322258);
    u = fma(u, s, 0.00833307858556509017944336);
    u = fma(u, s, -0.166666597127914428710938);
    return fma(s, u * r, r);
}

struct SnakeParams {
    // `numel` for the elementwise entry points, `outer` for the reduction.
    extent: u32,
    channels: u32,
    inner: u32,
    eps: f32,
}

// ============================================================================
// Forward: x, alpha, beta -> out
// ============================================================================

@group(0) @binding(0) var<storage, read_write> fwd_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> fwd_alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> fwd_beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> fwd_out: array<f32>;
@group(0) @binding(4) var<uniform> fwd_params: SnakeParams;

@compute @workgroup_size(256)
fn snake_beta_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= fwd_params.extent) {
        return;
    }
    let c = (idx / fwd_params.inner) % fwd_params.channels;
    let a = fwd_alpha[c];
    let inv = 1.0 / (fwd_beta[c] + fwd_params.eps);
    let xv = fwd_x[idx];
    let s = snake_sin(a * xv);
    fwd_out[idx] = xv + s * s * inv;
}

// ============================================================================
// Backward d_x: grad, x, alpha, beta -> d_x
// ============================================================================

@group(0) @binding(0) var<storage, read_write> dx_grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> dx_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx_alpha: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx_beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx_out: array<f32>;
@group(0) @binding(5) var<uniform> dx_params: SnakeParams;

@compute @workgroup_size(256)
fn snake_beta_dx_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dx_params.extent) {
        return;
    }
    let c = (idx / dx_params.inner) % dx_params.channels;
    let a = dx_alpha[c];
    let inv = 1.0 / (dx_beta[c] + dx_params.eps);
    let xv = dx_x[idx];
    let s2 = snake_sin(2.0 * a * xv);
    dx_out[idx] = dx_grad[idx] * (1.0 + a * s2 * inv);
}

// ============================================================================
// Backward params: grad, x, alpha, beta -> d_alpha, d_beta (one group per channel)
// ============================================================================

@group(0) @binding(0) var<storage, read_write> dp_grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> dp_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> dp_alpha: array<f32>;
@group(0) @binding(3) var<storage, read_write> dp_beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> dp_d_alpha: array<f32>;
@group(0) @binding(5) var<storage, read_write> dp_d_beta: array<f32>;
@group(0) @binding(6) var<uniform> dp_params: SnakeParams;

var<workgroup> dp_shared_alpha: array<f32, 256>;
var<workgroup> dp_shared_beta: array<f32, 256>;

@compute @workgroup_size(256)
fn snake_beta_dparams_f32(@builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let c = group_id.x;
    let channels = dp_params.channels;
    let inner = dp_params.inner;
    let span = dp_params.extent * inner;

    let a = dp_alpha[c];
    let inv = 1.0 / (dp_beta[c] + dp_params.eps);
    let inv_sq = inv * inv;

    var sum_alpha: f32 = 0.0;
    var sum_beta: f32 = 0.0;
    var j: u32 = tid;
    while (j < span) {
        let o = j / inner;
        let i = j - o * inner;
        let idx = (o * channels + c) * inner + i;
        let xv = dp_x[idx];
        let g = dp_grad[idx];
        let ax = a * xv;
        let s = snake_sin(ax);
        let s2 = snake_sin(2.0 * ax);
        sum_alpha = sum_alpha + g * xv * s2 * inv;
        sum_beta = sum_beta - g * s * s * inv_sq;
        j = j + WORKGROUP_SIZE;
    }

    dp_shared_alpha[tid] = sum_alpha;
    dp_shared_beta[tid] = sum_beta;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            dp_shared_alpha[tid] = dp_shared_alpha[tid] + dp_shared_alpha[tid + s];
            dp_shared_beta[tid] = dp_shared_beta[tid] + dp_shared_beta[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        dp_d_alpha[c] = dp_shared_alpha[0];
        dp_d_beta[c] = dp_shared_beta[0];
    }
}
