// Fused activation-mul WGSL shaders (F32 only)
// Forward: out = activation(a) * b
// Backward: d_a = grad * b * activation'(a), d_b = grad * activation(a)

// ============================================================================
// Forward kernels: 2 inputs (a, b), 1 output, uniform params
// ============================================================================

struct FusedFwdParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> fwd_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> fwd_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> fwd_out: array<f32>;
@group(0) @binding(3) var<uniform> fwd_params: FusedFwdParams;

@compute @workgroup_size(256)
fn silu_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < fwd_params.numel) {
        let x = fwd_a[idx];
        let sig = 1.0 / (1.0 + exp(-x));
        fwd_out[idx] = x * sig * fwd_b[idx];
    }
}

@compute @workgroup_size(256)
fn gelu_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < fwd_params.numel) {
        let x = fwd_a[idx];
        let c = 0.7978845608;
        let k = 0.044715;
        let inner = c * (x + k * x * x * x);
        let t = tanh(inner);
        fwd_out[idx] = 0.5 * x * (1.0 + t) * fwd_b[idx];
    }
}

@compute @workgroup_size(256)
fn relu_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < fwd_params.numel) {
        fwd_out[idx] = max(0.0, fwd_a[idx]) * fwd_b[idx];
    }
}

@compute @workgroup_size(256)
fn sigmoid_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < fwd_params.numel) {
        let sig = 1.0 / (1.0 + exp(-fwd_a[idx]));
        fwd_out[idx] = sig * fwd_b[idx];
    }
}

// ============================================================================
// Backward kernels: 3 inputs (grad, a, b), 2 outputs (d_a, d_b), uniform params
// ============================================================================

struct FusedBwdParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> bwd_grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> bwd_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> bwd_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> bwd_d_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> bwd_d_b: array<f32>;
@group(0) @binding(5) var<uniform> bwd_params: FusedBwdParams;

// silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
@compute @workgroup_size(256)
fn silu_mul_bwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < bwd_params.numel) {
        let x = bwd_a[idx];
        let g = bwd_grad[idx];
        let bv = bwd_b[idx];
        let sig = 1.0 / (1.0 + exp(-x));
        let silu_val = x * sig;
        let silu_deriv = sig * (1.0 + x * (1.0 - sig));
        bwd_d_b[idx] = g * silu_val;
        bwd_d_a[idx] = g * bv * silu_deriv;
    }
}

// gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t*t) * c * (1 + 3*k*x*x)
@compute @workgroup_size(256)
fn gelu_mul_bwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < bwd_params.numel) {
        let x = bwd_a[idx];
        let g = bwd_grad[idx];
        let bv = bwd_b[idx];
        let c = 0.7978845608;
        let k = 0.044715;
        let inner = c * (x + k * x * x * x);
        let t = tanh(inner);
        let gelu_val = 0.5 * x * (1.0 + t);
        let gelu_deriv = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * k * x * x);
        bwd_d_b[idx] = g * gelu_val;
        bwd_d_a[idx] = g * bv * gelu_deriv;
    }
}

// relu'(x) = 1 if x > 0, else 0
@compute @workgroup_size(256)
fn relu_mul_bwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < bwd_params.numel) {
        let x = bwd_a[idx];
        let g = bwd_grad[idx];
        let bv = bwd_b[idx];
        let relu_val = max(0.0, x);
        let relu_deriv = select(0.0, 1.0, x > 0.0);
        bwd_d_b[idx] = g * relu_val;
        bwd_d_a[idx] = g * bv * relu_deriv;
    }
}

// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
@compute @workgroup_size(256)
fn sigmoid_mul_bwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < bwd_params.numel) {
        let x = bwd_a[idx];
        let g = bwd_grad[idx];
        let bv = bwd_b[idx];
        let sig = 1.0 / (1.0 + exp(-x));
        let sig_deriv = sig * (1.0 - sig);
        bwd_d_b[idx] = g * sig;
        bwd_d_a[idx] = g * bv * sig_deriv;
    }
}
