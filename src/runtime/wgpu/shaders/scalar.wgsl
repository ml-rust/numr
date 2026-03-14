// F32 scalar operations

const WORKGROUP_SIZE: u32 = 256u;

struct ScalarParams {
    numel: u32,
    scalar: f32,
}

@group(0) @binding(0) var<storage, read_write> scalar_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> scalar_out: array<f32>;
@group(0) @binding(2) var<uniform> scalar_params: ScalarParams;

@compute @workgroup_size(256)
fn add_scalar_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] + scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn sub_scalar_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] - scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn rsub_scalar_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_params.scalar - scalar_a[idx];
    }
}

@compute @workgroup_size(256)
fn mul_scalar_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] * scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn div_scalar_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] / scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn pow_scalar_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = pow(scalar_a[idx], scalar_params.scalar);
    }
}

@compute @workgroup_size(256)
fn leaky_relu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        let x = scalar_a[idx];
        let slope = scalar_params.scalar;
        scalar_out[idx] = max(slope * x, x);
    }
}

@compute @workgroup_size(256)
fn elu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        let x = scalar_a[idx];
        let alpha = scalar_params.scalar;
        scalar_out[idx] = select(alpha * (exp(x) - 1.0), x, x > 0.0);
    }
}
