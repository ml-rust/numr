// F32 binary operations

const WORKGROUP_SIZE: u32 = 256u;

struct BinaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> binary_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> binary_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> binary_out: array<f32>;
@group(0) @binding(3) var<uniform> binary_params: BinaryParams;

@compute @workgroup_size(256)
fn add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] + binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn sub_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] - binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] * binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn div_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = binary_a[idx] / binary_b[idx];
    }
}

@compute @workgroup_size(256)
fn max_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = max(binary_a[idx], binary_b[idx]);
    }
}

@compute @workgroup_size(256)
fn min_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = min(binary_a[idx], binary_b[idx]);
    }
}

@compute @workgroup_size(256)
fn pow_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = pow(binary_a[idx], binary_b[idx]);
    }
}

@compute @workgroup_size(256)
fn atan2_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < binary_params.numel) {
        binary_out[idx] = atan2(binary_a[idx], binary_b[idx]);
    }
}
