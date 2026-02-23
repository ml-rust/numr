// F32 comparison operations (input F32, output F32: 1.0=true, 0.0=false)

const WORKGROUP_SIZE: u32 = 256u;

struct CompareParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> compare_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> compare_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> compare_out: array<f32>;
@group(0) @binding(3) var<uniform> compare_params: CompareParams;

@compute @workgroup_size(256)
fn eq_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] == compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn ne_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] != compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn lt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] < compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn le_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] <= compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn gt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] > compare_b[idx]);
    }
}

@compute @workgroup_size(256)
fn ge_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < compare_params.numel) {
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] >= compare_b[idx]);
    }
}
