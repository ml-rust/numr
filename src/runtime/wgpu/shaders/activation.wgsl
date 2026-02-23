// F32 clamp operation

const WORKGROUP_SIZE: u32 = 256u;

struct ClampParams {
    numel: u32,
    min_val: f32,
    max_val: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> clamp_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> clamp_out: array<f32>;
@group(0) @binding(2) var<uniform> clamp_params: ClampParams;

@compute @workgroup_size(256)
fn clamp_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < clamp_params.numel) {
        clamp_out[idx] = clamp(clamp_a[idx], clamp_params.min_val, clamp_params.max_val);
    }
}
