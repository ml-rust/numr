// irfft unpack shader - extracts real part from complex

const WORKGROUP_SIZE: u32 = 256u;

struct UnpackParams {
    n: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> unpack_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> unpack_output: array<f32>;
@group(0) @binding(2) var<uniform> unpack_params: UnpackParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn irfft_unpack(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = unpack_params.n;

    if (idx >= n) {
        return;
    }

    let in_offset = batch_idx * n;
    let out_offset = batch_idx * n;

    unpack_output[out_offset + idx] = unpack_input[in_offset + idx].x;
}
