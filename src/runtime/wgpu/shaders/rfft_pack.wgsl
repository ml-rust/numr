// rfft pack shader - converts real input to complex

const WORKGROUP_SIZE: u32 = 256u;

struct PackParams {
    n: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> pack_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> pack_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> pack_params: PackParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn rfft_pack(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = pack_params.n;

    if (idx >= n) {
        return;
    }

    let in_offset = batch_idx * n;
    let out_offset = batch_idx * n;

    pack_output[out_offset + idx] = vec2<f32>(pack_input[in_offset + idx], 0.0);
}
