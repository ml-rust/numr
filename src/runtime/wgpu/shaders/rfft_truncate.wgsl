// rfft truncate shader - keeps only N/2+1 complex values from full FFT

const WORKGROUP_SIZE: u32 = 256u;

struct TruncateParams {
    n: u32,         // Full FFT size (input)
    half_n: u32,    // N/2 + 1 (output size)
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> truncate_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> truncate_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> truncate_params: TruncateParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn rfft_truncate(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = truncate_params.n;
    let half_n = truncate_params.half_n;

    if (idx >= half_n) {
        return;
    }

    let in_offset = batch_idx * n;
    let out_offset = batch_idx * half_n;

    truncate_output[out_offset + idx] = truncate_input[in_offset + idx];
}
