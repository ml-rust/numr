// Hermitian extend shader - extends N/2+1 complex to N complex using symmetry

const WORKGROUP_SIZE: u32 = 256u;

struct ExtendParams {
    n: u32,         // Full FFT size
    half_n: u32,    // N/2 + 1 (input size)
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> extend_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> extend_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> extend_params: ExtendParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn hermitian_extend(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = extend_params.n;
    let half_n = extend_params.half_n;

    if (idx >= n) {
        return;
    }

    let in_offset = batch_idx * half_n;
    let out_offset = batch_idx * n;

    if (idx < half_n) {
        // Direct copy for first half
        extend_output[out_offset + idx] = extend_input[in_offset + idx];
    } else {
        // Conjugate symmetry for second half: X[N-k] = conj(X[k])
        let k = n - idx;
        let val = extend_input[in_offset + k];
        extend_output[out_offset + idx] = vec2<f32>(val.x, -val.y);
    }
}
