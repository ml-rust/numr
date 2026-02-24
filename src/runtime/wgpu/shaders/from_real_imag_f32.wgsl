// Construct Complex64 from real and imaginary parts
// entry point: from_real_imag_f32

struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> real_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> imag_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn from_real_imag_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        output[idx] = vec2<f32>(real_input[idx], imag_input[idx]);
    }
}
