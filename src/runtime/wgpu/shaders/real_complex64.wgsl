// Complex real-part extraction shader
// entry point: real_complex64

struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn real_complex64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        output[idx] = input[idx].x;  // Extract real component
    }
}
