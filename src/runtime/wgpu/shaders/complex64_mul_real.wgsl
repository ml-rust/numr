// Complex × real multiplication shader
// entry point: complex64_mul_real
// (a + bi) * r = ar + br*i

struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> complex_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> real_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn complex64_mul_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        let c = complex_input[idx];
        let r = real_input[idx];
        output[idx] = vec2<f32>(c.x * r, c.y * r);
    }
}
