// Phase angle of real numbers shader
// entry point: angle_real_f32
// angle(x) = 0 if x >= 0, π if x < 0

struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// PI constant (WGSL has no standard math library, so this is defined literally)
// Value matches std::f32::consts::PI exactly (f32 precision: ~7 significant digits)
const PI: f32 = 3.14159265f;

@compute @workgroup_size(256)
fn angle_real_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        let val = input[idx];
        output[idx] = select(0.0, PI, val < 0.0);  // 0 if x >= 0, π if x < 0
    }
}
