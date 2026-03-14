// Copy complex array

const WORKGROUP_SIZE: u32 = 256u;

struct CopyParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> copy_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> copy_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> copy_params: CopyParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn copy_complex(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let n = copy_params.n;

    if (idx < n) {
        copy_output[idx] = copy_input[idx];
    }
}
