// F32 to U32 cast operation

const WORKGROUP_SIZE: u32 = 256u;

struct CastParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> cast_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<u32>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_f32_to_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < cast_params.numel) {
        cast_output[idx] = u32(cast_input[idx]);
    }
}
