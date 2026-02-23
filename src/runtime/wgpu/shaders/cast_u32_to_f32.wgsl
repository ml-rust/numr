// U32 to F32 cast operation

const WORKGROUP_SIZE: u32 = 256u;

struct CastParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> cast_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<f32>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_u32_to_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < cast_params.numel) {
        cast_output[idx] = f32(cast_input[idx]);
    }
}
