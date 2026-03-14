// F32 fill operation

const WORKGROUP_SIZE: u32 = 256u;

struct FillParams {
    numel: u32,
    value: f32,
}

@group(0) @binding(0) var<storage, read_write> fill_out: array<f32>;
@group(0) @binding(1) var<uniform> fill_params: FillParams;

@compute @workgroup_size(256)
fn fill_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < fill_params.numel) {
        fill_out[idx] = fill_params.value;
    }
}
