// I32 to I64 cast: sign extension. Concatenated after cast_i64_common.wgsl.

@group(0) @binding(0) var<storage, read_write> cast_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_i32_to_i64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = cast_thread_index(gid);
    if (idx >= cast_params.numel) {
        return;
    }
    let w = cast_input[idx];
    var hi = 0u;
    if ((w >> 31u) != 0u) {
        hi = 0xffffffffu;
    }
    cast_output[idx] = vec2<u32>(w, hi);
}
