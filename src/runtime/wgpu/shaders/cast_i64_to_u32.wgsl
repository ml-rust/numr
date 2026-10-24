// I64 to U32 cast. Concatenated after cast_i64_common.wgsl.
//
// CPU computes `(v as f64) as u32`, which saturates: a negative value becomes
// 0 and a value above u32::MAX becomes u32::MAX. The f64 step rounds only
// magnitudes at or above 2^53, and those saturate either way.

@group(0) @binding(0) var<storage, read_write> cast_input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<u32>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_i64_to_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = cast_thread_index(gid);
    if (idx >= cast_params.numel) {
        return;
    }
    let v = cast_input[idx];
    var narrowed: u32;
    if ((v.y >> 31u) != 0u) {
        narrowed =0u;
    } else if (v.y != 0u) {
        narrowed =0xffffffffu;
    } else {
        narrowed =v.x;
    }
    cast_output[idx] = narrowed;
}
