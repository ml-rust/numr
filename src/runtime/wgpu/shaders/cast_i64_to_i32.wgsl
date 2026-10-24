// I64 to I32 cast. Concatenated after cast_i64_common.wgsl.
//
// CPU computes `(v as f64) as i32`, which saturates: a value past the i32
// range clamps to the nearest bound. The f64 step rounds only magnitudes at
// or above 2^53, and those saturate either way, so the result is an exact
// clamp of `v`.

@group(0) @binding(0) var<storage, read_write> cast_input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<u32>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_i64_to_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = cast_thread_index(gid);
    if (idx >= cast_params.numel) {
        return;
    }
    let v = cast_input[idx];
    // A value inside the i32 range has a high word equal to the sign
    // extension of the low word's top bit.
    var narrowed: u32;
    if ((v.y >> 31u) == 0u) {
        if (v.y == 0u && v.x <= 0x7fffffffu) {
            narrowed =v.x;
        } else {
            narrowed =0x7fffffffu;
        }
    } else {
        if (v.y == 0xffffffffu && v.x >= 0x80000000u) {
            narrowed =v.x;
        } else {
            narrowed =0x80000000u;
        }
    }
    cast_output[idx] = narrowed;
}
