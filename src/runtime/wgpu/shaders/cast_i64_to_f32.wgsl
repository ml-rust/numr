// I64 to F32 cast. Concatenated after cast_i64_common.wgsl.
//
// CPU computes `(v as f64) as f32`: two roundings, ties to even, first to
// f64's 53-bit significand and then to f32's 24 bits. Rounding once to 24
// bits differs where the first rounding lands on an exact tie, so the
// kernel rounds twice too. Every i64 magnitude is far inside the f32 normal
// range, so no subnormal or overflow case exists.

@group(0) @binding(0) var<storage, read_write> cast_input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<u32>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_i64_to_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = cast_thread_index(gid);
    if (idx >= cast_params.numel) {
        return;
    }
    let v = cast_input[idx];
    let negative = (v.y >> 31u) != 0u;
    // i64::MIN negates to itself, which read unsigned is its magnitude 2^63.
    var mag = v;
    if (negative) {
        mag = u64_neg(v);
    }
    if (mag.x == 0u && mag.y == 0u) {
        cast_output[idx] = 0u;
        return;
    }
    let as_f64 = u64_round_to_bits(mag, 53u);
    let as_f32 = u64_round_to_bits(as_f64, 24u);
    var sign_bit = 0u;
    if (negative) {
        sign_bit = 0x80000000u;
    }
    cast_output[idx] = sign_bit | u64_to_f32_bits(as_f32);
}
