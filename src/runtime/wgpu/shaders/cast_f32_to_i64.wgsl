// F32 to I64 cast. Concatenated after cast_i64_common.wgsl.
//
// CPU computes `(v as f64) as i64`. The f64 step is exact, so the result is
// Rust's saturating `as`: truncate toward zero, NaN becomes 0, and a value
// at or past +-2^63 (infinities included) clamps to i64::MAX or i64::MIN.
// The kernel decodes the f32 bits directly, so no float-to-int instruction
// with backend-defined out-of-range behavior runs.

@group(0) @binding(0) var<storage, read_write> cast_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> cast_output: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> cast_params: CastParams;

@compute @workgroup_size(256)
fn cast_f32_to_i64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = cast_thread_index(gid);
    if (idx >= cast_params.numel) {
        return;
    }
    let bits = cast_input[idx];
    let negative = (bits >> 31u) != 0u;
    let biased = (bits >> 23u) & 0xffu;
    let frac = bits & 0x7fffffu;

    // NaN, and every magnitude below 1 (zeros and subnormals included).
    if ((biased == 0xffu && frac != 0u) || biased < 127u) {
        cast_output[idx] = vec2<u32>(0u, 0u);
        return;
    }
    let e = biased - 127u;
    if (e >= 63u) {
        if (negative) {
            cast_output[idx] = vec2<u32>(0u, 0x80000000u);
        } else {
            cast_output[idx] = vec2<u32>(0xffffffffu, 0x7fffffffu);
        }
        return;
    }
    // |v| = mant * 2^(e - 23). Below e = 23 the shift drops the fraction bits,
    // which truncates toward zero.
    let mant = frac | 0x800000u;
    var mag: vec2<u32>;
    if (e <= 23u) {
        mag = vec2<u32>(mant >> (23u - e), 0u);
    } else {
        mag = u64_shl(vec2<u32>(mant, 0u), e - 23u);
    }
    if (negative) {
        mag = u64_neg(mag);
    }
    cast_output[idx] = mag;
}
