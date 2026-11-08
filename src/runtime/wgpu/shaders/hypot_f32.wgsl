// Integer storage preserves subnormal and non-finite input bits.
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

// Two-limb integers use (low, high). Inputs contain at most 24 bits.
fn square24(x: u32) -> vec2<u32> {
    let low = x & 0xffffu;
    let high = x >> 16u;
    let base = low * low;
    let cross = 2u * low * high;
    let lo = base + (cross << 16u);
    let carry = select(0u, 1u, lo < base);
    return vec2<u32>(lo, high * high + (cross >> 16u) + carry);
}

fn wide_add(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let lo = x.x + y.x;
    return vec2<u32>(lo, x.y + y.y + select(0u, 1u, lo < x.x));
}

fn wide_le(x: vec2<u32>, y: vec2<u32>) -> bool {
    return x.y < y.y || (x.y == y.y && x.x <= y.x);
}

// Fractions count units of 2^-149. Integer sqrt avoids double rounding.
fn subnormal_hypot(x: u32, y: u32) -> u32 {
    let sum = wide_add(square24(x), square24(y));
    var q = 0u;
    for (var bit = 23; bit >= 0; bit = bit - 1) {
        let candidate = q | (1u << u32(bit));
        if (wide_le(square24(candidate), sum)) {
            q = candidate;
        }
    }
    // sqrt(sum) > q + 1/2 iff sum > q*q + q. Equality at half is impossible.
    let midpoint = wide_add(square24(q), vec2<u32>(q, 0u));
    if (!wide_le(sum, midpoint)) {
        q = q + 1u;
    }
    // This integer encoding also covers the first normal binade.
    return q;
}

// Decode a positive finite value as mantissa in [1, 2) times 2^exponent.
fn normalized(bits: u32) -> vec2<i32> {
    let field = bits >> 23u;
    let fraction = bits & 0x7fffffu;
    if (field != 0u) {
        return vec2<i32>(i32(0x3f800000u | fraction), i32(field) - 127);
    }
    let shift = countLeadingZeros(fraction) - 8u;
    let mantissa = 0x3f800000u | ((fraction << shift) & 0x7fffffu);
    return vec2<i32>(i32(mantissa), -126 - i32(shift));
}

fn magnitude_bits(a_bits: u32, b_bits: u32) -> u32 {
    let x = a_bits & 0x7fffffffu;
    let y = b_bits & 0x7fffffffu;
    if (x == 0x7f800000u || y == 0x7f800000u) {
        return 0x7f800000u;
    }
    if (x > 0x7f800000u || y > 0x7f800000u) {
        return 0x7fc00000u;
    }
    let high = max(x, y);
    let low = min(x, y);
    if (low == 0u) {
        return high;
    }
    if (high < 0x800000u) {
        return subnormal_hypot(high, low);
    }
    let h = normalized(high);
    let l = normalized(low);
    let distance = h.y - l.y;
    if (distance > 30) {
        return high;
    }
    let high_mantissa = bitcast<f32>(u32(h.x));
    let low_mantissa = bitcast<f32>(u32(l.x));
    let scale = bitcast<f32>(u32(127 - distance) << 23u);
    let scaled_low = low_mantissa * scale;
    let norm = sqrt(high_mantissa * high_mantissa + scaled_low * scaled_low);
    let norm_bits = bitcast<u32>(norm);
    let exponent = h.y + i32(norm_bits >> 23u);
    if (exponent >= 255) {
        return 0x7f800000u;
    }
    return (u32(exponent) << 23u) | (norm_bits & 0x7fffffu);
}

@compute @workgroup_size(256)
fn hypot_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = gid.x + gid.y * groups.x * 256u;
    if (index >= params[0]) {
        return;
    }
    let rank = params[1];
    var a_offset = params[2];
    var b_offset = params[3];
    var remaining = index;
    for (var dimension = i32(rank) - 1; dimension >= 0; dimension = dimension - 1) {
        let d = u32(dimension);
        let extent = params[4u + d];
        let coordinate = remaining % extent;
        remaining = remaining / extent;
        a_offset = a_offset + coordinate * params[4u + rank + d];
        b_offset = b_offset + coordinate * params[4u + 2u * rank + d];
    }
    output[index] = magnitude_bits(a[a_offset], b[b_offset]);
}
