// scatter_reduce reductions for f32.
//
// Concatenated after scatter_reduce_common.wgsl, whose bindings, params and
// run search this file builds on.
//
// The accumulator is the element type, as on CPU. Addition, multiplication
// and comparison are correctly rounded in WGSL, so folding in source order
// gives CPU's bits. Division is not: WGSL allows 2.5 ULP. So `mean` divides
// with sr_mean_div below, which needs no float division at all.

@group(0) @binding(0) var<storage, read> sr_src: array<f32>;
@group(0) @binding(3) var<storage, read> sr_dst: array<f32>;
@group(0) @binding(4) var<storage, read_write> sr_out: array<f32>;

// Seed of a destination element: its own value when include_self is set,
// otherwise the reduction's identity.
fn sr_seed(d: u32, op: u32) -> f32 {
    if (sr_params.include_self != 0u) {
        return sr_dst[d];
    }
    if (op == SR_PROD) {
        return 1.0;
    }
    if (op == SR_MAX) {
        return bitcast<f32>(0xff800000u);
    }
    if (op == SR_MIN) {
        return bitcast<f32>(0x7f800000u);
    }
    return 0.0;
}

// `x >> s` rounded to nearest, ties to even, for a 64-bit `x` held as
// (hi, lo) and 29 <= s <= 53. The result fits in 25 bits.
fn sr_shr_rne(hi: u32, lo: u32, s: u32) -> u32 {
    var r: u32;
    var rem_hi = 0u;
    var rem_lo: u32;
    var half_hi = 0u;
    var half_lo = 0u;
    if (s < 32u) {
        r = (lo >> s) | (hi << (32u - s));
        rem_lo = lo & ((1u << s) - 1u);
        half_lo = 1u << (s - 1u);
    } else if (s == 32u) {
        r = hi;
        rem_lo = lo;
        half_lo = 0x80000000u;
    } else {
        let t = s - 32u;
        r = hi >> t;
        rem_hi = hi & ((1u << t) - 1u);
        rem_lo = lo;
        half_hi = 1u << (t - 1u);
    }
    let above = rem_hi > half_hi || (rem_hi == half_hi && rem_lo > half_lo);
    let tie = rem_hi == half_hi && rem_lo == half_lo;
    if (above || (tie && (r & 1u) != 0u)) {
        r = r + 1u;
    }
    return r;
}

// `f32(f64(a) / f64(c))` for a count `c >= 1`, the expression the CPU kernel
// evaluates, built from integer operations only.
//
// The quotient's leading 55 bits come from restoring long division, the rest
// collapse into a sticky bit. Rounding to 53 bits gives the f64 quotient, and
// rounding that to the f32 grid, subnormals included, gives the cast.
fn sr_mean_div(a: f32, c: u32) -> f32 {
    let bits = bitcast<u32>(a);
    let sign = bits & 0x80000000u;
    let exp_field = (bits >> 23u) & 0xffu;
    let frac = bits & 0x7fffffu;
    if (exp_field == 0xffu) {
        if (frac != 0u) {
            // NaN keeps its payload with the quiet bit set, as the f64 round
            // trip does.
            return bitcast<f32>(bits | 0x400000u);
        }
        return a;
    }
    if ((bits & 0x7fffffffu) == 0u || c == 1u) {
        return a;
    }

    // a = m * 2^e with m an integer below 2^24.
    var m = frac;
    var e = -149;
    if (exp_field != 0u) {
        m = frac | 0x800000u;
        e = i32(exp_field) - 150;
    }

    // After `steps` steps, q = floor(m * 2^(steps - 24) / c) and `rem` is the
    // remainder. The loop stops once q holds 55 bits. The remainder stays
    // below c, so only the shift can reach 33 bits, and `carry` holds that bit.
    var q_hi = 0u;
    var q_lo = 0u;
    var rem = 0u;
    var steps = 0;
    for (var guard = 0; guard < 160; guard = guard + 1) {
        if (q_hi >= 0x400000u) {
            break;
        }
        var bit = 0u;
        if (steps < 24) {
            bit = (m >> u32(23 - steps)) & 1u;
        }
        let carry = rem >> 31u;
        let shifted = (rem << 1u) | bit;
        var q_bit = 0u;
        if (carry != 0u || shifted >= c) {
            rem = shifted - c;
            q_bit = 1u;
        } else {
            rem = shifted;
        }
        q_hi = (q_hi << 1u) | (q_lo >> 31u);
        q_lo = (q_lo << 1u) | q_bit;
        steps = steps + 1;
    }

    // a / c = (q + rem / c) * 2^scale.
    let scale = e + 24 - steps;

    // Round q to 53 bits: the f64 quotient is (h, l) * 2^(scale + 2).
    let low2 = q_lo & 3u;
    var h = q_hi >> 2u;
    var l = (q_lo >> 2u) | (q_hi << 30u);
    let up = low2 > 2u || (low2 == 2u && (rem != 0u || (l & 1u) != 0u));
    if (up) {
        l = l + 1u;
        if (l == 0u) {
            h = h + 1u;
        }
    }
    let k53 = scale + 2;

    // Round the f64 to the f32 grid: 24 bits for a normal result, a fixed
    // 2^-149 step for a subnormal one.
    var lead = 52;
    if (h >= 0x200000u) {
        lead = 53;
    }
    var drop = lead - 23;
    if (lead + k53 < -126) {
        drop = -149 - k53;
    }
    if (drop >= 54) {
        // Below half the smallest subnormal, or exactly half: rounds to zero.
        return bitcast<f32>(sign);
    }
    var mant = sr_shr_rne(h, l, u32(drop));
    var k = k53 + drop;
    if (mant >= 0x1000000u) {
        mant = mant >> 1u;
        k = k + 1;
    }
    if (mant >= 0x800000u) {
        return bitcast<f32>(sign | (u32(k + 150) << 23u) | (mant - 0x800000u));
    }
    return bitcast<f32>(sign | mant);
}

fn sr_reduce(d: u32, op: u32) {
    var acc = sr_seed(d, op);
    let lo = sr_lower_bound(d);
    let hi = sr_lower_bound(d + 1u);
    for (var j = lo; j < hi; j = j + 1u) {
        let v = sr_src[sr_vals[j]];
        if (op == SR_SUM || op == SR_MEAN) {
            acc = acc + v;
        } else if (op == SR_PROD) {
            acc = acc * v;
        } else if (op == SR_MAX) {
            if (v > acc) {
                acc = v;
            }
        } else if (v < acc) {
            acc = v;
        }
    }
    if (op == SR_MEAN) {
        let count = sr_mean_count(lo, hi);
        if (count > 0u) {
            acc = sr_mean_div(acc, count);
        }
    }
    sr_out[d] = acc;
}

@compute @workgroup_size(256)
fn scatter_reduce_sum_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_SUM);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_prod_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_PROD);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_max_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MAX);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_min_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MIN);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_mean_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MEAN);
    }
}
