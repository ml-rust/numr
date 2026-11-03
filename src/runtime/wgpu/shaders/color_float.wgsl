// Binary32 arithmetic on u32 bit patterns for the colour kernels.
//
// WGSL leaves the rounding direction of `+`, `-` and `*` to the
// implementation, allows 2.5 ULP for `/`, can flush subnormals, can fuse or
// reassociate float operations, and can assume no NaN or infinity at run
// time. Integer operations, by contrast, have one defined result. Every
// function here computes in integers and returns the IEEE-754
// round-to-nearest-even result, subnormals included. A kernel built from them
// evaluates the CPU f32 kernels' operations in the same order and gets the
// same bits. No float type appears on the data path: floats load and store
// as u32 words.
//
// Operands are finite, and no result overflows: every value of the colour
// formulas stays below 2^24 in magnitude. Callers clamp inputs with
// `f_clamp01` first, which maps NaN and infinities to finite values.
//
// A value `(m + f) * 2^e` with integer m and 0 <= f < 1 carries `sticky =
// (f > 0)`. When sticky is set, m has at least 26 significant bits, so the
// round bit and the sticky bit both lie below the kept 24 bits.

const F_SIGN: u32 = 0x80000000u;
const F_ABS: u32 = 0x7fffffffu;
const F_INF: u32 = 0x7f800000u;
const F_ONE: u32 = 0x3f800000u;
const F_255: u32 = 0x437f0000u;
const F_TWO: u32 = 0x40000000u;
const F_QUARTER: u32 = 0x3e800000u;
const F_EIGHTH: u32 = 0x3e000000u;

// m >> r for 1 <= r, rounded to nearest even. `sticky` marks nonzero bits
// below m.
fn f_shr_rne(m: u32, r: u32, sticky: bool) -> u32 {
    if (r >= 32u) {
        let up = r == 32u && (m > F_SIGN || (m == F_SIGN && sticky));
        return select(0u, 1u, up);
    }
    let q = m >> r;
    let rem = m & ((1u << r) - 1u);
    let halfway = 1u << (r - 1u);
    let up = rem > halfway || (rem == halfway && (sticky || (q & 1u) == 1u));
    return select(q, q + 1u, up);
}

// Round sign s and value (m + f) * 2^e to binary32.
fn f_pack(s: u32, m: u32, e: i32, sticky: bool) -> u32 {
    if (m == 0u) {
        return s << 31u;
    }
    let n = 31 - i32(countLeadingZeros(m));
    let be = e + n + 127;
    // A subnormal result keeps the bits at 2^-149 and above.
    var shift = -149 - e;
    var base = 0u;
    if (be >= 1) {
        shift = n - 23;
        base = u32(be - 1) << 23u;
    }
    var q: u32;
    if (shift <= 0) {
        q = m << u32(-shift);
    } else {
        q = f_shr_rne(m, u32(shift), sticky);
    }
    // q includes the implicit bit, so a carry to 2^24 raises the exponent
    // and a subnormal that rounds to 2^23 becomes the smallest normal.
    return (s << 31u) | (base + q);
}

// Integer significand of a: value = f_sig(a) * 2^f_exp(a).
fn f_sig(a: u32) -> u32 {
    let fr = a & 0x7fffffu;
    return select(fr, fr | 0x800000u, (a & F_INF) != 0u);
}

fn f_exp(a: u32) -> i32 {
    return max(i32((a >> 23u) & 0xffu), 1) - 150;
}

fn f_mul(a: u32, b: u32) -> u32 {
    let s = (a ^ b) >> 31u;
    let ma = f_sig(a);
    let mb = f_sig(b);
    if (ma == 0u || mb == 0u) {
        return s << 31u;
    }
    // The 48-bit product as hi * 2^32 + lo, from 16-bit halves.
    let al = ma & 0xffffu;
    let ah = ma >> 16u;
    let bl = mb & 0xffffu;
    let bh = mb >> 16u;
    let ll = al * bl;
    let mid = al * bh + ah * bl;
    let lo = ll + (mid << 16u);
    let carry = select(0u, 1u, lo < ll);
    let hi = ah * bh + (mid >> 16u) + carry;
    let e = f_exp(a) + f_exp(b);
    if (hi == 0u) {
        return f_pack(s, lo, e, false);
    }
    let k = 32u - countLeadingZeros(hi);
    let m = (hi << (32u - k)) | (lo >> k);
    let sticky = (lo & ((1u << k) - 1u)) != 0u;
    return f_pack(s, m, e + i32(k), sticky);
}

fn f_add(a: u32, b: u32) -> u32 {
    // x has the larger magnitude.
    var x = a;
    var y = b;
    if ((b & F_ABS) > (a & F_ABS)) {
        x = b;
        y = a;
    }
    let mx = f_sig(x);
    let my = f_sig(y);
    if (my == 0u) {
        if (mx == 0u) {
            return a & b & F_SIGN;
        }
        return x;
    }
    let ex = f_exp(x);
    let d = u32(ex - f_exp(y));
    // Six guard bits. Bits shifted out of the smaller operand fold into its
    // lowest bit, which stays below the round bit.
    let big = mx << 6u;
    var small = my << 6u;
    if (d >= 32u) {
        small = 1u;
    } else if (d > 0u) {
        let lost = (small & ((1u << d) - 1u)) != 0u;
        small = (small >> d) | select(0u, 1u, lost);
    }
    var m = big + small;
    if (((x ^ y) & F_SIGN) != 0u) {
        m = big - small;
    }
    if (m == 0u) {
        return 0u;
    }
    return f_pack(x >> 31u, m, ex - 6, false);
}

fn f_sub(a: u32, b: u32) -> u32 {
    return f_add(a, b ^ F_SIGN);
}

// a / b for a normal b.
fn f_div(a: u32, b: u32) -> u32 {
    let s = (a ^ b) >> 31u;
    var ma = f_sig(a);
    if (ma == 0u) {
        return s << 31u;
    }
    let za = countLeadingZeros(ma) - 8u;
    ma = ma << za;
    let ea = f_exp(a) - i32(za);
    let mb = f_sig(b);
    let eb = f_exp(b);
    // q = floor(ma * 2^26 / mb), in [2^25, 2^27), by restoring division.
    var r = ma;
    var q = 0u;
    if (r >= mb) {
        q = 1u;
        r = r - mb;
    }
    for (var i = 0u; i < 26u; i = i + 1u) {
        r = r << 1u;
        q = q << 1u;
        if (r >= mb) {
            q = q | 1u;
            r = r - mb;
        }
    }
    return f_pack(s, q, ea - eb - 26, r != 0u);
}

// f32(n), exact for |n| < 2^24.
fn f_from_i32(n: i32) -> u32 {
    return f_pack(select(0u, 1u, n < 0), u32(abs(n)), 0, false);
}

// min(max(x, 0), 1), with NaN and -0 giving +0.
fn f_clamp01(x: u32) -> u32 {
    if ((x & F_ABS) > F_INF || (x & F_SIGN) != 0u) {
        return 0u;
    }
    return min(x, F_ONE);
}

// Key whose unsigned order is the numeric order of finite floats.
fn f_key(x: u32) -> u32 {
    return select(x | F_SIGN, ~x, (x & F_SIGN) != 0u);
}

// round_ties_even(x) for a non-negative x below 2^31.
fn f_round_u(x: u32) -> u32 {
    let ex = i32((x >> 23u) & 0xffu) - 127;
    if (ex < -1) {
        return 0u;
    }
    let m = (x & 0x7fffffu) | 0x800000u;
    let r = 23 - ex;
    if (r <= 0) {
        return m << u32(-r);
    }
    return f_shr_rne(m, u32(r), false);
}

// round_ties_even(clamp(v * mul + off, lo, hi)) for 0 <= lo.
fn f_quantize(v: u32, mul: u32, off: u32, lo: u32, hi: u32) -> u32 {
    var t = f_add(f_mul(v, mul), off);
    if (f_key(t) < f_key(lo)) {
        t = lo;
    }
    if (f_key(t) > f_key(hi)) {
        t = hi;
    }
    return f_round_u(t);
}
