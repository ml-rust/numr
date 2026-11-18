// Exact and midpoint cases of pow, and the integer tests on binary64 values. Mirrors
// src/algorithm/correctly_rounded/pow_exact.rs: Algorithm detectRoundingBoundaryCase of
// Lauter and Lefevre, IEEE Transactions on Computers 58(2), 2009.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/pow.c. Copyright (c) 2022-2025 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_POW_EXACT_CUH
#define NUMR_CORRECTLY_ROUNDED_POW_EXACT_CUH

#include "correctly_rounded_pow_dint.cuh"

// xmax[y] for 1 <= y <= 33: the largest odd m with m^y below 2^53. Same as XMAX in
// pow_exact.rs.
__device__ const cr_u64 CR_POW_XMAX[34] = {
    0,  0xffffffffffffffffULL, 94906265, 208063, 9741, 1551, 455, 189, 97, 59, 39, 27, 21, 15, 13, 11, 9,
    7,  7, 5, 5, 5, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
};

// Whether the finite x is an integer, read from its bits.
__device__ __forceinline__ bool cr_is_int(double x) {
    const cr_u64 bits = cr_bits(x) & CR_ABS_MASK;
    const cr_u64 biased = bits >> 52;
    if (biased < 0x3ff) return bits == 0;
    if (biased >= 0x433) return true;
    return (bits & ((1ULL << (0x433 - biased)) - 1)) == 0;
}

// Whether the finite y is an odd integer.
__device__ __forceinline__ bool cr_is_odd_int(double y) { return cr_is_int(y) && !cr_is_int(cr_mul(y, 0.5)); }

// Floor of the square root of n.
__device__ __forceinline__ cr_u64 cr_isqrt(cr_u64 n) {
    cr_u64 r = 0;
    cr_u64 bit = 1ULL << 62;
    while (bit > n) bit >>= 2;
    while (bit != 0) {
        if (n >= r + bit) {
            n -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}

// |x| = 2^e * m with m odd. Needs a finite nonzero x.
__device__ __forceinline__ void cr_pow_extract(double x, cr_i64* e, cr_u64* m) {
    const cr_u64 bits = cr_bits(x);
    const cr_i64 biased = (cr_i64)((bits >> 52) & 0x7ff);
    const cr_u64 sig = (bits & CR_FRACTION_MASK) + (biased != 0 ? (1ULL << 52) : 0);
    const int t = cr_ctz(sig);
    *e = biased + t - (0x433 - (biased == 0 ? 1 : 0));
    *m = sig >> t;
}

// |x| = m * 2^e with m odd, from the integer layout of cr_pow_is_exact.
__device__ __forceinline__ void cr_pow_odd_split(cr_u64 bits, cr_u64* m_out, cr_i64* e_out) {
    cr_u64 m = bits & CR_FRACTION_MASK;
    cr_i64 e = (cr_i64)((bits << 1) >> 53) - 0x433;
    if (e >= -1074) {
        m |= 1ULL << 52;
    } else {
        e += 1;
    }
    const int t = cr_ctz(m);
    *m_out = t < 64 ? m >> t : 0;
    *e_out = e + t;
}

// x * 2^e, rounded once. Needs -2044 <= e <= 2047.
__device__ __forceinline__ double cr_pow2(double x, cr_i64 e) {
    if ((e & 1) != 0) x = cr_mul(x, 2.0);
    const double e2 = cr_from_bits((((cr_u64)((e >> 1) + 0x3ff)) & 0x7ff) << 52);
    return cr_mul(cr_mul(x, e2), e2);
}

// Whether x^y is exactly representable, for finite x > 0 and finite y.
__device__ __noinline__ bool cr_pow_is_exact(double x, double y) {
    const cr_u64 v = cr_bits(x);
    const cr_u64 w = cr_bits(y);
    // Every exact case has |x| = 1 or the low 42 bits of y zero.
    if ((v << 1) != 0x7fe0000000000000ULL && (w << 22) != 0) return false;
    if ((v << 1) == 0x7fe0000000000000ULL) return true;
    cr_u64 m;
    cr_i64 e;
    cr_pow_odd_split(v, &m, &e);
    if (y >= 0.0 && cr_is_int(y)) {
        if (y == 0.0 || y == 1.0) return true;
        if (m == 1) {
            const double ye = cr_mul(y, cr_i2d(e));
            return ye >= -1074.0 && ye < 1024.0;
        }
        if (!(y >= 0.0 && y <= 33.0)) return false;
        const cr_i64 yi = cr_d2i(y);
        if (m > CR_POW_XMAX[yi]) return false;
        const cr_i64 t = 64 - cr_clz(m);
        const cr_i64 ez = e * yi + t;
        if (ez <= -1074 || 1024 < ez) return false;
        return e * yi >= -1074;
    }
    cr_u64 n;
    cr_i64 f;
    cr_pow_odd_split(w, &n, &f);
    if (y < 0.0) {
        if (m != 1) return false;
        // e != 0 here, since |x| = 1 returned above.
        cr_i64 ez;
        if (f >= 0) {
            ez = f < 12 ? (cr_i64)(((cr_u64)0 - n) * (cr_u64)e << f) : 1024;
        } else {
            if (-f > cr_ctz((cr_u64)e)) return false;
            ez = ((-e) >> (-f)) * (cr_i64)n;
        }
        return ez >= -1074 && ez < 1024;
    }
    // y > 0 is not an integer: y = n * 2^f with n odd and f < 0, so x must be a
    // (2^-f)-th power. Each step takes one exact square root of m * 2^e.
    while (f != 0) {
        f += 1;
        if ((e & 1) != 0) return false;
        e /= 2;
        const cr_u64 s = cr_isqrt(m);
        if (s * s != m) return false;
        m = s;
    }
    if (m > 1 && (33 < n || m > CR_POW_XMAX[n])) return false;
    // n <= 2^11 here, as y has at most 11 significant bits.
    cr_u64 my = m;
    for (cr_u64 i = 1; i < n; ++i) my *= m;
    const cr_i64 t = 64 - cr_clz(my);
    const cr_i64 en = e * (cr_i64)n;
    return -1074 <= en && en + t <= 1024;
}

// The value of x^y if it is exact or a midpoint, from the accurate approximation z.
// Returns false when it is neither. `exact` is cr_pow_is_exact of (|x|, y).
__device__ __noinline__ bool cr_pow_exact_value(double x, double y, CrDint z, bool exact, double* out) {
    const cr_i64 sign = z.sgn != 0 ? -1 : 1;
    cr_i64 e;
    cr_u64 m;
    cr_pow_extract(x, &e, &m);
    if (m == 1) {
        // x is a power of two.
        const double g = cr_mul(cr_i2d(e), y);
        if (!cr_is_int(g)) return false;
        // Rust `g as i64` saturates; every |g| > 1100 gives inf or 0 after the clamp.
        const cr_i64 gi = g >= 1100.0 ? 1100 : (g <= -1100.0 ? -1100 : cr_d2i(g));
        *out = cr_pow2(cr_i2d(sign), gi);
        return true;
    }
    if (!(y >= 0.0 && y <= 34.0)) return false;
    cr_i64 f;
    cr_u64 n;
    cr_pow_extract(y, &f, &n);
    if (n > 34 || f < -5) return false;
    if (f < 0) {
        // y = n * 2^f with -5 <= f < 0: e must be a multiple of 2^-f.
        if (((cr_u64)e & (~0ULL >> (64 + f))) != 0) return false;
        const cr_i64 g = (e >> (-f)) * (cr_i64)n;
        // z rounded to 54 bits is k * 2^big_g.
        const cr_i64 big_g = z.ex - 53;
        const cr_u64 k = (z.hi >> 10) + ((z.hi >> 9) & 1);
        if (!exact) {
            // Reject unless |k * 2^big_g - z| < 2^-116 * |z|.
            const int cnt = cr_clz(k);
            const CrDint rounded = cr_dint(k << cnt, 0, big_g + 63 - cnt, 1 - z.sgn);
            CrDint d = cr_pd_add(z, rounded);
            d.ex += 116;
            if (cr_pd_cmp_abs(d, z) >= 0) return false;
        }
        if (big_g > g) return false;
        // k must be an odd number times 2^(g - big_g).
        const cr_i64 shift = g - big_g;
        if (shift >= 64) return false;
        if ((k & ~(~1ULL << shift)) == (1ULL << shift)) {
            *out = cr_pow2(cr_i2d((cr_i64)(k >> shift) * sign), g);
            return true;
        }
        return false;
    }
    // y is an integer with 2 <= y <= 34: compute m^y exactly if it fits in 54 bits.
    cr_u64 t = n << f;
    cr_i64 k = 1;
    cr_u64 base = m;
    while (t != 0) {
        if ((t & 1) != 0) {
            // base * k must fit in i64; k > 0 throughout.
            if (__umul64hi(base, (cr_u64)k) != 0) return false;
            const cr_u64 product = base * (cr_u64)k;
            if (product > 0x7fffffffffffffffULL) return false;
            k = (cr_i64)product;
        }
        t >>= 1;
        if (t != 0) {
            if (__umul64hi(base, base) != 0) return false;
            base *= base;
        }
    }
    if ((k >> 54) != 0) return false;
    *out = cr_pow2(cr_i2d(k * sign), e * (cr_i64)(n << f));
    return true;
}

#endif
