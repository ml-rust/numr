// Fast path of pow: log|x| and exp in double-double arithmetic. Mirrors
// src/algorithm/correctly_rounded/pow_fast.rs. Lemma numbers refer to Hubrecht,
// Jeannerod and Zimmermann, ARITH 2023, <https://inria.hal.science/hal-04159652>.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/pow.c. Copyright (c) 2022-2025 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_POW_FAST_CUH
#define NUMR_CORRECTLY_ROUNDED_POW_FAST_CUH

#include "correctly_rounded_ops.cuh"

#define CR_POW_LOG2_H 0x3fe62e42fefa3800ULL
#define CR_POW_LOG2_L 0x3d2ef35793c76730ULL
#define CR_POW_SQRT2_SIGNIFICAND 0x16a09e667f3bcdULL
#define CR_POW_RHO0 0xc0874910ee4e8a27ULL
#define CR_POW_RHO1 0xc08483b8cca421afULL
#define CR_POW_RHO2 0x40862e42e709a95bULL
#define CR_POW_RHO3 0x40862e4316ea5df9ULL
#define CR_POW_INVLOG2 0x40b71547652b82feULL
#define CR_POW_LOG2H 0x3f262e42fefa39efULL
#define CR_POW_LOG2L 0x3bbabc9e3b39803fULL
#define CR_POW_FAST_ERR 0x3bf2700000000000ULL
#define CR_POW_FAST_ERR_CANCEL 0x3c55700000000000ULL

struct CrPowLog {
    double h;
    double l;
    bool cancel;
};

// a + (bh + bl). Needs |a| >= |bh|.
__device__ __forceinline__ CrDD cr_pow_fast_sum(double a, double bh, double bl) {
    const CrDD s = cr_fast_two_sum(a, bh);
    return cr_dd(s.h, cr_add(s.l, bl));
}

// a * (bh + bl): the high product is exact, a * bl is fused into the low part.
__device__ __forceinline__ CrDD cr_pow_s_mul(double a, double bh, double bl) {
    const CrDD p = cr_two_prod(a, bh);
    return cr_dd(p.h, cr_fma(a, bl, p.l));
}

// (ah + al) * (bh + bl) - al * bl.
__device__ __forceinline__ CrDD cr_pow_d_mul(double ah, double al, double bh, double bl) {
    const CrDD p = cr_two_prod(ah, bh);
    const double t = cr_fma(al, bh, p.l);
    return cr_dd(p.h, cr_fma(ah, bl, t));
}

// log(1 + z) - z for |z| <= 33 * 2^-13 (Lemma 2).
__device__ __forceinline__ CrDD cr_pow_p_1(double z) {
    const cr_u64* c = CR_POW_FAST_LOG_POLY;
    const CrDD w = cr_two_prod(z, z);
    const double t = cr_fma(cr_tab(c, 5), z, cr_tab(c, 4));
    double u = cr_fma(cr_tab(c, 3), z, cr_tab(c, 2));
    double v = cr_fma(cr_tab(c, 1), z, cr_tab(c, 0));
    u = cr_fma(t, w.h, u);
    v = cr_fma(u, w.h, v);
    u = cr_mul(v, w.h);
    return cr_dd(cr_mul(-0.5, w.h), cr_fma(u, z, cr_mul(-0.5, w.l)));
}

// log(x) as (h, l, cancel) for 2^-1074 <= x < 2^1024 (Lemma 4).
__device__ __forceinline__ CrPowLog cr_pow_log_1(double x) {
    const cr_u64 bits = cr_bits(x);
    cr_u64 m = bits & CR_FRACTION_MASK;
    cr_i64 e = (cr_i64)((bits >> 52) & 0x7ff);
    double t;
    if (e != 0) {
        t = cr_from_bits(m | (0x3ffULL << 52));
        m += 1ULL << 52;
        e -= 0x3ff;
    } else {
        // Subnormal x: normalize the significand.
        const int k = cr_clz(m) - 11;
        e = -0x3fe - k;
        m <<= k;
        t = cr_from_bits(m | (0x3ffULL << 52));
    }
    // Now x = 2^e * t with 1 <= t < 2, and 2^52 <= m < 2^53.
    const bool c = m >= CR_POW_SQRT2_SIGNIFICAND;
    e += c ? 1 : 0;
    unsigned int i;
    if (c) {
        i = (unsigned int)(m >> 45);
        t = cr_mul(t, 0.5);
    } else {
        i = (unsigned int)(m >> 44);
    }
    // Now 1/sqrt(2) <= t < sqrt(2) and log(x) = e * log(2) + log(t).
    const double big_e = cr_i2d(e);
    const double r = cr_tab(CR_POW_INVERSE, i - 181);
    const double l1 = cr_tab(CR_POW_LOG_INV, 2 * (i - 181));
    const double l2 = cr_tab(CR_POW_LOG_INV, 2 * (i - 181) + 1);
    const double z = cr_fma(r, t, -1.0);
    const double th = cr_fma(big_e, cr_from_bits(CR_POW_LOG2_H), l1);
    const double tl = cr_fma(big_e, cr_from_bits(CR_POW_LOG2_L), l2);
    CrDD s = cr_pow_fast_sum(th, z, tl);
    const CrDD p = cr_pow_p_1(z);
    s = cr_pow_fast_sum(s.h, p.h, cr_add(s.l, p.l));
    CrPowLog out;
    out.cancel = false;
    if (e == 0 && cr_abs(s.l) > cr_mul(cr_abs(s.h), cr_from_bits(0x3e70000000000000ULL))) {
        s = cr_fast_two_sum(s.h, s.l);
        out.cancel = true;
    }
    out.h = s.h;
    out.l = s.l;
    return out;
}

// exp(z) for |z| < 2^-12.905 (Lemma 6).
__device__ __forceinline__ CrDD cr_pow_q_1(double z) {
    const cr_u64* c = CR_POW_FAST_EXP_POLY;
    double q = cr_fma(cr_tab(c, 4), z, cr_tab(c, 3));
    q = cr_fma(q, z, cr_tab(c, 2));
    const double h0 = cr_fma(q, z, cr_tab(c, 1));
    const CrDD p = cr_two_prod(z, h0);
    return cr_pow_fast_sum(cr_tab(c, 0), p.h, p.l);
}

// s * exp(rh + rl) (Lemma 7). A NaN pair where the result can overflow or be subnormal:
// the rounding test then fails and the accurate path decides.
__device__ __forceinline__ CrDD cr_pow_exp_1(double rh, double rl, double s) {
    const bool nan = cr_is_nan(rh);
    if (nan || rh > cr_from_bits(CR_POW_RHO2)) {
        if (!nan && rh > cr_from_bits(CR_POW_RHO3)) {
            const double big = cr_mul(cr_dbl_max(), s);
            return cr_dd(big, cr_mul(cr_dbl_max(), s));
        }
        return cr_dd(cr_nan(), cr_nan());
    }
    if (rh < cr_from_bits(CR_POW_RHO1)) {
        if (rh < cr_from_bits(CR_POW_RHO0)) {
            // Both parts round to a zero of the result's sign.
            return cr_dd(cr_mul(0.0, s), cr_mul(cr_from_bits(1), cr_mul(0.5, s)));
        }
        return cr_dd(cr_nan(), cr_nan());
    }
    const double k = cr_round_ties_even(cr_mul(rh, cr_from_bits(CR_POW_INVLOG2)));
    const double zh = cr_fma(cr_from_bits(CR_POW_LOG2H), -k, rh);
    const double zl = cr_fma(cr_from_bits(CR_POW_LOG2L), -k, rl);
    const cr_i64 ki = cr_d2i(k);
    const cr_u64 scale = (cr_u64)((ki >> 12) + 0x3ff);
    const unsigned int i2 = (unsigned int)((ki >> 6) & 0x3f);
    const unsigned int i1 = (unsigned int)(ki & 0x3f);
    CrDD e = cr_pow_d_mul(cr_tab(CR_POW_T2, 2 * i1), cr_tab(CR_POW_T2, 2 * i1 + 1), cr_tab(CR_POW_T1, 2 * i2),
                          cr_tab(CR_POW_T1, 2 * i2 + 1));
    const CrDD q = cr_pow_q_1(cr_add(zh, zl));
    e = cr_pow_d_mul(e.h, e.l, q.h, q.l);
    // 1 < scale < 2047: the range checks above exclude overflow and underflow.
    const double d = cr_mul(cr_from_bits(scale << 52), s);
    return cr_dd(cr_mul(e.h, d), cr_mul(e.l, d));
}

#endif
