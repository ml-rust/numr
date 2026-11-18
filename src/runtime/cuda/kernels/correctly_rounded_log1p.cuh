// Correctly rounded binary64 log1p. Mirrors src/algorithm/correctly_rounded/log1p.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log1p/log1p.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2024-2026 Alexei Sibidanov <sibid@uvic.ca>.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_LOG1P_CUH
#define NUMR_CORRECTLY_ROUNDED_LOG1P_CUH

#include "correctly_rounded_log1p_refine.cuh"

// ax = bits << 1 thresholds: |x| < 2^-4, 2^-12 and 0x1.19bp-21.
#define CR_LOG1P_AX_SMALL 0x7f60000000000000ULL
#define CR_LOG1P_AX_SERIES 0x7e60000000000000ULL
#define CR_LOG1P_AX_SHORT_SERIES 0x7d43360000000000ULL
#define CR_LOG1P_EPS_SERIES 0x3bfa000000000000ULL
#define CR_LOG1P_EPS_MEDIUM 0x3cbb600000000000ULL
#define CR_LOG1P_EPS_LARGE 0x3bea000000000000ULL
#define CR_LOG1P_LN2_HI 0x3fe62e42fefa4000ULL
#define CR_LOG1P_LN2_LO 0xbd48432a1b0e2634ULL
#define CR_LOG1P_MINUS_ONE_BITS 0xbff0000000000000ULL

// NaN, +inf, -1, -inf and x < -1.
__device__ __forceinline__ double cr_log1p_special(cr_u64 ix, cr_u64 ax) {
    if (ax > 0xffe0000000000000ULL) return cr_nan();
    if (ix == CR_INF_BITS) return cr_inf();
    if (ix == CR_LOG1P_MINUS_ONE_BITS) return cr_neg_inf();
    return cr_nan();
}

// log(t + dt) as a double-double for |x| >= 2^-4.
__device__ __forceinline__ CrDD cr_log1p_large(double t, double dt) {
    const cr_i64 j = (cr_i64)(cr_bits(t) - 0x3fe6a00000000000ULL);
    const unsigned int j1 = (unsigned int)((j >> (52 - 6)) & 0x3f);
    const cr_i64 je = j >> 52;
    const cr_u64 eoff = (cr_u64)je << 52;
    cr_u64 rs = cr_word(CR_LOG1P_RF, j1);
    if (je < 1022) {
        rs -= eoff;
    } else {
        // Scale t and dt instead of rs, so no subnormal reciprocal appears.
        rs -= 1021ULL << 52;
        const double sc = je == 1022 ? 0.5 : (je == 1023 ? 0.25 : 0.125);
        t = cr_mul(t, sc);
        dt = cr_mul(dt, sc);
    }
    const double r = cr_from_bits(rs);
    const double dh = cr_mul(r, t);
    const double dl = cr_add(cr_fma(r, t, -dh), cr_mul(r, dt));
    const CrDD xv = cr_fast_two_sum(cr_sub(dh, 1.0), dl);
    const double xh = xv.h;
    const double x2 = cr_mul(xh, xh);
    const cr_u64* c = CR_LOG1P_LARGE_POLY;
    const double poly = cr_mul(
        x2, cr_add(cr_add(cr_tab(c, 0), cr_mul(xh, cr_tab(c, 1))),
                   cr_mul(x2, cr_add(cr_add(cr_tab(c, 2), cr_mul(xh, cr_tab(c, 3))),
                                     cr_mul(x2, cr_add(cr_tab(c, 4), cr_mul(xh, cr_tab(c, 5))))))));
    const double xl = cr_add(xv.l, poly);
    const double jef = cr_i2d(je);
    const double l1 = cr_mul(cr_from_bits(CR_LOG1P_LN2_HI), jef);
    const double l0 = cr_mul(cr_from_bits(CR_LOG1P_LN2_LO), jef);
    const double ln1 = cr_add(cr_tab(CR_LOG1P_LF, 2 * j1 + 1), l1);
    const double ln0 = cr_add(cr_tab(CR_LOG1P_LF, 2 * j1), l0);
    return cr_fast_sum(ln1, ln0, xh, xl);
}

// ln(1 + x), correctly rounded to nearest, ties to even, without forming 1 + x.
__device__ __forceinline__ double cr_log1p(double x) {
    const cr_u64 ix = cr_bits(x);
    const cr_u64 ax = ix << 1;
    double ln1, ln0, eps;
    if (ax < CR_LOG1P_AX_SMALL) {
        // Check tiny x first to avoid a spurious underflow in x * x.
        if (ax < CR_LOG1P_AX_TINY) return cr_log1p_tiny(x, ax);
        const double x2 = cr_mul(x, x);
        if (ax < CR_LOG1P_AX_SERIES) {
            ln1 = x;
            eps = cr_mul(cr_from_bits(CR_LOG1P_EPS_SERIES), x);
            if (ax < CR_LOG1P_AX_SHORT_SERIES) {
                const cr_u64* c = CR_LOG1P_TINY_POLY;
                ln0 = cr_mul(x2, cr_add(cr_tab(c, 0), cr_mul(x, cr_tab(c, 1))));
            } else {
                const cr_u64* c = CR_LOG1P_SMALL_POLY;
                ln0 = cr_mul(x2, cr_add(cr_add(cr_tab(c, 0), cr_mul(x, cr_tab(c, 1))),
                                        cr_mul(x2, cr_add(cr_tab(c, 2), cr_mul(x, cr_tab(c, 3))))));
            }
        } else {
            const cr_u64* c = CR_LOG1P_MEDIUM_POLY;
            const double x3 = cr_mul(x2, x);
            const double x4 = cr_mul(x2, x2);
            const double hx = cr_mul(-0.5, x);
            ln1 = cr_fma(hx, x, x);
            const double low = cr_fma(hx, x, cr_sub(x, ln1));
#define CR_PAIR(k) cr_add(cr_tab(c, k), cr_mul(x, cr_tab(c, (k) + 1)))
            const double series = cr_add(
                cr_add(CR_PAIR(0), cr_mul(x2, CR_PAIR(2))),
                cr_mul(x4, cr_add(cr_add(CR_PAIR(4), cr_mul(x2, CR_PAIR(6))),
                                  cr_mul(x4, cr_add(CR_PAIR(8), cr_mul(x2, CR_PAIR(10)))))));
#undef CR_PAIR
            ln0 = cr_add(low, cr_mul(x3, series));
            eps = cr_mul(x3, cr_from_bits(CR_LOG1P_EPS_MEDIUM));
        }
    } else {
        double t, dt;
        if ((cr_i64)ix < 0x4340000000000000LL && ix < CR_LOG1P_MINUS_ONE_BITS) {
            // -1 < x <= -2^-4 or 2^-4 <= x < 2^53: the fast two-sum stays exact.
            const CrDD s = cr_fast_two_sum(1.0, x);
            t = s.h;
            dt = s.l;
        } else if (ix < 0x4690000000000000ULL) {
            t = x;
            dt = 1.0;
        } else if (ix < CR_INF_BITS) {
            t = x;
            dt = 0.0;
        } else {
            return cr_log1p_special(ix, ax);
        }
        const CrDD r = cr_log1p_large(t, dt);
        ln1 = r.h;
        ln0 = r.l;
        eps = cr_from_bits(CR_LOG1P_EPS_LARGE);
    }
    const double lb = cr_add(ln1, cr_sub(ln0, eps));
    const double ub = cr_add(ln1, cr_add(ln0, eps));
    if (lb != ub) return cr_log1p_refine(x, lb);
    return lb;
}

#endif
