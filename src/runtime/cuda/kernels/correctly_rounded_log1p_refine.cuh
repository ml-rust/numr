// Accurate path of the correctly rounded binary64 log1p, in triple-double arithmetic.
// Mirrors src/algorithm/correctly_rounded/log1p_refine.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log1p/log1p.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2024-2026 Alexei Sibidanov <sibid@uvic.ca>.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_LOG1P_REFINE_CUH
#define NUMR_CORRECTLY_ROUNDED_LOG1P_REFINE_CUH

#include "correctly_rounded_ops.cuh"

#define CR_LOG1P_AX_TINY 0x7940000000000000ULL
#define CR_LOG1P_AX_REFINE_SERIES 0x7ea0000000000000ULL
#define CR_LOG1P_MINUS_TWO_POW_54 0xbc90000000000000ULL
#define CR_LOG1P_REFINE_SCALE 0x40f71547652b82feULL
#define CR_LOG1P_INDEX_OFFSET 34952
#define CR_LOG1P_LN2_2 0x3fe62e42fefa4000ULL
#define CR_LOG1P_LN2_1 0xbd48432a1b0e2800ULL
#define CR_LOG1P_LN2_0 0x3a8cc01f97b57a08ULL

// log1p(x) for 0 < |x| < 2^-53: one fused operation rounds once.
__device__ __forceinline__ double cr_log1p_tiny(double x, cr_u64 ax) {
    if (ax == 0) return x;
    return cr_fma(cr_abs(x), cr_from_bits(CR_LOG1P_MINUS_TWO_POW_54), x);
}

// Horner evaluation in double-double of c at the double x.
__device__ __forceinline__ CrDD cr_log1p_poly_d(double x, const cr_u64* c, int n, double low) {
    const CrDD s = cr_fast_two_sum(cr_tab(c, 2 * (n - 1)), low);
    CrDD r = cr_dd(s.h, cr_add(cr_tab(c, 2 * (n - 1) + 1), s.l));
    for (int i = n - 2; i >= 0; --i) {
        r = cr_mul_d_dd(x, r.h, r.l);
        r = cr_fast_sum(cr_tab(c, 2 * i), cr_tab(c, 2 * i + 1), r.h, r.l);
    }
    return r;
}

__device__ __forceinline__ double cr_log1p_ln(unsigned int k, unsigned int j, unsigned int m) {
    return cr_tab(CR_LOG1P_LN, 3 * (16 * k + j) + m);
}

// Accurate log1p(x); a is the fast-path lower bound.
__device__ __noinline__ double cr_log1p_refine(double x, double a) {
    const cr_u64 ix = cr_bits(x);
    const cr_u64 ax = ix << 1;
    double ln22, ln21, ln20;
    if (ax < CR_LOG1P_AX_REFINE_SERIES) {
        if (ax < CR_LOG1P_AX_TINY) return cr_log1p_tiny(x, ax);
        double x2h = cr_mul(x, x);
        double x2l = cr_fma(x, x, -x2h);
        const CrDD x3 = cr_mul_d_dd(x, x2h, x2l);
        const double sl0 = cr_mul(
            x, cr_add(cr_add(cr_tab(CR_LOG1P_REFINE_CZL, 0), cr_mul(x, cr_tab(CR_LOG1P_REFINE_CZL, 1))),
                      cr_mul(x2h, cr_add(cr_tab(CR_LOG1P_REFINE_CZL, 2), cr_mul(x, cr_tab(CR_LOG1P_REFINE_CZL, 3))))));
        CrDD s = cr_log1p_poly_d(x, CR_LOG1P_REFINE_CZ, (int)CR_LEN(CR_LOG1P_REFINE_CZ) / 2, sl0);
        s = cr_mul_dd(s.h, s.l, x3.h, x3.l);
        x2h = cr_mul(x2h, -0.5);
        x2l = cr_mul(x2l, -0.5);
        ln22 = x;
        const CrDD t = cr_fast_sum(x2h, x2l, s.h, s.l);
        ln21 = t.h;
        ln20 = t.l;
    } else {
        const cr_i64 j = cr_d2i(cr_round_ties_even(cr_mul(a, cr_from_bits(CR_LOG1P_REFINE_SCALE))));
        const cr_i64 i = j + CR_LOG1P_INDEX_OFFSET;
        const unsigned int j1 = (unsigned int)((i >> 12) & 0xf);
        const unsigned int j2 = (unsigned int)((i >> 8) & 0xf);
        const unsigned int j3 = (unsigned int)((i >> 4) & 0xf);
        const unsigned int j4 = (unsigned int)(i & 0xf);
        const cr_i64 je = i >> 16;
        double l[3];
        for (unsigned int m = 0; m < 3; ++m) {
            l[m] = cr_add(cr_add(cr_log1p_ln(0, j1, m), cr_log1p_ln(1, j2, m)),
                          cr_add(cr_log1p_ln(2, j3, m), cr_log1p_ln(3, j4, m)));
        }

        double t, dt;
        if ((cr_i64)ix < 0x4690000000000000LL && ix < 0xbfe0000000000000ULL) {
            const CrDD s = cr_two_sum(1.0, x);
            t = s.h;
            dt = (cr_bits(s.l) << 1) == 0 ? 0.0 : s.l;
        } else if ((cr_i64)ix >= 0x4690000000000000LL) {
            t = x;
            dt = 0.0;
        } else {
            t = cr_add(1.0, x);
            dt = 0.0;
        }
        t = cr_from_bits(cr_bits(t) - ((cr_u64)je << 52));

        const double t12 = cr_mul(cr_tab(CR_LOG1P_RT, j1), cr_tab(CR_LOG1P_RT, 16 + j2));
        const double t34 = cr_mul(cr_tab(CR_LOG1P_RT, 32 + j3), cr_tab(CR_LOG1P_RT, 48 + j4));
        const double th = cr_mul(t12, t34);
        const double tl = cr_fma(t12, t34, -th);
        const double dh = cr_mul(th, t);
        const double dl = cr_fma(th, t, -dh);
        const double sh = cr_mul(tl, t);
        const double sl = cr_fma(tl, t, -sh);
        CrDD xv = cr_fast_two_sum(cr_sub(dh, 1.0), dl);
        xv = cr_fast_sum(xv.h, xv.l, sh, sl);
        if (cr_bits(dt) != 0) {
            dt = cr_from_bits(cr_bits(dt) - ((cr_u64)je << 52));
            const double ddh = cr_mul(th, dt);
            const double ddl = cr_add(cr_fma(th, dt, -ddh), cr_mul(tl, dt));
            xv = cr_fast_sum(xv.h, xv.l, ddh, ddl);
        }
        const double xh = xv.h;
        const double xl = xv.l;
        const double sl0 = cr_mul(
            xh, cr_add(cr_add(cr_tab(CR_LOG1P_REFINE_CL, 0), cr_mul(xh, cr_tab(CR_LOG1P_REFINE_CL, 1))),
                       cr_mul(cr_mul(xh, xh),
                              cr_add(cr_tab(CR_LOG1P_REFINE_CL, 2), cr_mul(xh, cr_tab(CR_LOG1P_REFINE_CL, 3))))));
        CrDD s = cr_horner_dd(xh, xl, CR_LOG1P_REFINE_CY, (int)CR_LEN(CR_LOG1P_REFINE_CY) / 2, sl0);
        s = cr_mul_dd(xh, xl, s.h, s.l);
        if (je != 0) {
            const double jef = cr_i2d(je);
            ln22 = cr_mul(cr_from_bits(CR_LOG1P_LN2_2), jef);
            ln21 = cr_mul(cr_from_bits(CR_LOG1P_LN2_1), jef);
            ln20 = cr_mul(cr_from_bits(CR_LOG1P_LN2_0), jef);
            ln22 = cr_add(ln22, l[2]);
            const CrDD u = cr_fast_sum(ln21, ln20, l[1], l[0]);
            ln21 = u.h;
            ln20 = u.l;
        } else {
            ln22 = l[2];
            ln21 = l[1];
            ln20 = l[0];
        }
        const CrDD u = cr_sum(ln21, ln20, s.h, s.l);
        ln21 = u.h;
        ln20 = u.l;
    }
    CrDD u = cr_fast_two_sum(ln22, ln21);
    ln22 = u.h;
    ln21 = u.l;
    u = cr_fast_two_sum(ln21, ln20);
    ln21 = u.h;
    ln20 = u.l;
    // A power-of-two middle term moves one ulp toward the low term, so the final add sees
    // the sticky information of ln20.
    cr_u64 t = cr_bits(ln21);
    if ((t & (~0ULL >> 12)) == 0) {
        const cr_u64 w = cr_bits(ln20);
        t = ((w ^ t) >> 63) != 0 ? t - 1 : t + 1;
        ln21 = cr_from_bits(t);
    }
    return cr_add(ln22, ln21);
}

#endif
