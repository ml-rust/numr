// Correctly rounded binary64 acos. Mirrors src/algorithm/correctly_rounded/acos.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/acos/acos.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2024-2025 Alexei Sibidanov.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_ACOS_CUH
#define NUMR_CORRECTLY_ROUNDED_ACOS_CUH

#include "correctly_rounded_acos_refine.cuh"

#define CR_ACOS_PI_HI 0x400921fb54442d18ULL
#define CR_ACOS_PI_LO 0x3ca1a62633145c07ULL
#define CR_ACOS_HALF_PI_LO 0x3c91a62633145c07ULL
// ax = bits << 1 of |x| = 0.5, 1, 2^-15 and 0x1.cb3b3869747f4p-55.
#define CR_ACOS_AX_HALF 0x7fc0000000000000ULL
#define CR_ACOS_AX_ONE 0x7fe0000000000000ULL
#define CR_ACOS_AX_TINY 0x7e00000000000000ULL
#define CR_ACOS_AX_NO_CUBE 0x791967670d2e8fe8ULL
#define CR_ACOS_TINY_CUBE 0xbfc5555555555555ULL
#define CR_ACOS_TINY_EPS 0x3b03400000000000ULL
#define CR_ACOS_TABLE_EPS 0x3cb8100000000000ULL
#define CR_ACOS_DOUBLE_ANGLE_EPS 0x3cb8c00000000000ULL
#define CR_ACOS_DOUBLE_ANGLE_EPS_ABS 0x3960000000000000ULL
#define CR_ACOS_ROW_STEP 0x3f80000000000000ULL

// |x| <= 2^-15: acos(x) = pi/2 - x - x^3/6, with the cubic term left out where it cannot
// change the result.
__device__ __forceinline__ double cr_acos_tiny(double x, cr_u64 ax) {
    const double v = ax <= CR_ACOS_AX_NO_CUBE
                         ? 0.0
                         : cr_mul(cr_mul(x, x), cr_mul(cr_from_bits(CR_ACOS_TINY_CUBE), x));
    const CrDD s = cr_fast_two_sum(cr_from_bits(CR_ACOS_HALF_PI_HI), -x);
    const double l = cr_add(v, cr_add(s.l, cr_from_bits(CR_ACOS_HALF_PI_LO)));
    const double eps = cr_from_bits(CR_ACOS_TINY_EPS);
    const double lb = cr_add(s.h, cr_sub(l, eps));
    const double ub = cr_add(s.h, cr_add(l, eps));
    if (lb != ub) return cr_acos_refine(x, lb);
    return lb;
}

// Arc cosine in radians, correctly rounded to nearest, ties to even.
__device__ __forceinline__ double cr_acos(double x) {
    const cr_u64 ix = cr_bits(x);
    const cr_u64 ax = ix << 1;
    double z, zl, t, jd, f0h, f0l, eps;
    if (ax > CR_ACOS_AX_HALF) {
        const double oh = (ix >> 63) ? cr_from_bits(CR_ACOS_PI_HI) : 0.0;
        const double ol = (ix >> 63) ? cr_from_bits(CR_ACOS_PI_LO) : 0.0;
        if (ax >= CR_ACOS_AX_ONE) {
            if (ax == CR_ACOS_AX_ONE) return cr_add(oh, ol);
            // NaN, infinity or |x| > 1.
            return cr_nan();
        }
        // acos(x) = 2 asin(sqrt((1 - x)/2)) for x > 0.5, and
        // acos(x) = pi - 2 asin(sqrt((1 + x)/2)) for x < -0.5.
        const double w = cr_sub(2.0, cr_mul(2.0, cr_abs(x)));
        jd = cr_round_ties_even(cr_mul(w, 32.0));
        z = cr_copysign(cr_sqrt(w), x);
        zl = cr_mul(cr_fma(z, z, -w), cr_mul(cr_div(-0.5, w), z));
        t = cr_sub(cr_mul(0.25, w), cr_mul(jd, cr_from_bits(CR_ACOS_ROW_STEP)));
        eps = cr_add(cr_mul(cr_abs(cr_mul(z, t)), cr_from_bits(CR_ACOS_DOUBLE_ANGLE_EPS)),
                     cr_from_bits(CR_ACOS_DOUBLE_ANGLE_EPS_ABS));
        f0h = oh;
        f0l = ol;
    } else {
        if (ax <= CR_ACOS_AX_TINY) return cr_acos_tiny(x, ax);
        // acos(x) = pi/2 - asin(x), on the same asin tables as the double-angle branch.
        const double w = cr_mul(x, x);
        jd = cr_round_ties_even(cr_mul(w, 128.0));
        t = cr_fma(x, x, cr_mul(-cr_from_bits(CR_ACOS_ROW_STEP), jd));
        z = -x;
        zl = 0.0;
        // eps is negative for x > 0, and the rounding test still holds.
        eps = cr_mul(cr_mul(z, t), cr_from_bits(CR_ACOS_TABLE_EPS));
        f0h = cr_from_bits(CR_ACOS_HALF_PI_HI);
        f0l = cr_from_bits(CR_ACOS_HALF_PI_LO);
    }
    // asin(z + zl) = (z + zl) * (c0 + (c1 + t * P(t))) with t = z^2 - j/128.
    const unsigned int row = 8 * (unsigned int)cr_d2i(jd);
    const double t2 = cr_mul(t, t);
    const double c2 = cr_tab(CR_ACOS_CC, row + 2);
    const double c3 = cr_tab(CR_ACOS_CC, row + 3);
    const double c4 = cr_tab(CR_ACOS_CC, row + 4);
    const double c5 = cr_tab(CR_ACOS_CC, row + 5);
    const double c6 = cr_tab(CR_ACOS_CC, row + 6);
    const double c7 = cr_tab(CR_ACOS_CC, row + 7);
    const double d = cr_mul(
        t, cr_add(cr_add(c2, cr_mul(t, c3)),
                  cr_mul(t2, cr_add(cr_add(c4, cr_mul(t, c5)), cr_mul(t2, cr_add(c6, cr_mul(t, c7)))))));
    CrDD f = cr_mul_dd(z, zl, cr_tab(CR_ACOS_CC, row), cr_add(cr_tab(CR_ACOS_CC, row + 1), d));
    f = cr_fast_sum(f0h, f0l, f.h, f.l);
    const double lb = cr_add(f.h, cr_sub(f.l, eps));
    const double ub = cr_add(f.h, cr_add(f.l, eps));
    if (lb != ub) return cr_acos_refine(x, lb);
    return lb;
}

#endif
