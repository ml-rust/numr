// Correctly rounded binary64 exp. Mirrors src/algorithm/correctly_rounded/exp.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/exp/exp.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_EXP_CUH
#define NUMR_CORRECTLY_ROUNDED_EXP_CUH

#include "correctly_rounded_ops.cuh"

#define CR_EXP_SCALE 0x40b71547652b82feULL
#define CR_EXP_L2H 0x3f262e42ff000000ULL
#define CR_EXP_L2L 0x3d0718432a1b0e26ULL
#define CR_EXP_L2LL 0x3999ff0342542fc3ULL
#define CR_EXP_FAST_EPS 0x3c12700000000000ULL
#define CR_EXP_TINY_ABS_BITS 0x3c90000000000000ULL
#define CR_EXP_LARGE_ABS_BITS 0x40862e42fefa39f0ULL
#define CR_EXP_ZERO_ABS_BITS 0x40874910d52d3052ULL
#define CR_EXP_SUBNORMAL_BITS 0xc086232bdd7abcd2ULL

struct CrExpReduced {
    double t;
    cr_i64 ie;
    double th;
    double tl;
};

// t = round(x * 2^12 / log 2), 2^ie its power of two, th + tl = 2^((t mod 2^12) / 2^12).
__device__ __forceinline__ CrExpReduced cr_exp_reduce(double x) {
    CrExpReduced r;
    r.t = cr_round_ties_even(cr_mul(x, cr_from_bits(CR_EXP_SCALE)));
    const cr_i64 jt = cr_d2i(r.t);
    const unsigned int i0 = (unsigned int)((jt >> 6) & 0x3f);
    const unsigned int i1 = (unsigned int)(jt & 0x3f);
    r.ie = jt >> 12;
    const CrDD th = cr_mul_dd(cr_tab(CR_EXP_T0, 2 * i0 + 1), cr_tab(CR_EXP_T0, 2 * i0), cr_tab(CR_EXP_T1, 2 * i1 + 1),
                              cr_tab(CR_EXP_T1, 2 * i1));
    r.th = th.h;
    r.tl = th.l;
    return r;
}

// 2^(-1022 - ie), which puts the rounding point of a sum in [1, 2) on the subnormal grid.
__device__ __forceinline__ double cr_exp_subnormal_offset(cr_i64 ie) { return cr_from_bits(((cr_u64)(1 - ie)) << 52); }

// Clear the exponent field: the rounded subnormal result scaled by 2^ie.
__device__ __forceinline__ double cr_exp_clear_exponent(double x) { return cr_from_bits(cr_bits(x) & (~0ULL >> 12)); }

// Horner evaluation of CR_EXP_ACCURATE_POLY in double-double, highest degree first.
__device__ __forceinline__ CrDD cr_exp_poly_dd(double xh, double xl) {
    const int n = (int)CR_LEN(CR_EXP_ACCURATE_POLY) / 2;
    CrDD c = cr_dd(cr_tab(CR_EXP_ACCURATE_POLY, 2 * (n - 1)), cr_tab(CR_EXP_ACCURATE_POLY, 2 * (n - 1) + 1));
    for (int i = n - 2; i >= 0; --i) {
        c = cr_fma_dw(xh, xl, c.h, c.l, cr_tab(CR_EXP_ACCURATE_POLY, 2 * i), cr_tab(CR_EXP_ACCURATE_POLY, 2 * i + 1));
    }
    return c;
}

// The listed result for x before scaling by 2^ie, or fallback when x is not listed.
__device__ __forceinline__ double cr_exp_exception(double x, double fallback) {
    unsigned int a = 0;
    unsigned int b = (unsigned int)CR_LEN(CR_EXP_EXCEPTIONS) / 3;
    while (a + 1 < b) {
        const unsigned int m = (a + b) / 2;
        if (x < cr_tab(CR_EXP_EXCEPTIONS, 3 * m)) {
            b = m;
        } else {
            a = m;
        }
    }
    if (x == cr_tab(CR_EXP_EXCEPTIONS, 3 * a)) {
        return cr_add(cr_tab(CR_EXP_EXCEPTIONS, 3 * a + 1), cr_tab(CR_EXP_EXCEPTIONS, 3 * a + 2));
    }
    return fallback;
}

// Accurate path. Assumes |x| > 2^-54.
__device__ __noinline__ double cr_exp_accurate(double x) {
    const cr_u64 ix = cr_bits(x);
    const CrExpReduced r = cr_exp_reduce(x);
    const double l2l = cr_from_bits(CR_EXP_L2L);
    const double dx = cr_sub(x, cr_mul(cr_from_bits(CR_EXP_L2H), r.t));
    const double dxl0 = cr_mul(l2l, r.t);
    const double dxll = cr_add(cr_mul(cr_from_bits(CR_EXP_L2LL), r.t), cr_fma(l2l, r.t, -dxl0));
    const CrDD d = cr_fast_two_sum(dx, dxl0);
    const double dxh = d.h;
    const double dxl = cr_add(d.l, dxll);
    CrDD f = cr_exp_poly_dd(dxh, dxl);
    f = cr_mul_dd(dxh, dxl, f.h, f.l);
    f = cr_mul_dd(f.h, f.l, r.th, r.tl);
    f = cr_fast_sum(r.th, r.tl, f.h, f.l);
    if (ix > CR_EXP_SUBNORMAL_BITS) {
        const CrDD s = cr_fast_two_sum(cr_exp_subnormal_offset(r.ie), f.h);
        return cr_exp_clear_exponent(cr_add(s.h, cr_add(f.l, s.l)));
    }
    const CrDD s = cr_fast_two_sum(f.h, f.l);
    const cr_u64 dist = (cr_bits(s.l) + 4) & (~0ULL >> 12);
    const double value = dist <= 4 ? cr_exp_exception(x, s.h) : s.h;
    return cr_add_exponent(value, r.ie);
}

// e^x, correctly rounded to nearest, ties to even.
__device__ __forceinline__ double cr_exp(double x) {
    const cr_u64 ix = cr_bits(x);
    const cr_u64 aix = ix & CR_ABS_MASK;
    if (aix <= CR_EXP_TINY_ABS_BITS) return cr_add(1.0, x);
    if (aix >= CR_EXP_LARGE_ABS_BITS) {
        if (aix > CR_INF_BITS) return cr_nan();
        if (aix == CR_INF_BITS) return (ix >> 63) == 1 ? 0.0 : cr_inf();
        if ((ix >> 63) == 0) return cr_inf();
        if (aix >= CR_EXP_ZERO_ABS_BITS) return 0.0;
    }
    const CrExpReduced r = cr_exp_reduce(x);
    const double dx = cr_add(cr_sub(x, cr_mul(cr_from_bits(CR_EXP_L2H), r.t)), cr_mul(cr_from_bits(CR_EXP_L2L), r.t));
    const double dx2 = cr_mul(dx, dx);
    const double p = cr_add(cr_add(cr_tab(CR_EXP_FAST_POLY, 0), cr_mul(dx, cr_tab(CR_EXP_FAST_POLY, 1))),
                            cr_mul(dx2, cr_add(cr_tab(CR_EXP_FAST_POLY, 2), cr_mul(dx, cr_tab(CR_EXP_FAST_POLY, 3)))));
    double fh = r.th;
    const double tx = cr_mul(r.th, dx);
    double fl = cr_add(r.tl, cr_mul(tx, p));
    const double eps = cr_from_bits(CR_EXP_FAST_EPS);
    if (ix > CR_EXP_SUBNORMAL_BITS) {
        const CrDD s = cr_fast_two_sum(cr_exp_subnormal_offset(r.ie), fh);
        fh = s.h;
        fl = cr_add(fl, s.l);
        const double ub = cr_add(fh, cr_add(fl, eps));
        const double lb = cr_add(fh, cr_sub(fl, eps));
        if (ub != lb) return cr_exp_accurate(x);
        return cr_exp_clear_exponent(lb);
    }
    const double ub = cr_add(fh, cr_add(fl, eps));
    const double lb = cr_add(fh, cr_sub(fl, eps));
    if (ub != lb) return cr_exp_accurate(x);
    return cr_add_exponent(lb, r.ie);
}

#endif
