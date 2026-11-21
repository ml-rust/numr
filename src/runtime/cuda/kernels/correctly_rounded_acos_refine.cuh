// Accurate path of the correctly rounded binary64 acos: angle rotation by a multiple of
// pi/64, an asin series near zero, and a triple-double sum. Mirrors
// src/algorithm/correctly_rounded/acos_refine.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/acos/acos.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2024-2025 Alexei Sibidanov.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_ACOS_REFINE_CUH
#define NUMR_CORRECTLY_ROUNDED_ACOS_REFINE_CUH

#include "correctly_rounded_ops.cuh"

#define CR_ACOS_HALF_PI_HI 0x3ff921fb54442d18ULL
#define CR_ACOS_SIXTY_FOUR_OVER_PI 0x40345f306dc9c883ULL
#define CR_ACOS_MAGIC 0x3fb8000000000000ULL
#define CR_ACOS_PI64_H 0x3fa921fb54442d00ULL
#define CR_ACOS_PI64_M 0x3ca8469898cc5180ULL
#define CR_ACOS_PI64_L 0xb99fc8f8cbb5bf6cULL

// (s, e) with s + e == a + b exactly: Knuth's two-sum in acos.c's operation order.
__device__ __forceinline__ CrDD cr_acos_two_sum(double a, double b) {
    const double s = cr_add(a, b);
    const double a_prime = cr_sub(s, b);
    const double b_prime = cr_sub(s, a_prime);
    return cr_dd(s, cr_add(cr_sub(a, a_prime), cr_sub(b, b_prime)));
}

// Double-double sum with cr_acos_two_sum on the high parts.
__device__ __forceinline__ CrDD cr_acos_sum(double xh, double xl, double ch, double cl) {
    const CrDD s = cr_acos_two_sum(xh, ch);
    return cr_dd(s.h, cr_add(cr_add(xl, cl), s.l));
}

// True when the bits of pl below 2^-107 relative to the result's unit round to zero, so
// ph + pl may sit on a midpoint.
__device__ __forceinline__ bool cr_acos_near_rounding_boundary(double pl) {
    const cr_u64 t = cr_bits(pl);
    cr_i64 e = (cr_i64)((t >> 52) & 0x7ff) - 1023;
    e = 52 - (107 + e);
    e = e < 0 ? 0 : (e > 52 ? 52 : e);
    const cr_u64 m = (1ULL << 52) - (1ULL << e);
    if (e == 0) e = 64;
    return ((t + (1ULL << (e - 1))) & m) == 0;
}

// The listed result for x in *out, or false when x is not listed.
__device__ __forceinline__ bool cr_acos_exception(double x, double* out) {
    const unsigned int n = (unsigned int)CR_LEN(CR_ACOS_EXCEPTIONS) / 3;
    for (unsigned int k = 0; k < n; ++k) {
        if (cr_tab(CR_ACOS_EXCEPTIONS, 3 * k) == x) {
            *out = cr_add(cr_tab(CR_ACOS_EXCEPTIONS, 3 * k + 1), cr_tab(CR_ACOS_EXCEPTIONS, 3 * k + 2));
            return true;
        }
    }
    return false;
}

// Accurate acos(x) for |x| < 1. phi is the fast-path lower bound.
__device__ __noinline__ double cr_acos_refine(double x, double phi) {
    // With x = cos(y), ch + cl approximates sin(y) = sqrt(1 - x^2).
    const CrDD s2 = cr_two_prod(x, x);
    CrDD c2 = cr_fast_two_sum(1.0, -s2.h);
    c2 = cr_fast_two_sum(c2.h, cr_sub(c2.l, s2.l));
    const double ch = cr_sqrt(c2.h);
    const double cl = cr_mul(cr_sub(c2.l, cr_fma(ch, ch, -c2.h)), cr_div(0.5, ch));

    // y = jt * pi/64 - delta with jt = 32 -+ jf and |delta| < pi/128.
    const double jfd = cr_round_ties_even(
        cr_mul(cr_abs(cr_sub(phi, cr_from_bits(CR_ACOS_HALF_PI_HI))), cr_from_bits(CR_ACOS_SIXTY_FOUR_OVER_PI)));
    const unsigned int jf = (unsigned int)cr_d2i(jfd);
    const double cos_h = cr_tab(CR_ACOS_S, 2 * (32 - jf) + 1);
    const double cos_l = cr_tab(CR_ACOS_S, 2 * (32 - jf));
    const double sin_h = cr_tab(CR_ACOS_S, 2 * jf + 1);
    const double sin_l = cr_tab(CR_ACOS_S, 2 * jf);
    const double dsh = cr_sub(cr_abs(x), sin_h);
    const double dsl = -sin_l;
    const double dch = cr_sub(ch, cos_h);
    const double dcl = cr_sub(cl, cos_l);
    const double magic = cr_from_bits(CR_ACOS_MAGIC);
    const double sc = cr_sub(cr_fma(sin_h, dch, magic), magic);
    const double dsc = cr_fma(sin_h, dch, -sc);
    const double cs = cr_sub(cr_fma(cos_h, dsh, magic), magic);
    const double dcs = cr_fma(cos_h, dsh, -cs);
    const double v0 = cr_sub(cs, sc);
    const double dv0 = cr_sub(cr_sub(cr_add(cr_mul(cos_h, dsl), cr_mul(cos_l, dsh)),
                                     cr_add(cr_mul(sin_h, dcl), cr_mul(sin_l, dch))),
                              cr_sub(dsc, dcs));
    const CrDD vv = cr_fast_two_sum(v0, dv0);
    const double sgn = cr_copysign(1.0, x);
    const double jt = cr_sub(32.0, cr_mul(cr_i2d((cr_i64)jf), sgn));

    // fh + fl = asin(-delta), an odd series in v + dv.
    const CrDD v2 = cr_mul_dd(vv.h, vv.l, vv.h, vv.l);
    const double v = cr_mul(vv.h, -sgn);
    const double dv = cr_mul(vv.l, -sgn);
    const double low = cr_mul(
        v2.h, cr_add(cr_tab(CR_ACOS_CT, 0),
                     cr_mul(v2.h, cr_add(cr_tab(CR_ACOS_CT, 1), cr_mul(v2.h, cr_tab(CR_ACOS_CT, 2))))));
    CrDD f = cr_horner_dd(v2.h, v2.l, CR_ACOS_C, (int)CR_LEN(CR_ACOS_C) / 2, low);
    f = cr_mul_dd(v, dv, f.h, f.l);

    // ph + pl + ps = jt * pi/64 - delta. jt * PI64_H and jt * PI64_M are exact.
    double ph = cr_mul(jt, cr_from_bits(CR_ACOS_PI64_H));
    double pl = cr_mul(cr_from_bits(CR_ACOS_PI64_M), jt);
    double ps = cr_mul(cr_from_bits(CR_ACOS_PI64_L), jt);
    CrDD p = cr_acos_sum(f.h, f.l, pl, ps);
    pl = p.h;
    ps = p.l;
    p = cr_fast_two_sum(ph, pl);
    ph = p.h;
    pl = p.l;
    p = cr_fast_two_sum(pl, ps);
    pl = p.h;
    ps = p.l;
    p = cr_fast_two_sum(ph, pl);
    ph = p.h;
    pl = p.l;
    p = cr_fast_two_sum(pl, ps);
    pl = p.h;
    ps = p.l;

    if (cr_acos_near_rounding_boundary(pl)) {
        double listed;
        if (cr_acos_exception(x, &listed)) return listed;
        // Move pl one ulp toward ps, so the final add sees the sticky information of ps.
        cr_u64 t = cr_bits(pl);
        t = ((cr_bits(ps) ^ t) >> 63) != 0 ? t - 1 : t + 1;
        pl = cr_from_bits(t);
    }
    return cr_add(ph, pl);
}

#endif
