// Correctly rounded binary64 log. Mirrors src/algorithm/correctly_rounded/log.rs and
// log_accurate.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log/log.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2022-2026 INRIA and CERN. Authors: Paul Zimmermann and Tom Hubrecht.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_LOG_CUH
#define NUMR_CORRECTLY_ROUNDED_LOG_CUH

#include "correctly_rounded_dint.cuh"

#define CR_LOG_OFFSET 362
#define CR_LOG_LOG2_H 0x3fe62e42fefa3800ULL
#define CR_LOG_LOG2_L 0x3d2ef35793c76730ULL
#define CR_LOG_FAST_ERR 0x3bab600000000000ULL
#define CR_LOG_TINY_EPS 0x3c2d100000000000ULL
#define CR_LOG_TWO_POW_52 0x4330000000000000ULL
#define CR_LOG_SQRT2_SIGNIFICAND 0x0016a09e667f3bcdULL
#define CR_LOG_TINY_LOW_BITS 0x3feff00000000000ULL
#define CR_LOG_TINY_HIGH_BITS 0x3ff0080000000000ULL
#define CR_LOG_DINT_SQRT2_SIGNIFICAND 0xb504f333f9de6484ULL

// log(1 + z) by Horner's rule over CR_LOG_DINT_POLY, then a final product with z.
__device__ __forceinline__ CrDint cr_log_dint_poly(CrDint z) {
    const unsigned int n = (unsigned int)CR_LEN(CR_LOG_DINT_POLY) / 4;
    CrDint r = cr_dint_table(CR_LOG_DINT_POLY, 0);
    for (unsigned int k = 1; k < n; ++k) {
        r = cr_dint_mul(r, z);
        r = cr_dint_add(cr_dint_table(CR_LOG_DINT_POLY, k), r);
    }
    return cr_dint_mul(r, z);
}

// log(x) for a positive finite x != 1, correctly rounded, in dint arithmetic:
// E * log(2) - log(r) + log(1 + z) with z = x * r - 1.
__device__ __noinline__ double cr_log_accurate(double x) {
    CrDint d = cr_dint_from_f64(x);
    cr_i64 e = d.ex;
    unsigned int i = (unsigned int)(d.hi >> 55);
    if (d.hi > CR_LOG_DINT_SQRT2_SIGNIFICAND) {
        e += 1;
        i >>= 1;
    }
    d.ex -= e;
    CrDint z = cr_dint_mul(d, cr_dint_table(CR_LOG_DINT_INVERSE, i - 128));
    z = cr_dint_add(cr_dint_table(CR_LOG_DINT_MINUS_ONE, 0), z);
    const CrDint r = cr_dint_mul_int(e, cr_dint_table(CR_LOG_DINT_LN2, 0));
    CrDint p = cr_log_dint_poly(z);
    p = cr_dint_add(cr_dint_table(CR_LOG_DINT_LOG_INV, i - 128), p);
    return cr_dint_to_f64(cr_dint_add(p, r));
}

__device__ __forceinline__ CrDD cr_log_fast(int e, double m) {
    const cr_u64 bits = 0x0010000000000000ULL | (cr_bits(m) & CR_FRACTION_MASK);
    // Above sqrt(2) the argument is halved to avoid cancellation.
    const int c = bits >= CR_LOG_SQRT2_SIGNIFICAND ? 1 : 0;
    e += c;
    const unsigned int i = (unsigned int)(bits >> (c ? 44 : 43));
    const double y = cr_mul(m, c ? 0.5 : 1.0);
    const unsigned int k = i - CR_LOG_OFFSET;
    const double r = cr_tab(CR_LOG_INVERSE, k);
    const double l1 = cr_tab(CR_LOG_LOG_INV, 2 * k);
    const double l2 = cr_tab(CR_LOG_LOG_INV, 2 * k + 1);
    const double z = cr_fma(r, y, -1.0);
    const double z2 = cr_mul(z, z);
    const double p45 = cr_fma(cr_tab(CR_LOG_FAST_POLY, 5), z, cr_tab(CR_LOG_FAST_POLY, 4));
    const double p23 = cr_fma(cr_tab(CR_LOG_FAST_POLY, 3), z, cr_tab(CR_LOG_FAST_POLY, 2));
    double ph = cr_fma(p45, z2, p23);
    ph = cr_fma(ph, z, cr_tab(CR_LOG_FAST_POLY, 1));
    ph = cr_mul(ph, z2);
    const double ee = cr_i2d(e);
    const CrDD s = cr_fast_two_sum(cr_fma(ee, cr_from_bits(CR_LOG_LOG2_H), l1), z);
    const double l = cr_add(ph, cr_add(s.l, l2));
    return cr_dd(s.h, cr_fma(ee, cr_from_bits(CR_LOG_LOG2_L), l));
}

__device__ __forceinline__ double cr_log_tiny(double x) {
    if (x == 1.0) return 0.0;
    const double r = cr_sub(x, 1.0);
    const double r2 = cr_mul(r, r);
    const double p6 = cr_add(cr_tab(CR_LOG_TINY_POLY, 4), cr_mul(r, cr_tab(CR_LOG_TINY_POLY, 5)));
    double p4 = cr_add(cr_tab(CR_LOG_TINY_POLY, 2), cr_mul(r, cr_tab(CR_LOG_TINY_POLY, 3)));
    double p2 = cr_add(cr_tab(CR_LOG_TINY_POLY, 0), cr_mul(r, cr_tab(CR_LOG_TINY_POLY, 1)));
    p4 = cr_add(p4, cr_mul(r2, p6));
    p2 = cr_add(p2, cr_mul(r2, p4));
    const CrDD s = cr_fast_two_sum(r, cr_mul(r2, p2));
    const double eps = cr_mul(cr_from_bits(CR_LOG_TINY_EPS), r);
    const double lb = cr_add(s.h, cr_sub(s.l, eps));
    const double ub = cr_add(s.h, cr_add(s.l, eps));
    if (lb == ub) return lb;
    return cr_log_accurate(x);
}

// Natural logarithm, correctly rounded to nearest, ties to even.
__device__ __forceinline__ double cr_log(double x) {
    cr_u64 v = cr_bits(x);
    // The sign bit joins the exponent here, so every negative x lands above 0x400.
    int e = (int)(v >> 52) - 0x3ff;
    if (e >= 0x400 || e == -0x3ff) {
        if (e == 0x400 || (e == 0xc00 && v != CR_NEG_INF_BITS)) {
            // +inf, or NaN of either sign.
            return cr_is_nan(x) ? cr_nan() : x;
        }
        if (x <= 0.0) return x < 0.0 ? cr_nan() : cr_neg_inf();
        // Positive subnormal.
        v = cr_bits(cr_mul(x, cr_from_bits(CR_LOG_TWO_POW_52)));
        e = (int)(v >> 52) - 0x3ff - 52;
    }
    if (v >= CR_LOG_TINY_LOW_BITS && v < CR_LOG_TINY_HIGH_BITS) return cr_log_tiny(x);
    const double m = cr_from_bits((0x3ffULL << 52) | (v & CR_FRACTION_MASK));
    const CrDD f = cr_log_fast(e, m);
    const double err = cr_from_bits(CR_LOG_FAST_ERR);
    const double left = cr_add(f.h, cr_sub(f.l, err));
    const double right = cr_add(f.h, cr_add(f.l, err));
    if (left == right) return left;
    return cr_log_accurate(x);
}

#endif
