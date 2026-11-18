// Accurate path of pow: log|x| and exp with 128-bit significands. Mirrors
// src/algorithm/correctly_rounded/pow_accurate.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/pow.c. Copyright (c) 2022-2025 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_POW_ACCURATE_CUH
#define NUMR_CORRECTLY_ROUNDED_POW_ACCURATE_CUH

#include "correctly_rounded_pow_dint.cuh"

// Top 64 bits of sqrt(2) as a significand.
#define CR_POW_SQRT2_HI 0xb504f333f9de6484ULL

__device__ __forceinline__ CrDint cr_pow_log_coef(unsigned int k) { return cr_dint_table(CR_POW_DINT_LOG_POLY, k); }

// log(1 + z) for |z| <= 2^-13 with a zero low word.
__device__ __noinline__ CrDint cr_pow_p_2(CrDint z) {
    // The degree 6 to 9 coefficients have 64 bits.
    CrDint r = cr_pd_mul_11(z, cr_pow_log_coef(0));
    r = cr_pd_add_11(cr_pow_log_coef(1), r);
    for (unsigned int k = 2; k < 4; ++k) {
        r = cr_pd_mul_11(z, r);
        r = cr_pd_add_11(cr_pow_log_coef(k), r);
    }
    r = cr_pd_mul_11(z, r);
    r = cr_pd_add(cr_pow_log_coef(4), r);
    for (unsigned int k = 5; k < 9; ++k) {
        r = cr_pd_mul_21(r, z);
        r = cr_pd_add(cr_pow_log_coef(k), r);
    }
    return cr_pd_mul_21(r, z);
}

// log(x) for positive x with 53 significant bits.
__device__ __noinline__ CrDint cr_pow_log_2(CrDint x) {
    cr_i64 e = x.ex;
    unsigned int i;
    if (x.hi > CR_POW_SQRT2_HI) {
        e += 1;
        i = (unsigned int)(x.hi >> 57);
    } else {
        i = (unsigned int)(x.hi >> 56);
    }
    x.ex -= e;
    // 0.9921875 <= z <= 1.0078125, exact: 53 bits times 9 bits fit in 64.
    CrDint z = cr_pd_mul_11(x, cr_dint_table(CR_POW_DINT_INVERSE_1, i - 90));
    const unsigned int j = (unsigned int)(z.hi >> (50 - z.ex));
    z = cr_pd_mul_11(z, cr_dint_table(CR_POW_DINT_INVERSE_2, j - 8128));
    // |z - 1| <= 2^-13, and the subtraction is exact.
    z = cr_pd_add(cr_dint_table(CR_LOG_DINT_MINUS_ONE, 0), z);
    const CrDint r = cr_pd_mul_int64(cr_dint_table(CR_LOG_DINT_LN2, 0), e);
    CrDint p = cr_pow_p_2(z);
    p = cr_pd_add(cr_dint_table(CR_POW_DINT_LOG_INV_2, j - 8128), p);
    p = cr_pd_add(cr_dint_table(CR_POW_DINT_LOG_INV_1, i - 90), p);
    return cr_pd_add(p, r);
}

// exp(y) for |y| < 0.00016923.
__device__ __noinline__ CrDint cr_pow_q_2(CrDint y) {
    const cr_u64* c = CR_POW_DINT_EXP_POLY;
    CrDint r = cr_pd_mul_11(y, cr_dint_table(c, 0));
    r = cr_pd_add_11(cr_dint_table(c, 1), r);
    r = cr_pd_mul_11(y, r);
    r = cr_pd_add_11(cr_dint_table(c, 2), r);
    r = cr_pd_mul_11(y, r);
    r = cr_pd_add(cr_dint_table(c, 3), r);
    for (unsigned int k = 4; k < 8; ++k) {
        r = cr_pd_mul(y, r);
        r = cr_pd_add(cr_dint_table(c, k), r);
    }
    return r;
}

// exp(x) for |x| < 744.45. For |x| >= 2^10 the result is a marker with exponent 1025
// (overflow) or -1076 (underflow), which rounds to inf or 0.
__device__ __noinline__ CrDint cr_pow_exp_2(CrDint x) {
    if (x.ex >= 10) return cr_dint(x.hi, x.lo, x.sgn == 1 ? -1076 : 1025, 0);
    // Only used to round x * 2^12 / log(2) to an integer.
    const cr_i64 k = cr_pd_to_i64(cr_pd_mul_11(x, cr_dint_table(CR_POW_DINT_LOG2_INV, 0)));
    CrDint kl = cr_pd_mul_int64(cr_dint_table(CR_LOG_DINT_LN2, 0), k);
    kl.ex -= 12;
    kl.sgn = kl.sgn == 0 ? 1 : 0;
    const CrDint y = cr_pd_add(x, kl);
    const unsigned int i2 = (unsigned int)((k >> 6) & 0x3f);
    const unsigned int i1 = (unsigned int)(k & 0x3f);
    CrDint r = cr_pow_q_2(y);
    r = cr_pd_mul(cr_dint_table(CR_POW_DINT_T1, i2), r);
    r = cr_pd_mul(cr_dint_table(CR_POW_DINT_T2, i1), r);
    r.ex += k >> 12;
    return r;
}

// |x|^y for finite nonzero x and y.
__device__ __forceinline__ CrDint cr_pow_2(double x, double y) {
    CrDint xd = cr_dint_from_f64(x);
    xd.sgn = 0;
    return cr_pow_exp_2(cr_pd_mul_21(cr_pow_log_2(xd), cr_dint_from_f64(y)));
}

// Whether the rounding of r to binary64 is certain under the accurate-path error bound.
__device__ __forceinline__ bool cr_pow_rounding_is_certain(CrDint r) {
    if (r.ex < -1022) {
        // -1075 <= r.ex <= -1023, so 1 <= ex <= 53.
        const int ex = (int)(-(1022 + r.ex));
        const cr_u64 m = (r.lo >> (10 + ex)) | (r.hi << (54 - ex));
        return m + 14 > 28;
    }
    const cr_u64 lo = (r.lo >> 10) | (r.hi << 54);
    return lo + 28 > 56;
}

#endif
