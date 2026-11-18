// Extended path of pow: log|x| and exp with 256-bit significands. Mirrors
// src/algorithm/correctly_rounded/pow_extended.rs.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/pow.c. Copyright (c) 2022-2025 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_POW_EXTENDED_CUH
#define NUMR_CORRECTLY_ROUNDED_POW_EXTENDED_CUH

#include "correctly_rounded_pow_accurate.cuh"
#include "correctly_rounded_qint_mul.cuh"

// Margin of the extended rounding test, in units of the 64 bits after the third limb's
// round position.
#define CR_POW_ERR_BND_3 60ULL

// log(1 + z) for |z| <= 2^-13 with only the top limb nonzero.
__device__ __noinline__ CrQint cr_pow_p_3(CrQint z) {
    const cr_u64* c = CR_POW_QINT_LOG_POLY;
    CrQint r = cr_qint_mul_11(cr_qint_table(c, 0), z);
    r = cr_qint_add_22(cr_qint_table(c, 1), r);
    for (unsigned int k = 2; k < 4; ++k) {
        r = cr_qint_mul_11(r, z);
        r = cr_qint_add_22(cr_qint_table(c, k), r);
    }
    for (unsigned int k = 4; k < 8; ++k) {
        r = cr_qint_mul_21(r, z);
        r = cr_qint_add_22(cr_qint_table(c, k), r);
    }
    for (unsigned int k = 8; k < 14; ++k) {
        r = cr_qint_mul_31(r, z);
        r = cr_qint_add(cr_qint_table(c, k), r);
    }
    for (unsigned int k = 14; k < 18; ++k) {
        r = cr_qint_mul_41(r, z);
        r = cr_qint_add(cr_qint_table(c, k), r);
    }
    return cr_qint_mul_41(r, z);
}

// log(x) for positive x with 53 significant bits.
__device__ __noinline__ CrQint cr_pow_log_3(CrQint x) {
    cr_i64 e = x.ex;
    unsigned int i;
    if (x.rh.hi > CR_POW_SQRT2_HI) {
        e += 1;
        i = (unsigned int)(x.rh.hi >> 57);
    } else {
        i = (unsigned int)(x.rh.hi >> 56);
    }
    x.ex -= e;
    CrQint z = cr_qint_mul(x, cr_qint_widened(CR_POW_DINT_INVERSE_1, i - 90));
    const unsigned int j = (unsigned int)(z.rh.hi >> (50 - z.ex));
    z = cr_qint_mul(z, cr_qint_widened(CR_POW_DINT_INVERSE_2, j - 8128));
    // |z - 1| <= 2^-13 fits in 64 bits, and the subtraction is exact.
    z = cr_qint_add(cr_qint_widened(CR_LOG_DINT_MINUS_ONE, 0), z);
    const CrQint r = cr_qint_mul_int(e, cr_qint_table(CR_POW_QINT_LN2, 0));
    CrQint p = cr_pow_p_3(z);
    // Smallest terms first.
    p = cr_qint_add(cr_qint_table(CR_POW_QINT_LOG_INV_2, j - 8128), p);
    p = cr_qint_add(cr_qint_table(CR_POW_QINT_LOG_INV_1, i - 90), p);
    return cr_qint_add(p, r);
}

// exp(y) for |y| < 0.00016923.
__device__ __noinline__ CrQint cr_pow_q_3(CrQint y) {
    const cr_u64* c = CR_POW_QINT_EXP_POLY;
    // Degrees 13 and 14 have 64 bits, degrees 8 to 12 have 128 bits.
    CrQint r = cr_qint_mul_11(y, cr_qint_table(c, 0));
    r = cr_qint_add_22(cr_qint_table(c, 1), r);
    for (unsigned int k = 2; k < 7; ++k) {
        r = cr_qint_mul_22(y, r);
        r = cr_qint_add_22(cr_qint_table(c, k), r);
    }
    for (unsigned int k = 7; k < 12; ++k) {
        r = cr_qint_mul_33(y, r);
        r = cr_qint_add(cr_qint_table(c, k), r);
    }
    for (unsigned int k = 12; k < 15; ++k) {
        r = cr_qint_mul(y, r);
        r = cr_qint_add(cr_qint_table(c, k), r);
    }
    return r;
}

// exp(x) for |x| < 744.45. For |x| >= 2^10 the result is the overflow or underflow
// marker of the accurate path.
__device__ __noinline__ CrQint cr_pow_exp_3(CrQint x) {
    if (x.ex >= 10) return cr_qint(x.rh, x.rl, x.sgn == 1 ? -1076 : 1025, 0);
    const cr_i64 k = cr_qint_to_i64(cr_qint_mul_11(x, cr_qint_widened(CR_POW_DINT_LOG2_INV, 0)));
    CrQint kl = cr_qint_mul_int(k, cr_qint_table(CR_POW_QINT_LN2, 0));
    kl.ex -= 12;
    kl.sgn = kl.sgn == 0 ? 1 : 0;
    const CrQint y = cr_qint_add(x, kl);
    const unsigned int i2 = (unsigned int)((k >> 6) & 0x3f);
    const unsigned int i1 = (unsigned int)(k & 0x3f);
    CrQint r = cr_pow_q_3(y);
    r = cr_qint_mul(cr_qint_table(CR_POW_QINT_T1, i2), r);
    r = cr_qint_mul(cr_qint_table(CR_POW_QINT_T2, i1), r);
    r.ex += k >> 12;
    return r;
}

// The extended result for finite nonzero x and y with sign s. *certified says whether
// the rounding test certified it.
__device__ __noinline__ double cr_pow_3(double x, double y, double s, bool* certified) {
    CrQint qx = cr_qint_from_dint(cr_dint_from_f64(x));
    qx.sgn = 0;
    const CrQint qr = cr_qint_mul_41(cr_pow_log_3(qx), cr_qint_from_dint(cr_dint_from_f64(y)));
    CrQint qz = cr_pow_exp_3(qr);
    const cr_u64 r1 = (qz.rh.hi << 54) | (qz.rh.lo >> 10);
    const cr_u64 r2 = (qz.rh.lo << 54) | (qz.rl.hi >> 10);
    const cr_u64 r3 = (qz.rl.hi << 54) | (qz.rl.lo >> 10);
    const bool near_boundary = (r1 == 0 && r2 == 0 && r3 <= CR_POW_ERR_BND_3) ||
                               (~r1 == 0 && ~r2 == 0 && ~r3 <= CR_POW_ERR_BND_3);
    qz.sgn = s == -1.0 ? 1 : 0;
    qz.rl.lo &= ~0ULL << 10;
    *certified = true;
    if (!near_boundary) return cr_qint_to_f64(qz);
    // |y log|x|| < 2^-55: 1 - 2^-54 < |x|^y < 1 + 2^-53 rounds to 1, and s = 1 here.
    if (qr.ex < -56) return 1.0;
    *certified = false;
    return cr_qint_to_f64(qz);
}

#endif
