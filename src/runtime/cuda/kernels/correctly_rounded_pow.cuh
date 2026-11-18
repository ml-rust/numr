// Correctly rounded binary64 pow with C Annex F special values. Mirrors
// src/algorithm/correctly_rounded/pow.rs, including the uncertified branch: when the
// 256-bit rounding test fails, the result is the 256-bit approximation rounded to nearest.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/pow.c. Copyright (c) 2022-2025 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_POW_CUH
#define NUMR_CORRECTLY_ROUNDED_POW_CUH

#include "correctly_rounded_pow_exact.cuh"
#include "correctly_rounded_pow_extended.cuh"
#include "correctly_rounded_pow_fast.cuh"

// Results for a NaN or infinite operand. Returns false for finite operands.
__device__ __forceinline__ bool cr_pow_non_finite(double x, double y, double* out) {
    if (cr_is_nan(x)) {
        // Annex F: pow(x, +-0) = 1 even for a NaN x.
        *out = y == 0.0 ? 1.0 : cr_nan();
        return true;
    }
    if (cr_is_nan(y)) {
        *out = x == 1.0 ? 1.0 : cr_nan();
        return true;
    }
    const cr_u64 ix = cr_bits(x);
    const cr_u64 iy = cr_bits(y);
    if (ix == CR_INF_BITS) {
        *out = y == 0.0 ? 1.0 : (y < 0.0 ? 0.0 : cr_inf());
        return true;
    }
    if (ix == CR_NEG_INF_BITS) {
        if (iy == CR_INF_BITS) {
            *out = cr_inf();
            return true;
        }
        if (iy == CR_NEG_INF_BITS) {
            *out = 0.0;
            return true;
        }
        // y == 0 falls through to the x <= 0 branch, which returns 1.
        if (y != 0.0) {
            const bool odd = cr_is_odd_int(y);
            if (y < 0.0) {
                *out = odd ? -0.0 : 0.0;
            } else {
                *out = odd ? cr_neg_inf() : cr_inf();
            }
            return true;
        }
    }
    if (iy == CR_INF_BITS || iy == CR_NEG_INF_BITS) {
        const double ax = cr_abs(x);
        const bool grows = (ax > 1.0) == (iy == CR_INF_BITS);
        *out = ax == 1.0 ? 1.0 : (grows ? cr_inf() : 0.0);
        return true;
    }
    return false;
}

// Results for a finite x <= 0. Returns false when x < 0 and y is a nonzero integer.
__device__ __forceinline__ bool cr_pow_non_positive_base(double x, double y, double* out) {
    if (y == 0.0) {
        *out = 1.0;
        return true;
    }
    if (x == 0.0) {
        const bool negative_zero = cr_bits(x) != 0;
        const bool odd = cr_is_odd_int(y);
        if (y < 0.0) {
            *out = (odd && negative_zero) ? cr_neg_inf() : cr_inf();
        } else {
            *out = (odd && negative_zero) ? -0.0 : 0.0;
        }
        return true;
    }
    if (!cr_is_int(y)) {
        *out = cr_nan();
        return true;
    }
    return false;
}

// The accurate, exact-case and extended paths, after the fast rounding test failed.
__device__ __noinline__ double cr_pow_slow(double x, double ax, double y, double s) {
    CrDint r = cr_pow_2(ax, y);
    // |x|^y < 2^-1075 rounds to a zero of the result's sign.
    if (r.ex < -1075) return cr_mul(0.5, cr_mul(s, cr_from_bits(1)));
    r.sgn = s == -1.0 ? 1 : 0;
    if (cr_pow_rounding_is_certain(r)) return cr_pd_to_f64(r);
    double value;
    if (cr_pow_exact_value(x, y, r, cr_pow_is_exact(ax, y), &value)) return value;
    bool certified;
    return cr_pow_3(ax, y, s, &certified);
}

// s * |x|^y for finite nonzero x and y, ax = |x|.
__device__ __forceinline__ double cr_pow_finite(double x, double ax, double y, double s) {
    const CrPowLog lg = cr_pow_log_1(ax);
    double lh = lg.h;
    double ll = lg.l;
    // A NaN logarithm sends |y| < 2^-969 and |y| >= 2^1014 to the accurate path.
    const cr_u64 ey = (cr_bits(y) >> 52) & 0x7ff;
    if (!(ey >= 0x36 && ey < 0x7f5)) {
        lh = cr_nan();
        ll = cr_nan();
    }
    const CrDD r = cr_pow_s_mul(y, lh, ll);
    const CrDD res = cr_pow_exp_1(r.h, r.l, s);
    const double err = cr_from_bits(lg.cancel ? CR_POW_FAST_ERR_CANCEL : CR_POW_FAST_ERR);
    const double res_min = cr_add(res.h, cr_fma(err, -res.h, res.l));
    const double res_max = cr_add(res.h, cr_fma(err, res.h, res.l));
    // res_max is +-0 in the underflow case.
    if (res_min == res_max) return res_max;
    if (y == 1.0) return cr_mul(s, ax);
    if (y == 2.0) return cr_mul(ax, ax);
    if (y == 0.0) return 1.0;
    return cr_pow_slow(x, ax, y, s);
}

// x^y, correctly rounded to nearest, ties to even, for every pair whose 256-bit rounding
// test succeeds.
__device__ __forceinline__ double cr_pow(double x, double y) {
    const cr_u64 ix = cr_bits(x);
    const cr_u64 iy = cr_bits(y);
    double out;
    // The unsigned compare also admits every negative operand; finite ones fall through.
    if ((ix >= CR_INF_BITS || iy >= CR_INF_BITS) && cr_pow_non_finite(x, y, &out)) return out;
    double s = 1.0;
    double ax = x;
    if (x <= 0.0) {
        if (cr_pow_non_positive_base(x, y, &out)) return out;
        // x < 0 and y is a nonzero integer: |x|^y with the parity sign.
        if (cr_abs(y) < 9007199254740992.0 && (cr_d2i(y) & 1) == 1) s = -1.0;
        ax = -x;
    }
    return cr_pow_finite(x, ax, y, s);
}

#endif
