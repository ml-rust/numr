// 128-bit dint operations of the accurate pow path. Mirrors
// src/algorithm/correctly_rounded/pow_dint.rs: a shifted operand is truncated, a
// subtraction under Sterbenz's condition is exact, and each product drops a fixed set of
// partial products. They differ from the log versions in correctly_rounded_dint.cuh.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/dint.h. Copyright (c) 2022, 2023 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_POW_DINT_CUH
#define NUMR_CORRECTLY_ROUNDED_POW_DINT_CUH

#include "correctly_rounded_dint.cuh"

// Zero, with the exponent that makes cr_pd_to_f64 return +0.
__device__ __forceinline__ CrDint cr_pd_zero() { return cr_dint(0, 0, -1076, 0); }

// Order of |a| and |b|. A value with a zero high word counts as zero.
__device__ __forceinline__ int cr_pd_cmp_abs(CrDint a, CrDint b) {
    if (a.hi == 0) return b.hi == 0 ? 0 : -1;
    if (b.hi == 0) return 1;
    if (a.ex != b.ex) return a.ex < b.ex ? -1 : 1;
    return cr_u128_cmp(cr_dint_sig(a), cr_dint_sig(b));
}

// a + b with error below 2 ulps of the 128-bit result. Exact under Sterbenz's condition.
__device__ __noinline__ CrDint cr_pd_add(CrDint a, CrDint b) {
    if ((a.hi | a.lo) == 0) return b;
    const int order = cr_pd_cmp_abs(a, b);
    if (order == 0) {
        if ((a.sgn ^ b.sgn) != 0) return cr_pd_zero();
        a.ex += 1;
        return a;
    }
    if (order < 0) {
        const CrDint t = a;
        a = b;
        b = t;
    }
    const CrU128 big = cr_dint_sig(a);
    CrU128 small = cr_dint_sig(b);
    const cr_u64 k = (cr_u64)a.ex - (cr_u64)b.ex;
    if (k > 0) small = k < 128 ? cr_u128_shr(small, (int)k) : cr_u128(0, 0);
    cr_i64 ex = a.ex;
    CrU128 c;
    if ((a.sgn ^ b.sgn) != 0) {
        c = cr_u128_sub(big, small);
        int shift = cr_u128_clz(c);
        if (shift > 0) {
            if (k == 1) {
                // Sterbenz case: no bit of b is lost.
                c = cr_u128_sub(cr_u128_shl(big, shift), cr_u128_shl(cr_dint_sig(b), shift - 1));
            } else {
                c = cr_u128_sub(cr_u128_shl(big, shift), cr_u128_shl(small, shift));
            }
            ex -= shift;
            shift = cr_clz(c.hi);
        }
        ex -= shift;
        c = cr_u128_shl(c, shift);
    } else {
        bool carry;
        c = cr_u128_add_carry(big, small, &carry);
        if (carry) {
            ex += 1;
            c = cr_u128_or(cr_u128(1ULL << 63, 0), cr_u128_shr(c, 1));
        }
    }
    return cr_dint_parts(c, ex, a.sgn);
}

// a + b on the high words only, with error below 2 ulps of 64 bits. The low word is zero.
__device__ __noinline__ CrDint cr_pd_add_11(CrDint a, CrDint b) {
    if (a.hi == 0) return b;
    if (b.hi == 0) return a;
    int order = a.ex != b.ex ? (a.ex < b.ex ? -1 : 1) : (a.hi != b.hi ? (a.hi < b.hi ? -1 : 1) : 0);
    if (order == 0) {
        if ((a.sgn ^ b.sgn) != 0) return cr_pd_zero();
        a.ex += 1;
        return a;
    }
    if (order < 0) {
        const CrDint t = a;
        a = b;
        b = t;
    }
    const cr_u64 big = a.hi;
    cr_u64 small = b.hi;
    if (a.ex > b.ex) {
        const cr_u64 k = (cr_u64)(a.ex - b.ex);
        small = k < 64 ? small >> k : 0;
    }
    cr_i64 ex = a.ex;
    cr_u64 c;
    if ((a.sgn ^ b.sgn) != 0) {
        c = big - small;
        int shift = cr_clz(c);
        if (shift > 0) {
            c = (big << shift) - (small << shift);
            ex -= shift;
            shift = cr_clz(c);
        }
        ex -= shift;
        c = shift < 64 ? c << shift : 0;
    } else {
        c = big + small;
        if (c < big) {
            ex += 1;
            c = (1ULL << 63) | (c >> 1);
        }
    }
    return cr_dint(c, 0, ex, a.sgn);
}

// Shift a product so its top bit is set, and attach exponent and sign.
__device__ __forceinline__ CrDint cr_pd_normalize(CrU128 product, CrDint a, CrDint b) {
    const cr_u64 top = product.hi >> 63;
    return cr_dint_parts(cr_u128_shl(product, (int)(1 - top)), a.ex + b.ex + (cr_i64)top, a.sgn ^ b.sgn);
}

// a * b with error below 6 ulps: drops lo * lo and the low halves of the cross terms.
__device__ __forceinline__ CrDint cr_pd_mul(CrDint a, CrDint b) {
    const cr_u64 m1 = __umul64hi(a.hi, b.lo);
    const cr_u64 m2 = __umul64hi(a.lo, b.hi);
    CrU128 product = cr_u128_mul(a.hi, b.hi);
    product = cr_u128_add(product, cr_u128_from(m1));
    product = cr_u128_add(product, cr_u128_from(m2));
    return cr_pd_normalize(product, a, b);
}

// a * b ignoring the low word of b, with error below 2 ulps.
__device__ __forceinline__ CrDint cr_pd_mul_21(CrDint a, CrDint b) {
    const CrU128 hi = cr_u128_mul(a.hi, b.hi);
    return cr_pd_normalize(cr_u128_add(hi, cr_u128_from(__umul64hi(a.lo, b.hi))), a, b);
}

// The exact product of the high words of a and b.
__device__ __forceinline__ CrDint cr_pd_mul_11(CrDint a, CrDint b) { return cr_pd_normalize(cr_u128_mul(a.hi, b.hi), a, b); }

// b * a for an integer b with |b| < 2^63, with error below 1 ulp.
__device__ __noinline__ CrDint cr_pd_mul_int64(CrDint a, cr_i64 b) {
    if (b == 0) return cr_pd_zero();
    const cr_u64 c = b < 0 ? (cr_u64)0 - (cr_u64)b : (cr_u64)b;
    const cr_u64 sgn = b < 0 ? (a.sgn == 0 ? 1 : 0) : a.sgn;
    CrU128 r = cr_u128_mul(a.hi, c);
    const int m = r.hi != 0 ? cr_clz(r.hi) : 64;
    r = cr_u128_shl(r, m);
    cr_i64 ex = a.ex + 64 - m;
    const CrU128 l = cr_u128_shr(cr_u128_shl(cr_u128_mul(a.lo, c), m - 1), 63);
    r = cr_u128_add(r, l);
    if (cr_u128_cmp(r, l) < 0) {
        r = cr_u128_or(cr_u128(1ULL << 63, 0), cr_u128_shr(r, 1));
        ex += 1;
    }
    return cr_dint_parts(r, ex, sgn);
}

// a rounded toward zero to an integer. Needs a.ex < 64.
__device__ __forceinline__ cr_i64 cr_pd_to_i64(CrDint a) {
    if (a.ex < 0) return 0;
    const cr_i64 r = (cr_i64)(a.hi >> (63 - a.ex));
    return a.sgn != 0 ? -r : r;
}

// Round to the subnormal grid. Needs ex <= -1023.
__device__ __forceinline__ double cr_pd_round_subnormal(cr_u64 hi, bool sticky, cr_i64 ex, cr_u64 sgn) {
    // The number of bits of hi below 2^-1074.
    const cr_u64 shift = (cr_u64)(-(1011 + ex));
    cr_u64 q;
    if (shift >= 64) {
        // |a| < 2^-1074: the round bit is the top bit of hi only when shift = 64.
        const bool round = (hi >> 63) == 1;
        const bool rest = (hi << 1) != 0 || sticky;
        q = (shift == 64 && round && rest) ? 1 : 0;
    } else {
        q = hi >> shift;
        const cr_u64 round = (hi >> (shift - 1)) & 1;
        const bool rest = (hi << (65 - shift)) != 0 || sticky;
        q += rest ? round : (q & round);
    }
    // q <= 2^52; q = 2^52 is the bit pattern of 2^-1022.
    return cr_from_bits(q | (sgn << 63));
}

// Round (-1)^sgn * hi * 2^(ex - 63), plus sticky below hi, to nearest, ties to even, once
// in every range.
__device__ __noinline__ double cr_pd_round_to_f64(cr_u64 hi, bool sticky, cr_i64 ex, cr_u64 sgn) {
    if (ex < -1022) return cr_pd_round_subnormal(hi, sticky, ex, sgn);
    double r = cr_from_bits((hi >> 11) | (0x3ffULL << 52) | (sgn << 63));
    double rd = 0.0;
    if (((hi >> 10) & 1) == 1) rd = cr_add(rd, cr_from_bits(0x3ca0000000000000ULL));
    if ((hi & 0x3ff) != 0 || sticky) rd = cr_add(rd, cr_from_bits(0x3c90000000000000ULL));
    if (sgn != 0) rd = -rd;
    r = cr_add(r, rd);
    if (ex > 1023) {
        if (ex == 1024) return cr_mul(cr_mul(r, 2.0), cr_from_bits(0x7fe0000000000000ULL));
        return cr_mul(cr_dbl_max(), cr_dbl_max());
    }
    return cr_mul(r, cr_from_bits((((cr_u64)(ex + 1023)) & 0x7ff) << 52));
}

__device__ __forceinline__ double cr_pd_to_f64(CrDint a) { return cr_pd_round_to_f64(a.hi, a.lo != 0, a.ex, a.sgn); }

#endif
