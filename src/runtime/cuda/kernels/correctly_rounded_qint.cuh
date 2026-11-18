// 256-bit significand arithmetic of the extended pow path. Mirrors
// src/algorithm/correctly_rounded/qint.rs. A CrQint is
// (-1)^sgn * (hh + hl / 2^64 + lh / 2^128 + ll / 2^192) * 2^(ex - 63) with the top bit
// of hh set. The products live in correctly_rounded_qint_mul.cuh.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/qint.h. Copyright (c) 2022, 2023 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_QINT_CUH
#define NUMR_CORRECTLY_ROUNDED_QINT_CUH

#include "correctly_rounded_pow_dint.cuh"

struct CrQint {
    CrU128 rh;
    CrU128 rl;
    cr_i64 ex;
    cr_u64 sgn;
};

__device__ __forceinline__ CrQint cr_qint(CrU128 rh, CrU128 rl, cr_i64 ex, cr_u64 sgn) {
    CrQint r;
    r.rh = rh;
    r.rl = rl;
    r.ex = ex;
    r.sgn = sgn;
    return r;
}

__device__ __forceinline__ CrQint cr_qint_zero() { return cr_qint(cr_u128(0, 0), cr_u128(0, 0), 0, 0); }

// Entry `index` of a table of (hh, hl, lh, ll, ex, sgn) words.
__device__ __forceinline__ CrQint cr_qint_table(const cr_u64* table, unsigned int index) {
    const unsigned int b = 6 * index;
    return cr_qint(cr_u128(cr_word(table, b), cr_word(table, b + 1)), cr_u128(cr_word(table, b + 2), cr_word(table, b + 3)),
                   (cr_i64)cr_word(table, b + 4), cr_word(table, b + 5));
}

// The value of d, with zero low limbs.
__device__ __forceinline__ CrQint cr_qint_from_dint(CrDint d) { return cr_qint(cr_u128(d.hi, d.lo), cr_u128(0, 0), d.ex, d.sgn); }

// A 128-bit table entry widened to 256 bits.
__device__ __forceinline__ CrQint cr_qint_widened(const cr_u64* table, unsigned int index) {
    return cr_qint_from_dint(cr_dint_table(table, index));
}

__device__ __forceinline__ bool cr_qint_is_zero(CrQint a) { return cr_u128_is_zero(a.rh) && cr_u128_is_zero(a.rl); }

// Magnitude order: exponent first, then the significand.
__device__ __forceinline__ int cr_qint_cmp_magnitude(CrQint a, CrQint b) {
    if (a.ex != b.ex) return a.ex < b.ex ? -1 : 1;
    const int high = cr_u128_cmp(a.rh, b.rh);
    return high != 0 ? high : cr_u128_cmp(a.rl, b.rl);
}

// (h, l) - (bh, bl) modulo 2^256.
__device__ __forceinline__ void cr_sub256(CrU128 h, CrU128 l, CrU128 bh, CrU128 bl, CrU128* ch, CrU128* cl) {
    bool borrow;
    *cl = cr_u128_sub_borrow(l, bl, &borrow);
    *ch = cr_u128_sub(cr_u128_sub(h, bh), cr_u128_from(borrow ? 1 : 0));
}

// Leading zero bits of a 256-bit value, 256 for zero.
__device__ __forceinline__ int cr_clz256(CrU128 h, CrU128 l) { return !cr_u128_is_zero(h) ? cr_u128_clz(h) : 128 + cr_u128_clz(l); }

// (h, l) >> k: zero from k >= 256 and for k < 0.
__device__ __forceinline__ void cr_shr256(CrU128 h, CrU128 l, cr_i64 k, CrU128* oh, CrU128* ol) {
    if (k == 0) {
        *oh = h;
        *ol = l;
    } else if (k > 0 && k < 128) {
        *oh = cr_u128_shr(h, (int)k);
        *ol = cr_u128_or(cr_u128_shr(l, (int)k), cr_u128_shl(h, (int)(128 - k)));
    } else if (k >= 128 && k < 256) {
        *oh = cr_u128(0, 0);
        *ol = cr_u128_shr(h, (int)(k - 128));
    } else {
        *oh = cr_u128(0, 0);
        *ol = cr_u128(0, 0);
    }
}

// (h, l) << k modulo 2^256, for 0 <= k < 256.
__device__ __forceinline__ void cr_shl256(CrU128 h, CrU128 l, cr_i64 k, CrU128* oh, CrU128* ol) {
    if (k == 0) {
        *oh = h;
        *ol = l;
    } else if (k > 0 && k < 128) {
        *oh = cr_u128_or(cr_u128_shl(h, (int)k), cr_u128_shr(l, (int)(128 - k)));
        *ol = cr_u128_shl(l, (int)k);
    } else {
        *oh = cr_u128_shl(l, (int)(k - 128));
        *ol = cr_u128(0, 0);
    }
}

// a + b with error below 2 ulps of 256 bits. Exact under Sterbenz's condition.
__device__ __noinline__ CrQint cr_qint_add(CrQint a, CrQint b) {
    if (cr_qint_is_zero(a)) return b;
    if (cr_qint_is_zero(b)) return a;
    const int order = cr_qint_cmp_magnitude(a, b);
    if (order == 0) {
        if ((a.sgn ^ b.sgn) != 0) return cr_qint_zero();
        a.ex += 1;
        return a;
    }
    if (order < 0) {
        const CrQint t = a;
        a = b;
        b = t;
    }
    // From here |a| > |b|.
    CrU128 ah = a.rh, al = a.rl, bh, bl, ch, cl;
    const cr_i64 k = a.ex - b.ex;
    cr_shr256(b.rh, b.rl, k, &bh, &bl);
    cr_i64 ex = a.ex;
    if ((a.sgn ^ b.sgn) != 0) {
        cr_sub256(ah, al, bh, bl, &ch, &cl);
        // C != 0 because |a| > |b|.
        int shift = cr_clz256(ch, cl);
        if (shift > 0) {
            cr_shl256(ah, al, shift, &ah, &al);
            // Shift b left by shift - k from its unshifted value, so no bit is lost when
            // the cancellation exceeds the alignment.
            const cr_i64 sh = (cr_i64)shift - k;
            if (sh >= 0) {
                cr_shl256(b.rh, b.rl, sh, &bh, &bl);
            } else {
                cr_shr256(b.rh, b.rl, -sh, &bh, &bl);
            }
            ex -= shift;
            cr_sub256(ah, al, bh, bl, &ch, &cl);
            shift = cr_clz256(ch, cl);
        }
        ex -= shift;
        cr_shl256(ch, cl, shift, &ch, &cl);
    } else {
        bool carry, low_carry;
        ch = cr_u128_add_carry(ah, bh, &carry);
        cl = cr_u128_add_carry(al, bl, &low_carry);
        if (low_carry) {
            ch = cr_u128_add(ch, cr_u128_from(1));
            carry = carry || cr_u128_is_zero(ch);
        }
        if (carry) {
            ex += 1;
            const CrU128 nh = cr_u128_or(cr_u128(1ULL << 63, 0), cr_u128_shr(ch, 1));
            const CrU128 nl = cr_u128_or(cr_u128_shl(ch, 127), cr_u128_shr(cl, 1));
            ch = nh;
            cl = nl;
        }
    }
    return cr_qint(ch, cl, ex, a.sgn);
}

// a + b on the upper two limbs only, with error below 2 ulps of 128 bits. The lower two
// limbs of a computed sum are zero.
__device__ __noinline__ CrQint cr_qint_add_22(CrQint a, CrQint b) {
    if (cr_u128_is_zero(a.rh)) return b;
    if (cr_u128_is_zero(b.rh)) return a;
    const int order = a.ex != b.ex ? (a.ex < b.ex ? -1 : 1) : cr_u128_cmp(a.rh, b.rh);
    if (order == 0) {
        if ((a.sgn ^ b.sgn) != 0) return cr_qint_zero();
        a.ex += 1;
        return a;
    }
    if (order < 0) {
        const CrQint t = a;
        a = b;
        b = t;
    }
    CrU128 ah = a.rh;
    const cr_u64 k = (cr_u64)(a.ex - b.ex);
    CrU128 bh = k >= 128 ? cr_u128(0, 0) : cr_u128_shr(b.rh, (int)k);
    cr_i64 ex = a.ex;
    CrU128 ch;
    if ((a.sgn ^ b.sgn) != 0) {
        ch = cr_u128_sub(ah, bh);
        int shift = cr_u128_clz(ch);
        if (shift > 0) {
            // k < 128 here: otherwise bh = 0 and ch = ah has no leading zero.
            ah = cr_u128_shl(ah, shift);
            bh = (cr_u64)shift >= k ? cr_u128_shl(b.rh, (int)((cr_u64)shift - k)) : cr_u128_shr(b.rh, (int)(k - (cr_u64)shift));
            ex -= shift;
            ch = cr_u128_sub(ah, bh);
            shift = cr_u128_clz(ch);
        }
        ex -= shift;
        ch = cr_u128_shl(ch, shift);
    } else {
        bool carry;
        ch = cr_u128_add_carry(ah, bh, &carry);
        if (carry) {
            ex += 1;
            ch = cr_u128_or(cr_u128(1ULL << 63, 0), cr_u128_shr(ch, 1));
        }
    }
    return cr_qint(ch, cr_u128(0, 0), ex, a.sgn);
}

// a rounded toward zero to an integer. Needs a.ex < 64.
__device__ __forceinline__ cr_i64 cr_qint_to_i64(CrQint a) {
    if (a.ex < 0) return 0;
    const cr_i64 r = (cr_i64)(a.rh.hi >> (63 - a.ex));
    return a.sgn != 0 ? -r : r;
}

// a rounded to nearest binary64, once in every range.
__device__ __forceinline__ double cr_qint_to_f64(CrQint a) {
    return cr_pd_round_to_f64(a.rh.hi, (a.rh.lo | a.rl.hi | a.rl.lo) != 0, a.ex, a.sgn);
}

#endif
