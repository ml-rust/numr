// 128-bit significand arithmetic of the accurate log path.
//
// Mirrors src/algorithm/correctly_rounded/dint.rs. A CrDint is
// (-1)^sgn * (hi + lo / 2^64) * 2^(ex - 63) with the top bit of hi set.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log/dint.h. Copyright (c) 2022 CERN. Author: Tom Hubrecht.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_DINT_CUH
#define NUMR_CORRECTLY_ROUNDED_DINT_CUH

#include "correctly_rounded_u128.cuh"

struct CrDint {
    cr_u64 hi;
    cr_u64 lo;
    cr_i64 ex;
    cr_u64 sgn;
};

__device__ __forceinline__ CrDint cr_dint(cr_u64 hi, cr_u64 lo, cr_i64 ex, cr_u64 sgn) {
    CrDint r;
    r.hi = hi;
    r.lo = lo;
    r.ex = ex;
    r.sgn = sgn;
    return r;
}

__device__ __forceinline__ CrDint cr_dint_parts(CrU128 c, cr_i64 ex, cr_u64 sgn) { return cr_dint(c.hi, c.lo, ex, sgn); }

__device__ __forceinline__ CrU128 cr_dint_sig(CrDint a) { return cr_u128(a.hi, a.lo); }

// Entry `index` of a table of (hi, lo, ex, sgn) words.
__device__ __forceinline__ CrDint cr_dint_table(const cr_u64* table, unsigned int index) {
    return cr_dint(cr_word(table, 4 * index), cr_word(table, 4 * index + 1), (cr_i64)cr_word(table, 4 * index + 2),
                   cr_word(table, 4 * index + 3));
}

// Magnitude order of the log dint: exponent first, then the significand.
__device__ __forceinline__ int cr_dint_cmp_magnitude(CrDint a, CrDint b) {
    if (a.ex != b.ex) return a.ex < b.ex ? -1 : 1;
    return cr_u128_cmp(cr_dint_sig(a), cr_dint_sig(b));
}

// a + b, with the smaller operand rounded to nearest when shifted.
__device__ __noinline__ CrDint cr_dint_add(CrDint a, CrDint b) {
    if ((a.hi | a.lo) == 0) return b;
    if ((b.hi | b.lo) == 0) return a;
    const int order = cr_dint_cmp_magnitude(a, b);
    if (order == 0) {
        if ((a.sgn ^ b.sgn) != 0) return cr_dint(0, 0, 0, 0);
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
    cr_i64 m_ex = a.ex;
    if (a.ex > b.ex) {
        const cr_i64 sh = a.ex - b.ex;
        if (sh <= 128) small = cr_u128_add(small, cr_u128_from(cr_u128_shr(small, (int)(sh - 1)).lo & 1));
        small = sh < 128 ? cr_u128_shr(small, (int)sh) : cr_u128(0, 0);
    }
    CrU128 c;
    if ((a.sgn ^ b.sgn) != 0) {
        c = cr_u128_sub(big, small);
    } else {
        bool overflow;
        c = cr_u128_add_carry(big, small, &overflow);
        if (overflow) {
            c = cr_u128_add(c, cr_u128_from(c.lo & 1));
            c = cr_u128_or(cr_u128(1ULL << 63, 0), cr_u128_shr(c, 1));
            m_ex += 1;
        }
    }
    if (cr_u128_is_zero(c)) return cr_dint(0, 0, (cr_i64)((cr_u64)m_ex - (64 + (cr_u64)a.ex)), a.sgn);
    const int shift = cr_u128_clz(c);
    return cr_dint_parts(cr_u128_shl(c, shift), m_ex - shift, a.sgn);
}

// a * b with 126 correct bits; lo * lo is dropped.
__device__ __noinline__ CrDint cr_dint_mul(CrDint a, CrDint b) {
    CrU128 t = cr_u128_mul(a.hi, b.hi);
    const CrU128 m1 = cr_u128_mul(a.hi, b.lo);
    const CrU128 m2 = cr_u128_mul(a.lo, b.hi);
    bool carry;
    const CrU128 m = cr_u128_add_carry(m1, m2, &carry);
    if (carry) t = cr_u128_add(t, cr_u128(1, 0));
    t = cr_u128_add(t, cr_u128_from(m.hi));
    const cr_i64 ex = (t.hi >> 63) == 0 ? 1 : 0;
    if (ex == 1) t = cr_u128_shl(t, 1);
    t = cr_u128_add(t, cr_u128_from(m.lo >> 63));
    return cr_dint_parts(t, a.ex + b.ex - ex + 1, a.sgn ^ b.sgn);
}

// b * a for a small integer b (|b| < 2^11).
__device__ __noinline__ CrDint cr_dint_mul_int(cr_i64 b, CrDint a) {
    if (b == 0) return cr_dint(0, 0, 0, 0);
    const cr_u64 c = b < 0 ? (cr_u64)0 - (cr_u64)b : (cr_u64)b;
    const cr_u64 sgn = b < 0 ? (a.sgn == 0 ? 1 : 0) : a.sgn;
    CrU128 t = cr_u128_mul(a.hi, c);
    int m = t.hi != 0 ? cr_clz(t.hi) : 64;
    t = cr_u128_shl(t, m);
    const CrU128 l = cr_u128_shr(cr_u128_shl(cr_u128_mul(a.lo, c), m - 1), 63);
    bool overflow;
    t = cr_u128_add_carry(l, t, &overflow);
    if (overflow) {
        t = cr_u128_add(t, cr_u128_from(t.lo & 1));
        t = cr_u128_or(cr_u128(1ULL << 63, 0), cr_u128_shr(t, 1));
        m -= 1;
    }
    return cr_dint_parts(t, a.ex + 64 - m, sgn);
}

// The exact value of a finite nonzero binary64 x, subnormals included.
__device__ __forceinline__ CrDint cr_dint_from_f64(double x) {
    const cr_u64 bits = cr_bits(x);
    const cr_i64 biased = (cr_i64)((bits >> 52) & 0x7ff);
    const cr_u64 m = (bits & CR_FRACTION_MASK) + (biased != 0 ? (1ULL << 52) : 0);
    const int t = cr_clz(m);
    return cr_dint(m << t, 0, biased - 0x3ff - (t > 11 ? (cr_i64)t - 12 : 0), x < 0.0 ? 1 : 0);
}

// Round to the nearest binary64. Needs a normal result whose bits past the round bit are
// not all zero, which holds for every log result.
__device__ __forceinline__ double cr_dint_to_f64(CrDint a) {
    cr_u64 r = (a.hi >> 11) | (0x3ffULL << 52);
    double rd = cr_from_bits(0x3c90000000000000ULL);
    if (((a.hi >> 10) & 1) == 1) rd = cr_add(rd, cr_from_bits(0x3ca0000000000000ULL));
    r |= a.sgn << 63;
    const double rounded = cr_add(cr_from_bits(r), a.sgn == 0 ? rd : -rd);
    const double scale = cr_from_bits((((cr_u64)(a.ex + 1023)) & 0x7ff) << 52);
    return cr_mul(rounded, scale);
}

#endif
