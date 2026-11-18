// Products of CrQint values for the extended pow path. Mirrors
// src/algorithm/correctly_rounded/qint_mul.rs: mul_ab uses the upper a limbs of the
// first operand and the upper b limbs of the second, and drops the same partial
// products as CORE-MATH.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/qint.h. Copyright (c) 2022, 2023 CERN and Inria.
// Authors: Tom Hubrecht and Paul Zimmermann.
// MIT licence: see correctly_rounded_ops.cuh for the full notice.

#ifndef NUMR_CORRECTLY_ROUNDED_QINT_MUL_CUH
#define NUMR_CORRECTLY_ROUNDED_QINT_MUL_CUH

#include "correctly_rounded_qint.cuh"

// x + y as (sum, carry) with the carry as a 128-bit 0 or 1, as Rust `carry(overflowing_add)`.
__device__ __forceinline__ CrU128 cr_qm_add(CrU128 x, CrU128 y, CrU128* carry) {
    bool c;
    const CrU128 s = cr_u128_add_carry(x, y, &c);
    *carry = cr_u128_from(c ? 1 : 0);
    return s;
}

__device__ __forceinline__ CrU128 cr_qm_high(CrU128 x) { return cr_u128_from(x.hi); }

// Normalize the upper 256 bits t6 : low of a product so the top bit is set.
__device__ __forceinline__ CrQint cr_qm_finish(CrU128 t6, CrU128 low, CrQint a, CrQint b) {
    const cr_i64 ex = (t6.hi >> 63) == 0 ? 1 : 0;
    CrU128 rh = t6, rl = low;
    if (ex == 1) {
        rh = cr_u128_or(cr_u128_shl(t6, 1), cr_u128_shr(low, 127));
        rl = cr_u128_shl(low, 1);
    }
    return cr_qint(rh, rl, a.ex + b.ex + 1 - ex, a.sgn ^ b.sgn);
}

// Sum the terms of degree 4 and 5 above t4, then assemble the top 256 bits.
__device__ __forceinline__ CrQint cr_qm_assemble(CrU128 r33, CrU128 t5, CrU128 c5, CrU128 t4, CrU128 c4, CrQint a,
                                                 CrQint b) {
    const CrU128 mid = cr_u128(c5.lo, t5.hi);
    const CrU128 t6 = cr_u128_add(cr_u128_add(r33, mid), c4);
    return cr_qm_finish(t6, cr_u128(t5.lo, t4.lo), a, b);
}

// Degree 4 and 5 sums shared by mul and mul_33; t3 holds the degree 3 terms.
__device__ __forceinline__ CrQint cr_qm_upper(CrQint a, CrQint b, CrU128 t3) {
    const CrU128 r33 = cr_u128_mul(a.rh.hi, b.rh.hi);
    const CrU128 r32 = cr_u128_mul(a.rh.hi, b.rh.lo);
    const CrU128 r23 = cr_u128_mul(a.rh.lo, b.rh.hi);
    const CrU128 r31 = cr_u128_mul(a.rh.hi, b.rl.hi);
    const CrU128 r13 = cr_u128_mul(a.rl.hi, b.rh.hi);
    const CrU128 r22 = cr_u128_mul(a.rh.lo, b.rh.lo);
    CrU128 c4, c5, c;
    CrU128 t4 = cr_qm_add(r22, t3, &c4);
    t4 = cr_qm_add(r13, t4, &c);
    c4 = cr_u128_add(c4, c);
    t4 = cr_qm_add(r31, t4, &c);
    c4 = cr_u128_add(c4, c);
    CrU128 t5 = cr_qm_add(r23, cr_qm_high(t4), &c5);
    t5 = cr_qm_add(r32, t5, &c);
    c5 = cr_u128_add(c5, c);
    return cr_qm_assemble(r33, t5, c5, t4, c4, a, b);
}

// a * b with error below 14 ulps.
__device__ __noinline__ CrQint cr_qint_mul(CrQint a, CrQint b) {
    const CrU128 r30 = cr_u128_mul(a.rh.hi, b.rl.lo);
    const CrU128 r03 = cr_u128_mul(a.rl.lo, b.rh.hi);
    const CrU128 r21 = cr_u128_mul(a.rh.lo, b.rl.hi);
    const CrU128 r12 = cr_u128_mul(a.rl.hi, b.rh.lo);
    CrU128 t3 = cr_u128_add(cr_qm_high(r12), cr_qm_high(r21));
    t3 = cr_u128_add(t3, cr_qm_high(r03));
    t3 = cr_u128_add(t3, cr_qm_high(r30));
    return cr_qm_upper(a, b, t3);
}

// a * b on the upper three limbs of each, with error below 6 ulps.
__device__ __noinline__ CrQint cr_qint_mul_33(CrQint a, CrQint b) {
    const CrU128 r21 = cr_u128_mul(a.rh.lo, b.rl.hi);
    const CrU128 r12 = cr_u128_mul(a.rl.hi, b.rh.lo);
    return cr_qm_upper(a, b, cr_u128_add(cr_qm_high(r12), cr_qm_high(r21)));
}

// a * b on all limbs of a and the top limb of b, with error below 2 ulps.
__device__ __noinline__ CrQint cr_qint_mul_41(CrQint a, CrQint b) {
    const CrU128 r33 = cr_u128_mul(a.rh.hi, b.rh.hi);
    const CrU128 r23 = cr_u128_mul(a.rh.lo, b.rh.hi);
    const CrU128 r13 = cr_u128_mul(a.rl.hi, b.rh.hi);
    const CrU128 r03 = cr_u128_mul(a.rl.lo, b.rh.hi);
    CrU128 c4, c5;
    const CrU128 t4 = cr_qm_add(r13, cr_qm_high(r03), &c4);
    const CrU128 t5 = cr_qm_add(r23, cr_qm_high(t4), &c5);
    return cr_qm_assemble(r33, t5, c5, t4, c4, a, b);
}

// The exact product of the upper three limbs of a and the top limb of b.
__device__ __noinline__ CrQint cr_qint_mul_31(CrQint a, CrQint b) {
    const CrU128 r33 = cr_u128_mul(a.rh.hi, b.rh.hi);
    const CrU128 r23 = cr_u128_mul(a.rh.lo, b.rh.hi);
    const CrU128 t4 = cr_u128_mul(a.rl.hi, b.rh.hi);
    CrU128 c5;
    const CrU128 t5 = cr_qm_add(r23, cr_qm_high(t4), &c5);
    return cr_qm_assemble(r33, t5, c5, t4, cr_u128(0, 0), a, b);
}

// The exact product of the upper two limbs of a and b.
__device__ __noinline__ CrQint cr_qint_mul_22(CrQint a, CrQint b) {
    const CrU128 r33 = cr_u128_mul(a.rh.hi, b.rh.hi);
    const CrU128 r32 = cr_u128_mul(a.rh.hi, b.rh.lo);
    const CrU128 r23 = cr_u128_mul(a.rh.lo, b.rh.hi);
    const CrU128 t4 = cr_u128_mul(a.rh.lo, b.rh.lo);
    CrU128 c5, c;
    CrU128 t5 = cr_qm_add(r23, cr_qm_high(t4), &c5);
    t5 = cr_qm_add(r32, t5, &c);
    c5 = cr_u128_add(c5, c);
    return cr_qm_assemble(r33, t5, c5, t4, cr_u128(0, 0), a, b);
}

// The exact product of the upper two limbs of a and the top limb of b.
__device__ __noinline__ CrQint cr_qint_mul_21(CrQint a, CrQint b) {
    const CrU128 r33 = cr_u128_mul(a.rh.hi, b.rh.hi);
    const CrU128 r23 = cr_u128_mul(a.rh.lo, b.rh.hi);
    return cr_qm_finish(cr_u128_add(r33, cr_qm_high(r23)), cr_u128(r23.lo, 0), a, b);
}

// The exact product of the top limbs of a and b.
__device__ __forceinline__ CrQint cr_qint_mul_11(CrQint a, CrQint b) {
    return cr_qm_finish(cr_u128_mul(a.rh.hi, b.rh.hi), cr_u128(0, 0), a, b);
}

// b * a for an integer b, with error below 2 ulps.
__device__ __noinline__ CrQint cr_qint_mul_int(cr_i64 b, CrQint a) {
    if (b == 0) return cr_qint_zero();
    const cr_u64 sgn = (b < 0 ? 1 : 0) ^ a.sgn;
    cr_u64 c = b < 0 ? (cr_u64)0 - (cr_u64)b : (cr_u64)b;
    if (c == 1) {
        a.sgn = sgn;
        return a;
    }
    // Scale c so that 2^63 <= c < 2^64.
    const int k = cr_clz(c);
    c <<= k;
    cr_i64 ex = a.ex + 64 - k;
    CrU128 t3 = cr_u128_mul(a.rh.hi, c);
    CrU128 t2 = cr_u128_mul(a.rh.lo, c);
    CrU128 t1 = cr_u128_mul(a.rl.hi, c);
    const CrU128 t0 = cr_u128_mul(a.rl.lo, c);
    CrU128 cy;
    t1 = cr_qm_add(cr_qm_high(t0), t1, &cy);
    t2 = cr_qm_add(cr_u128(cy.lo, t1.hi), t2, &cy);
    t3 = cr_u128_add(t3, cr_u128(cy.lo, t2.hi));
    const CrU128 low = cr_u128(t2.lo, t1.lo);
    CrU128 rh = t3, rl = low;
    // The top bit of t3 is clear in at most one position, as a and c are normalized.
    if ((t3.hi >> 63) == 0) {
        ex -= 1;
        rh = cr_u128_or(cr_u128_shl(t3, 1), cr_u128_shr(low, 127));
        rl = cr_u128_shl(low, 1);
    }
    return cr_qint(rh, rl, ex, sgn);
}

#endif
