// Unsigned 128-bit integers from two 64-bit words, for the dint and qint kernels.
//
// Each operation equals the Rust `u128` operation the host kernel uses: wrapping add,
// sub and shift, `overflowing_add`, `leading_zeros`, and the exact 64 x 64 product.
// Shifts by 128 or more give zero; the host kernels never shift that far.

#ifndef NUMR_CORRECTLY_ROUNDED_U128_CUH
#define NUMR_CORRECTLY_ROUNDED_U128_CUH

#include "correctly_rounded_ops.cuh"

struct CrU128 {
    cr_u64 hi;
    cr_u64 lo;
};

__device__ __forceinline__ CrU128 cr_u128(cr_u64 hi, cr_u64 lo) {
    CrU128 r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

__device__ __forceinline__ CrU128 cr_u128_from(cr_u64 lo) { return cr_u128(0, lo); }

__device__ __forceinline__ bool cr_u128_is_zero(CrU128 a) { return (a.hi | a.lo) == 0; }

// The exact product of two 64-bit words.
__device__ __forceinline__ CrU128 cr_u128_mul(cr_u64 a, cr_u64 b) { return cr_u128(__umul64hi(a, b), a * b); }

// a + b modulo 2^128, with the carry out in *carry.
__device__ __forceinline__ CrU128 cr_u128_add_carry(CrU128 a, CrU128 b, bool* carry) {
    const cr_u64 lo = a.lo + b.lo;
    const cr_u64 c1 = lo < a.lo ? 1 : 0;
    const cr_u64 t = a.hi + b.hi;
    const bool c2 = t < a.hi;
    const cr_u64 hi = t + c1;
    const bool c3 = hi < t;
    *carry = c2 || c3;
    return cr_u128(hi, lo);
}

__device__ __forceinline__ CrU128 cr_u128_add(CrU128 a, CrU128 b) {
    bool carry;
    return cr_u128_add_carry(a, b, &carry);
}

// a - b modulo 2^128, with the borrow out in *borrow.
__device__ __forceinline__ CrU128 cr_u128_sub_borrow(CrU128 a, CrU128 b, bool* borrow) {
    const cr_u64 lo = a.lo - b.lo;
    const cr_u64 b1 = a.lo < b.lo ? 1 : 0;
    const cr_u64 t = a.hi - b.hi;
    const bool b2 = a.hi < b.hi;
    const bool b3 = t < b1;
    *borrow = b2 || b3;
    return cr_u128(t - b1, lo);
}

__device__ __forceinline__ CrU128 cr_u128_sub(CrU128 a, CrU128 b) {
    bool borrow;
    return cr_u128_sub_borrow(a, b, &borrow);
}

__device__ __forceinline__ CrU128 cr_u128_shl(CrU128 a, int s) {
    if (s == 0) return a;
    if (s < 64) return cr_u128((a.hi << s) | (a.lo >> (64 - s)), a.lo << s);
    if (s < 128) return cr_u128(a.lo << (s - 64), 0);
    return cr_u128(0, 0);
}

__device__ __forceinline__ CrU128 cr_u128_shr(CrU128 a, int s) {
    if (s == 0) return a;
    if (s < 64) return cr_u128(a.hi >> s, (a.lo >> s) | (a.hi << (64 - s)));
    if (s < 128) return cr_u128(0, a.hi >> (s - 64));
    return cr_u128(0, 0);
}

__device__ __forceinline__ CrU128 cr_u128_or(CrU128 a, CrU128 b) { return cr_u128(a.hi | b.hi, a.lo | b.lo); }

// Leading zero bits, 128 for zero.
__device__ __forceinline__ int cr_u128_clz(CrU128 a) { return a.hi != 0 ? cr_clz(a.hi) : 64 + cr_clz(a.lo); }

// -1, 0 or 1 as a is below, equal to or above b.
__device__ __forceinline__ int cr_u128_cmp(CrU128 a, CrU128 b) {
    if (a.hi != b.hi) return a.hi < b.hi ? -1 : 1;
    if (a.lo != b.lo) return a.lo < b.lo ? -1 : 1;
    return 0;
}

#endif
