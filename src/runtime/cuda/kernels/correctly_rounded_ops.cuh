// Rounding primitives and double-double steps of the correctly rounded kernels.
//
// Mirrors src/algorithm/correctly_rounded/double_double.rs. Every double operation is a
// round-to-nearest intrinsic, which nvcc never contracts or reassociates. The file
// compiles with --fmad=false, so a bare operator cannot contract either.
//
// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/{exp/exp.c,log/log.c,log1p/log1p.c,acos/acos.c}.
// Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.
// Copyright (c) 2022-2026 INRIA and CERN. Authors: Paul Zimmermann and Tom Hubrecht.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef NUMR_CORRECTLY_ROUNDED_OPS_CUH
#define NUMR_CORRECTLY_ROUNDED_OPS_CUH

#include "correctly_rounded_tables.cuh"

typedef unsigned long long cr_u64;
typedef long long cr_i64;

#define CR_CANONICAL_NAN_BITS 0x7ff8000000000000ULL
#define CR_INF_BITS 0x7ff0000000000000ULL
#define CR_NEG_INF_BITS 0xfff0000000000000ULL
#define CR_FRACTION_MASK 0x000fffffffffffffULL
#define CR_ABS_MASK 0x7fffffffffffffffULL
#define CR_LEN(table) (sizeof(table) / sizeof((table)[0]))

__device__ __forceinline__ cr_u64 cr_bits(double x) { return (cr_u64)__double_as_longlong(x); }
__device__ __forceinline__ double cr_from_bits(cr_u64 b) { return __longlong_as_double((cr_i64)b); }
__device__ __forceinline__ double cr_nan() { return cr_from_bits(CR_CANONICAL_NAN_BITS); }
__device__ __forceinline__ double cr_inf() { return cr_from_bits(CR_INF_BITS); }
__device__ __forceinline__ double cr_neg_inf() { return cr_from_bits(CR_NEG_INF_BITS); }
__device__ __forceinline__ double cr_dbl_max() { return cr_from_bits(0x7fefffffffffffffULL); }
__device__ __forceinline__ double cr_abs(double x) { return cr_from_bits(cr_bits(x) & CR_ABS_MASK); }
__device__ __forceinline__ bool cr_is_nan(double x) { return (cr_bits(x) & CR_ABS_MASK) > CR_INF_BITS; }

__device__ __forceinline__ double cr_add(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double cr_sub(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ double cr_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double cr_fma(double a, double b, double c) { return __fma_rn(a, b, c); }
__device__ __forceinline__ double cr_div(double a, double b) { return __ddiv_rn(a, b); }
__device__ __forceinline__ double cr_sqrt(double a) { return __dsqrt_rn(a); }
// The magnitude of x with the sign bit of s, as Rust `f64::copysign`.
__device__ __forceinline__ double cr_copysign(double x, double s) {
    return cr_from_bits((cr_bits(x) & CR_ABS_MASK) | (cr_bits(s) & ~CR_ABS_MASK));
}

// Integer to double rounds to nearest, as Rust `as f64`. Double to integer is used on
// integral values only, so truncation is exact.
__device__ __forceinline__ double cr_i2d(cr_i64 v) { return __ll2double_rn(v); }
__device__ __forceinline__ cr_i64 cr_d2i(double x) { return __double2ll_rz(x); }

// Rust `leading_zeros` and `trailing_zeros` on u64: 64 for zero.
__device__ __forceinline__ int cr_clz(cr_u64 x) { return __clzll((cr_i64)x); }
__device__ __forceinline__ int cr_ctz(cr_u64 x) { return x == 0 ? 64 : __ffsll((cr_i64)x) - 1; }

// Table entry `i` through the read-only cache.
__device__ __forceinline__ cr_u64 cr_word(const cr_u64* table, unsigned int i) { return __ldg(table + i); }
__device__ __forceinline__ double cr_tab(const cr_u64* table, unsigned int i) { return cr_from_bits(cr_word(table, i)); }

struct CrDD {
    double h;
    double l;
};

__device__ __forceinline__ CrDD cr_dd(double h, double l) {
    CrDD r;
    r.h = h;
    r.l = l;
    return r;
}

// (s, e) with s + e == a + b exactly. Needs |a| >= |b| or a == 0.
__device__ __forceinline__ CrDD cr_fast_two_sum(double a, double b) {
    const double s = cr_add(a, b);
    const double z = cr_sub(s, a);
    return cr_dd(s, cr_sub(b, z));
}

// (s, e) with s + e == a + b exactly, low part (b - d) + (a + (d - s)).
__device__ __forceinline__ CrDD cr_two_sum(double a, double b) {
    const double s = cr_add(a, b);
    const double d = cr_sub(s, a);
    return cr_dd(s, cr_add(cr_sub(b, d), cr_add(a, cr_sub(d, s))));
}

__device__ __forceinline__ CrDD cr_two_prod(double a, double b) {
    const double p = cr_mul(a, b);
    return cr_dd(p, cr_fma(a, b, -p));
}

__device__ __forceinline__ CrDD cr_fast_sum(double xh, double xl, double yh, double yl) {
    const CrDD s = cr_fast_two_sum(xh, yh);
    return cr_dd(s.h, cr_add(cr_add(xl, yl), s.l));
}

__device__ __forceinline__ CrDD cr_sum(double xh, double xl, double ch, double cl) {
    const CrDD s = cr_two_sum(xh, ch);
    return cr_dd(s.h, cr_add(cr_add(xl, cl), s.l));
}

// (xh + xl) * (ch + cl) without xl * cl: low part (cl * xh + ch * xl) + fma(ch, xh, -ch * xh).
__device__ __forceinline__ CrDD cr_mul_dd(double xh, double xl, double ch, double cl) {
    const CrDD p = cr_two_prod(ch, xh);
    return cr_dd(p.h, cr_add(cr_add(cr_mul(cl, xh), cr_mul(ch, xl)), p.l));
}

__device__ __forceinline__ CrDD cr_mul_d_dd(double x, double ch, double cl) {
    const CrDD p = cr_two_prod(ch, x);
    return cr_dd(p.h, cr_add(cr_mul(cl, x), p.l));
}

// (ah + al) * (bh + bl) + (ch + cl), Algorithm 4 of Jeannerod, Joldes, Louvet and Muller.
__device__ __forceinline__ CrDD cr_fma_dw(double ah, double al, double bh, double bl, double ch, double cl) {
    const double dh = cr_fma(ah, bh, ch);
    const double t = cr_sub(ch, dh);
    const double e = cr_fma(ah, bh, t);
    const double f = cr_add(e, cl);
    const double g = cr_add(cr_mul(ah, bl), f);
    return cr_dd(dh, cr_add(cr_mul(al, bh), g));
}

// Horner evaluation in double-double of the n (hi, lo) pairs of c, lowest degree first,
// at xh + xl. `low` joins the leading coefficient first. CORE-MATH's polydd.
__device__ __forceinline__ CrDD cr_horner_dd(double xh, double xl, const cr_u64* c, int n, double low) {
    const CrDD s = cr_fast_two_sum(cr_tab(c, 2 * (n - 1)), low);
    CrDD r = cr_dd(s.h, cr_add(cr_tab(c, 2 * (n - 1) + 1), s.l));
    for (int i = n - 2; i >= 0; --i) {
        r = cr_mul_dd(xh, xl, r.h, r.l);
        r = cr_fast_sum(cr_tab(c, 2 * i), cr_tab(c, 2 * i + 1), r.h, r.l);
    }
    return r;
}

// Round |y| < 2^51 to the nearest integer, ties to even, with 1.5 * 2^52.
__device__ __forceinline__ double cr_round_ties_even(double y) {
    const double magic = 6755399441055744.0;
    return cr_sub(cr_add(y, magic), magic);
}

// Add e to the biased exponent field of x. Exact while the result stays normal.
__device__ __forceinline__ double cr_add_exponent(double x, cr_i64 e) {
    return cr_from_bits(cr_bits(x) + ((cr_u64)e << 52));
}

#endif
