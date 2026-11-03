// Tensor-core primitives of the implicit-GEMM conv2d in
// conv_implicit_gemm.cuh: the shared-memory swizzle, ldmatrix, and one
// dtype trait per half type that carries the mma.sync instruction and the
// F32 pack and unpack.
//
// mma.sync shapes:
//   sm_80+       m16n8k16, F16 and BF16
//   sm_75, F16   two m16n8k8 per k16 step
// BF16 kernels are instantiated only for sm_80+. The host never routes a
// device below sm_75 here: sm_70 has no ldmatrix.
//
// Fragment layouts are the PTX ISA ones. With g = lane / 4, t = lane % 4:
//   A (16x16, row):  a0 (g, 2t..2t+1)   a1 (g+8, 2t..)
//                    a2 (g, 2t+8..)     a3 (g+8, 2t+8..)
//   B (16x8, col):   b0 (k 2t..2t+1, n g)   b1 (k 2t+8.., n g)
//   C (16x8, F32):   c0 c1 (g, 2t..2t+1)    c2 c3 (g+8, 2t..2t+1)

#ifndef NUMR_CONV_IMPLICIT_GEMM_MMA_CUH
#define NUMR_CONV_IMPLICIT_GEMM_MMA_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

/* 16-byte chunk of logical chunk `chunk` in row `row` of a stage buffer.
   A stage row is 32 halves: 64 bytes, 4 chunks, two rows per 128-byte bank
   window. The XOR spreads the 8 rows one ldmatrix phase reads, and the 8
   chunks one STS.128 phase writes, over 8 distinct 16-byte bank groups. */
__device__ __forceinline__ int conv_stage_swizzle(int row, int chunk) {
    return chunk ^ ((row >> 1) & 3);
}

/* Shared-window address of a generic pointer into shared memory. */
__device__ __forceinline__ unsigned int conv_smem_addr(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

/* Four 8x8 b16 matrices. Lane l supplies the start of row l % 8 of matrix
   l / 8. Volatile: the asm declares no memory operand, so this keeps it
   from moving across the barriers that order the shared-memory stores. */
__device__ __forceinline__ void conv_ldmatrix_x4(unsigned int (&d)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(addr));
}

template <typename T> struct ConvHalf;

template <> struct ConvHalf<__half> {
    using T2 = __half2;

    static __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
    static __device__ __forceinline__ float2 to_float2(__half2 v) { return __half22float2(v); }
    /* Round to nearest even, element by element, as __float2half does. */
    static __device__ __forceinline__ __half2 pack(float lo, float hi) {
        return __floats2half2_rn(lo, hi);
    }
    static __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }

    /* c += A (16x16) * B (16x8) in F32. */
    static __device__ __forceinline__ void mma_k16(float (&c)[4], const unsigned int (&a)[4],
                                                   unsigned int b0, unsigned int b1) {
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
        /* k 0-7 is {a0, a1} x b0, k 8-15 is {a2, a3} x b1. */
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(b0));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
    }
};

#if __CUDA_ARCH__ >= 800
template <> struct ConvHalf<__nv_bfloat16> {
    using T2 = __nv_bfloat162;

    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ float2 to_float2(__nv_bfloat162 v) {
        return __bfloat1622float2(v);
    }
    /* Round to nearest even, element by element, as __float2bfloat16 does. */
    static __device__ __forceinline__ __nv_bfloat162 pack(float lo, float hi) {
        return __floats2bfloat162_rn(lo, hi);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float v) {
        return __float2bfloat16(v);
    }

    static __device__ __forceinline__ void mma_k16(float (&c)[4], const unsigned int (&a)[4],
                                                   unsigned int b0, unsigned int b1) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};
#endif  // __CUDA_ARCH__ >= 800

#endif  // NUMR_CONV_IMPLICIT_GEMM_MMA_CUH
