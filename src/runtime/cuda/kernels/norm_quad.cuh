// Four consecutive row elements moved as one packed access, per norm dtype.
//
// NormQuad<T, Acc>::load reads p[0..4) with a single packed load and widens
// each element to Acc; store narrows four Acc values and writes them with a
// single packed store. The widening and narrowing match AccumTraits<T, Acc>
// element for element, so a kernel that switches between packed and scalar
// access produces the same bits.
//
// A packed access needs `p` aligned to the packed type: 16 bytes for f32 and
// f64, 8 for f16 and bf16, 4 for FP8. The launcher checks the base pointers
// and `hidden_size % 4` before picking a packed kernel.

#ifndef NUMR_NORM_QUAD_CUH
#define NUMR_NORM_QUAD_CUH

#include "norm_common.cuh"

template <typename T, typename Acc> struct NormQuad;

template <> struct NormQuad<float, float> {
    static __device__ __forceinline__ void load(const float* p, float v[4]) {
        float4 q = *reinterpret_cast<const float4*>(p);
        v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
    }
    static __device__ __forceinline__ void store(float* p, const float v[4]) {
        *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    }
};

template <> struct NormQuad<double, double> {
    static __device__ __forceinline__ void load(const double* p, double v[4]) {
        double2 lo = *reinterpret_cast<const double2*>(p);
        double2 hi = *reinterpret_cast<const double2*>(p + 2);
        v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
    }
    static __device__ __forceinline__ void store(double* p, const double v[4]) {
        *reinterpret_cast<double2*>(p) = make_double2(v[0], v[1]);
        *reinterpret_cast<double2*>(p + 2) = make_double2(v[2], v[3]);
    }
};

template <> struct NormQuad<__half, float> {
    static __device__ __forceinline__ void load(const __half* p, float v[4]) {
        uint2 raw = *reinterpret_cast<const uint2*>(p);
        __half2 lo = *reinterpret_cast<const __half2*>(&raw.x);
        __half2 hi = *reinterpret_cast<const __half2*>(&raw.y);
        v[0] = __low2float(lo); v[1] = __high2float(lo);
        v[2] = __low2float(hi); v[3] = __high2float(hi);
    }
    static __device__ __forceinline__ void store(__half* p, const float v[4]) {
        __half2 lo = __halves2half2(__float2half(v[0]), __float2half(v[1]));
        __half2 hi = __halves2half2(__float2half(v[2]), __float2half(v[3]));
        uint2 raw;
        raw.x = *reinterpret_cast<const unsigned int*>(&lo);
        raw.y = *reinterpret_cast<const unsigned int*>(&hi);
        *reinterpret_cast<uint2*>(p) = raw;
    }
};

template <> struct NormQuad<__nv_bfloat16, float> {
    static __device__ __forceinline__ void load(const __nv_bfloat16* p, float v[4]) {
        uint2 raw = *reinterpret_cast<const uint2*>(p);
        __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        v[0] = __low2float(lo); v[1] = __high2float(lo);
        v[2] = __low2float(hi); v[3] = __high2float(hi);
    }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, const float v[4]) {
        __nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(v[0]), __float2bfloat16(v[1]));
        __nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(v[2]), __float2bfloat16(v[3]));
        uint2 raw;
        raw.x = *reinterpret_cast<const unsigned int*>(&lo);
        raw.y = *reinterpret_cast<const unsigned int*>(&hi);
        *reinterpret_cast<uint2*>(p) = raw;
    }
};

template <> struct NormQuad<numr_fp8_e4m3, float> {
    static __device__ __forceinline__ void load(const numr_fp8_e4m3* p, float v[4]) {
        unsigned int raw = *reinterpret_cast<const unsigned int*>(p);
        v[0] = fp8_e4m3_to_f32((uint8_t)(raw & 0xFFu));
        v[1] = fp8_e4m3_to_f32((uint8_t)((raw >> 8) & 0xFFu));
        v[2] = fp8_e4m3_to_f32((uint8_t)((raw >> 16) & 0xFFu));
        v[3] = fp8_e4m3_to_f32((uint8_t)(raw >> 24));
    }
    static __device__ __forceinline__ void store(numr_fp8_e4m3* p, const float v[4]) {
        unsigned int raw = (unsigned int)f32_to_fp8_e4m3(v[0])
            | ((unsigned int)f32_to_fp8_e4m3(v[1]) << 8)
            | ((unsigned int)f32_to_fp8_e4m3(v[2]) << 16)
            | ((unsigned int)f32_to_fp8_e4m3(v[3]) << 24);
        *reinterpret_cast<unsigned int*>(p) = raw;
    }
};

template <> struct NormQuad<numr_fp8_e5m2, float> {
    static __device__ __forceinline__ void load(const numr_fp8_e5m2* p, float v[4]) {
        unsigned int raw = *reinterpret_cast<const unsigned int*>(p);
        v[0] = fp8_e5m2_to_f32((uint8_t)(raw & 0xFFu));
        v[1] = fp8_e5m2_to_f32((uint8_t)((raw >> 8) & 0xFFu));
        v[2] = fp8_e5m2_to_f32((uint8_t)((raw >> 16) & 0xFFu));
        v[3] = fp8_e5m2_to_f32((uint8_t)(raw >> 24));
    }
    static __device__ __forceinline__ void store(numr_fp8_e5m2* p, const float v[4]) {
        unsigned int raw = (unsigned int)f32_to_fp8_e5m2(v[0])
            | ((unsigned int)f32_to_fp8_e5m2(v[1]) << 8)
            | ((unsigned int)f32_to_fp8_e5m2(v[2]) << 16)
            | ((unsigned int)f32_to_fp8_e5m2(v[3]) << 24);
        *reinterpret_cast<unsigned int*>(p) = raw;
    }
};

#endif // NUMR_NORM_QUAD_CUH
