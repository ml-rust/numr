// Normalized fast Walsh-Hadamard transform CUDA kernels.
//
// Dtypes: f32, f64, f16, bf16. This file is PTX module "fwht"
// (kernel_names::FWHT_MODULE).
//
// Kernel naming, matching the names the Rust launcher builds in
// src/runtime/cuda/kernels/fwht.rs from dtype_suffix() in loader.rs:
//   fwht_{suffix}   one block per (row, segment) pair
//
// One CUDA block owns one `block_size` segment of one row. The segment is
// loaded into dynamic shared memory in the accumulator type, with `signs[col]`
// and the `1/sqrt(block_size)` scale folded into the load, then the Sylvester
// butterfly runs in place with a barrier per stage. Every thread owns
// `block_size / blockDim.x` elements, a runtime count, so one kernel covers
// every power-of-two segment up to the shared-memory budget.
//
// Accumulator per storage type, matching the CPU kernel
// (`runtime/cpu/kernels/fwht.rs`):
//
//   f32        float
//   f64        double
//   f16, bf16  float - the transform's cross-term cancellation is lost when
//              the running sums stay in a 10- or 7-bit mantissa.
//
// The launcher sizes dynamic shared memory as `block_size * sizeof(A)` and
// caps `block_size` so that stays within 48 KB (12288 for a float
// accumulator, 6144 for double).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Storage policies
// ============================================================================
// `S` is the element type in the kernel signature, `A` the accumulator.
// `load`/`store` are the only places a conversion happens.

struct FwhtF32 {
    typedef float S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct FwhtF64 {
    typedef double S;
    typedef double A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct FwhtF16 {
    typedef __half S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return __half2float(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = __float2half(v); }
};

struct FwhtBF16 {
    typedef __nv_bfloat16 S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return __bfloat162float(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = __float2bfloat16(v); }
};

// `1/sqrt(n)` at the accumulator's own precision.
__device__ __forceinline__ float numr_fwht_scale(float n) { return 1.0f / sqrtf(n); }
__device__ __forceinline__ double numr_fwht_scale(double n) { return 1.0 / sqrt(n); }

// ============================================================================
// Shared-memory butterfly
// ============================================================================

// Dynamic shared memory, reinterpreted as the accumulator type. One
// declaration for the whole translation unit: every instantiation aliases
// the same block-local buffer, and 16-byte alignment covers `double`.
extern __shared__ __align__(16) unsigned char numr_fwht_smem[];

// One (row, segment) pair per block. `signs` is null when absent.
template<typename P>
__device__ void fwht_impl(
    const typename P::S* __restrict__ input,
    typename P::S* __restrict__ output,
    const typename P::S* __restrict__ signs,
    unsigned int rows,
    unsigned int last_dim,
    unsigned int block_size,
    unsigned int n_segments
) {
    typedef typename P::A A;
    A* s = reinterpret_cast<A*>(numr_fwht_smem);

    const unsigned int row = blockIdx.x / n_segments;
    const unsigned int seg = blockIdx.x % n_segments;
    if (row >= rows) return;

    // Column of this segment's first element within the row.
    const unsigned int col0 = seg * block_size;
    const unsigned int base = row * last_dim + col0;
    const A scale = numr_fwht_scale((A)block_size);

    // Load, folding in the sign and the normalization.
    for (unsigned int i = threadIdx.x; i < block_size; i += blockDim.x) {
        A v = P::load(input, base + i) * scale;
        if (signs != nullptr) {
            v *= P::load(signs, col0 + i);
        }
        s[i] = v;
    }
    __syncthreads();

    // Sylvester butterfly: the low element of a pair takes u + v, the high
    // one u - v. Pair `p` at stage `h` sits at `i = (p / h) * 2h + (p % h)`.
    const unsigned int half = block_size / 2;
    for (unsigned int h = 1; h < block_size; h <<= 1) {
        for (unsigned int p = threadIdx.x; p < half; p += blockDim.x) {
            const unsigned int i = (p / h) * 2 * h + (p % h);
            const unsigned int j = i + h;
            const A u = s[i];
            const A v = s[j];
            s[i] = u + v;
            s[j] = u - v;
        }
        __syncthreads();
    }

    for (unsigned int i = threadIdx.x; i < block_size; i += blockDim.x) {
        P::store(output, base + i, s[i]);
    }
}

// ============================================================================
// Entry points
// ============================================================================

// One dtype's kernel. `P` is the storage policy, `S` the element type in the
// signature, `SUF` the kernel-name suffix.
#define NUMR_FWHT_ROW(P, S, SUF)                                                \
    __global__ void fwht_##SUF(                                                 \
        const S* in, S* out, const S* signs, unsigned int rows,                 \
        unsigned int last_dim, unsigned int block_size, unsigned int n_segments \
    ) { fwht_impl<P>(in, out, signs, rows, last_dim, block_size, n_segments); }

extern "C" {

NUMR_FWHT_ROW(FwhtF32, float, f32)
NUMR_FWHT_ROW(FwhtF64, double, f64)
NUMR_FWHT_ROW(FwhtF16, __half, f16)
NUMR_FWHT_ROW(FwhtBF16, __nv_bfloat16, bf16)

} // extern "C"
