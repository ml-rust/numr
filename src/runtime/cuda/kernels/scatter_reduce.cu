// Scatter-with-reduction CUDA kernels.
//
// The result is bit-identical run to run, and every destination element
// combines its contributions in increasing source position, the order the CPU
// reference `scatter_reduce_kernel` in src/runtime/cpu/kernels/index/scatter.rs
// uses. An atomic per source element cannot give either property for floats:
// the order in which colliding atomics land depends on scheduling, and float
// addition and multiplication are not associative.
//
// The pipeline groups the sources by destination with a stable LSD radix sort,
// then reduces each destination's run in order:
//
//  1. scatter_reduce_keys: one thread per source element writes the flat
//     destination position (the key) and the source's own flat position (the
//     value). An out-of-range index gets key `dst_numel`, which sorts past every
//     real destination and is never reduced.
//  2. One pass per 8-bit digit of the key, each three kernels:
//     scatter_reduce_radix_hist counts each tile's digits,
//     scatter_reduce_radix_scan turns the counts into each tile's exclusive
//     offset within its digit and each digit's total, and
//     scatter_reduce_radix_scatter moves every entry to its sorted slot. Ranks
//     inside a tile follow source order, so every pass is stable and the final
//     order within one key is increasing source position.
//  3. scatter_reduce_{op}_{dtype}: one thread per destination element finds
//     its run by binary search over the sorted keys and folds it in order.
//
// Every count above is an integer, so the shared-memory atomics that build the
// histograms give one result regardless of order.
//
// Accumulators:
//
//  * FLOAT (f32, f64): the element type, as on CPU. `mean` divides the sum by
//    the number of contributions, the destination's own value counted when
//    include_self is set, and leaves an element nobody reached unchanged.
//  * INTEGER (i64, i32, i16, i8, u64, u32, u16, u8): a Numr128 accumulator.
//    The running total is an ACCUMULATOR, and this project's convention is
//    that accumulators saturate while elementwise ops wrap (see
//    src/runtime/cpu/kernels/wide_acc.rs). `mean` divides ONCE, at the end,
//    inside the 128-bit accumulator, then narrows with saturation, as
//    `scatter_reduce_int_kernel` in src/runtime/cpu/kernels/scatter_reduce_int.rs
//    does.
//
// The reduce kernels read the destination already initialised by the caller: a
// copy of `dst` when include_self is set, otherwise the reduction's identity.
//
// Launch geometry is fixed by SR_THREADS, SR_ITEMS and SR_TILE below, which
// the launcher in src/runtime/cuda/kernels/index/scatter_reduce.rs mirrors.
// This is PTX module "scatter_reduce" (kernel_names::SCATTER_REDUCE_MODULE).

#include "dtype_traits.cuh"
#include "index_ops.cuh"
#include "numr128.cuh"

// ============================================================================
// Radix sort geometry
// ============================================================================

#define SR_RADIX 256
#define SR_THREADS 256
#define SR_WARPS (SR_THREADS / 32)
#define SR_ITEMS 16
#define SR_WARP_SPAN (32 * SR_ITEMS)
#define SR_TILE (SR_WARPS * SR_WARP_SPAN)

// ============================================================================
// Keys
// ============================================================================

// Source position `i` decomposes over `src_shape` (row-major). The
// destination position replaces the coordinate on `dim` with the index value
// and recombines with `dst_strides`, so a source smaller than the destination
// on another axis lands where the CPU reference puts it.
extern "C" __global__ void scatter_reduce_keys(
    const long long* __restrict__ indices,
    unsigned int* __restrict__ keys, unsigned int* __restrict__ vals,
    unsigned int n, unsigned int ndim, unsigned int dim, unsigned int dim_size,
    unsigned int invalid_key,
    NUMR_DIM_ARGS(src_shape), NUMR_DIM_ARGS(dst_strides)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int shape[INDEX_MAX_DIMS] = NUMR_DIM_PACK(src_shape);
    const unsigned int strides[INDEX_MAX_DIMS] = NUMR_DIM_PACK(dst_strides);

    unsigned int key = invalid_key;
    long long index_val = indices[i];
    if (index_val >= 0 && (unsigned long long)index_val < dim_size) {
        unsigned int rem = i;
        unsigned long long offset = 0;
        for (int a = (int)ndim - 1; a >= 0; --a) {
            unsigned int coord = rem % shape[a];
            rem /= shape[a];
            unsigned int c = ((unsigned int)a == dim) ? (unsigned int)index_val : coord;
            offset += (unsigned long long)c * strides[a];
        }
        key = (unsigned int)offset;
    }
    keys[i] = key;
    vals[i] = i;
}

// ============================================================================
// Radix sort passes
// ============================================================================

// Digit counts of one tile, written digit-major: hist[digit * tiles + tile].
extern "C" __global__ void scatter_reduce_radix_hist(
    const unsigned int* __restrict__ keys, unsigned int n, unsigned int shift,
    unsigned int tiles, unsigned int* __restrict__ hist
) {
    __shared__ unsigned int counts[SR_RADIX];
    for (unsigned int d = threadIdx.x; d < SR_RADIX; d += blockDim.x) counts[d] = 0;
    __syncthreads();

    unsigned long long base = (unsigned long long)blockIdx.x * SR_TILE;
    for (unsigned int k = threadIdx.x; k < SR_TILE; k += blockDim.x) {
        unsigned long long i = base + k;
        if (i < n) atomicAdd(&counts[(keys[i] >> shift) & (SR_RADIX - 1)], 1u);
    }
    __syncthreads();

    for (unsigned int d = threadIdx.x; d < SR_RADIX; d += blockDim.x) {
        hist[(unsigned long long)d * tiles + blockIdx.x] = counts[d];
    }
}

// Inclusive Hillis-Steele scan of SR_THREADS values held one per thread.
// Every thread of the block calls it.
__device__ __forceinline__ unsigned int sr_block_inclusive_scan(
    unsigned int* buf, unsigned int v
) {
    buf[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int off = 1; off < SR_THREADS; off <<= 1) {
        unsigned int x = threadIdx.x >= off ? buf[threadIdx.x - off] : 0;
        __syncthreads();
        buf[threadIdx.x] += x;
        __syncthreads();
    }
    unsigned int out = buf[threadIdx.x];
    __syncthreads();
    return out;
}

// One block per digit: replaces the digit's row of counts with each tile's
// exclusive offset within the digit, and writes the digit's total.
extern "C" __global__ void scatter_reduce_radix_scan(
    unsigned int* __restrict__ hist, unsigned int tiles,
    unsigned int* __restrict__ totals
) {
    __shared__ unsigned int buf[SR_THREADS];
    unsigned int* row = hist + (unsigned long long)blockIdx.x * tiles;
    unsigned int carry = 0;
    for (unsigned int c = 0; c < tiles; c += SR_THREADS) {
        unsigned int t = c + threadIdx.x;
        unsigned int v = t < tiles ? row[t] : 0;
        unsigned int incl = sr_block_inclusive_scan(buf, v);
        if (t < tiles) row[t] = carry + incl - v;
        // The last thread's inclusive value is the chunk total; every thread
        // reads it through shared memory.
        if (threadIdx.x == SR_THREADS - 1) buf[0] = incl;
        __syncthreads();
        carry += buf[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) totals[blockIdx.x] = carry;
}

// Moves one tile's entries to their sorted slots. Each warp owns a contiguous
// SR_WARP_SPAN of the tile and walks it in order, 32 entries per step, so the
// slot order inside one digit is source order.
extern "C" __global__ void scatter_reduce_radix_scatter(
    const unsigned int* __restrict__ keys_in, const unsigned int* __restrict__ vals_in,
    unsigned int n, unsigned int shift, unsigned int tiles,
    const unsigned int* __restrict__ hist, const unsigned int* __restrict__ totals,
    unsigned int* __restrict__ keys_out, unsigned int* __restrict__ vals_out
) {
    __shared__ unsigned int scan_buf[SR_THREADS];
    __shared__ unsigned int warp_off[SR_WARPS][SR_RADIX];

    unsigned int tid = threadIdx.x;
    unsigned int warp = tid / 32;
    unsigned int lane = tid % 32;

    for (unsigned int k = tid; k < SR_WARPS * SR_RADIX; k += SR_THREADS) {
        warp_off[k / SR_RADIX][k % SR_RADIX] = 0;
    }

    // Digit base: the exclusive scan of every digit's total. SR_THREADS equals
    // SR_RADIX, so thread `tid` owns digit `tid`.
    unsigned int total = totals[tid];
    unsigned int digit_base = sr_block_inclusive_scan(scan_buf, total) - total;

    unsigned long long warp_begin =
        (unsigned long long)blockIdx.x * SR_TILE + (unsigned long long)warp * SR_WARP_SPAN;

    for (unsigned int r = 0; r < SR_ITEMS; ++r) {
        unsigned long long i = warp_begin + r * 32 + lane;
        if (i < n) atomicAdd(&warp_off[warp][(keys_in[i] >> shift) & (SR_RADIX - 1)], 1u);
    }
    __syncthreads();

    // Thread `tid` turns digit `tid`'s per-warp counts into per-warp slots.
    {
        unsigned int running = digit_base + hist[(unsigned long long)tid * tiles + blockIdx.x];
        for (unsigned int w = 0; w < SR_WARPS; ++w) {
            unsigned int c = warp_off[w][tid];
            warp_off[w][tid] = running;
            running += c;
        }
    }
    __syncthreads();

    unsigned int lanes_before = (1u << lane) - 1u;
    for (unsigned int r = 0; r < SR_ITEMS; ++r) {
        unsigned long long i = warp_begin + r * 32 + lane;
        bool valid = i < n;
        unsigned int key = valid ? keys_in[i] : 0;
        unsigned int digit = (key >> shift) & (SR_RADIX - 1);

        // Lanes holding a valid entry with this lane's digit.
        unsigned int peers = __ballot_sync(0xffffffffu, valid);
        for (unsigned int b = 0; b < 8; ++b) {
            bool bit = (digit >> b) & 1u;
            unsigned int set = __ballot_sync(0xffffffffu, bit);
            peers &= bit ? set : ~set;
        }

        unsigned int earlier = peers & lanes_before;
        if (valid) {
            unsigned int slot = warp_off[warp][digit] + __popc(earlier);
            keys_out[slot] = key;
            vals_out[slot] = vals_in[i];
        }
        __syncwarp();
        if (valid && earlier == 0) warp_off[warp][digit] += __popc(peers);
        __syncwarp();
    }
}

// ============================================================================
// Ordered reduction, one thread per destination element
// ============================================================================

#define NUMR_SR_SUM  0
#define NUMR_SR_PROD 1
#define NUMR_SR_MAX  2
#define NUMR_SR_MIN  3
#define NUMR_SR_MEAN 4

// First position in sorted `keys[0..n)` whose key is not below `target`.
__device__ __forceinline__ unsigned int sr_lower_bound(
    const unsigned int* __restrict__ keys, unsigned int n, unsigned int target
) {
    unsigned int lo = 0, hi = n;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        if (keys[mid] < target) lo = mid + 1; else hi = mid;
    }
    return lo;
}

template<typename T, int OP>
__device__ __forceinline__ void scatter_reduce_float_impl(
    const T* __restrict__ src, const unsigned int* __restrict__ keys,
    const unsigned int* __restrict__ vals, T* __restrict__ dst,
    unsigned int n, unsigned int dst_numel, unsigned int include_self
) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dst_numel) return;
    unsigned int lo = sr_lower_bound(keys, n, d);
    unsigned int hi = sr_lower_bound(keys, n, d + 1);
    if (lo == hi) return;

    T acc = dst[d];
    for (unsigned int j = lo; j < hi; ++j) {
        T v = src[vals[j]];
        if (OP == NUMR_SR_SUM || OP == NUMR_SR_MEAN) {
            acc = acc + v;
        } else if (OP == NUMR_SR_PROD) {
            acc = acc * v;
        } else if (OP == NUMR_SR_MAX) {
            if (v > acc) acc = v;
        } else {
            if (v < acc) acc = v;
        }
    }
    if (OP == NUMR_SR_MEAN) {
        double count = (double)(hi - lo) + (include_self ? 1.0 : 0.0);
        acc = (T)((double)acc / count);
    }
    dst[d] = acc;
}

template<typename T, int OP>
__device__ __forceinline__ void scatter_reduce_int_impl(
    const T* __restrict__ src, const unsigned int* __restrict__ keys,
    const unsigned int* __restrict__ vals, T* __restrict__ dst,
    unsigned int n, unsigned int dst_numel, unsigned int include_self
) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dst_numel) return;
    unsigned int lo = sr_lower_bound(keys, n, d);
    unsigned int hi = sr_lower_bound(keys, n, d + 1);
    if (lo == hi) return;

    if (OP == NUMR_SR_MAX || OP == NUMR_SR_MIN) {
        // Comparison needs no accumulator: the result is always one of the
        // inputs, so it is exact in the element type.
        T best = dst[d];
        for (unsigned int j = lo; j < hi; ++j) {
            T v = src[vals[j]];
            if (OP == NUMR_SR_MAX ? (v > best) : (v < best)) best = v;
        }
        dst[d] = best;
        return;
    }

    Numr128 acc = Numr128From<T>::apply(dst[d]);
    for (unsigned int j = lo; j < hi; ++j) {
        Numr128 v = Numr128From<T>::apply(src[vals[j]]);
        acc = (OP == NUMR_SR_PROD) ? numr128_mul_sat(acc, v) : numr128_add_sat(acc, v);
    }
    if (OP == NUMR_SR_MEAN) {
        // include_self makes the destination's own value one of the averaged
        // contributions, matching the CPU kernel's count seed of 1.
        unsigned long long count = (unsigned long long)(hi - lo) + (include_self ? 1ULL : 0ULL);
        acc = numr128_div_u64_trunc(acc, count);
    }
    dst[d] = Numr128Narrow<T>::apply(acc);
}

#define NUMR_SCATTER_REDUCE(IMPL, OP_NAME, OP, T, S)                            \
    __global__ void scatter_reduce_##OP_NAME##_##S(                             \
        const T* __restrict__ src, const unsigned int* __restrict__ keys,       \
        const unsigned int* __restrict__ vals, T* __restrict__ dst,             \
        unsigned int n, unsigned int dst_numel, unsigned int include_self) {    \
        IMPL<T, OP>(src, keys, vals, dst, n, dst_numel, include_self);          \
    }

#define NUMR_SCATTER_REDUCE_ROW(IMPL, T, S)                                     \
    NUMR_SCATTER_REDUCE(IMPL, sum, NUMR_SR_SUM, T, S)                           \
    NUMR_SCATTER_REDUCE(IMPL, prod, NUMR_SR_PROD, T, S)                         \
    NUMR_SCATTER_REDUCE(IMPL, max, NUMR_SR_MAX, T, S)                           \
    NUMR_SCATTER_REDUCE(IMPL, min, NUMR_SR_MIN, T, S)                           \
    NUMR_SCATTER_REDUCE(IMPL, mean, NUMR_SR_MEAN, T, S)

extern "C" {

NUMR_SCATTER_REDUCE_ROW(scatter_reduce_float_impl, float, f32)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_float_impl, double, f64)

// The element types are spelled `long long` rather than `int64_t` because
// Numr128From and Numr128Narrow are specialised on the built-in names, and on
// LP64 `int64_t` is `long`, a third distinct type with no specialisation.
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, long long, i64)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, int, i32)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, short, i16)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, signed char, i8)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, unsigned long long, u64)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, unsigned int, u32)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, unsigned short, u16)
NUMR_SCATTER_REDUCE_ROW(scatter_reduce_int_impl, unsigned char, u8)

} // extern "C"
