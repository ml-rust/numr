// GroupNorm CUDA kernels
//
// Divides channels into num_groups and normalizes each group separately.
// Input shape: [batch, channels, spatial...]. Each block handles one
// (batch, group) pair.
//
// One templated body, instantiated per dtype through DEFINE_GROUP_NORM. The
// accumulator is F32 for f32, f16, bf16 and FP8, F64 for f64.
//
// Shared memory: 2 * blockDim.x accumulator elements (mean, then variance).

#include "norm_common.cuh"

template <typename T, typename Acc>
__device__ __forceinline__ void group_norm_impl(
    const T* input, const T* weight, const T* bias, T* output,
    unsigned int batch, unsigned int channels, unsigned int spatial,
    unsigned int num_groups, unsigned int channels_per_group, Acc eps, Acc* shared
) {
    typedef AccumTraits<T, Acc> AT;
    unsigned int b = blockIdx.x / num_groups;
    unsigned int g = blockIdx.x % num_groups;

    if (b >= batch || g >= num_groups) return;

    Acc* mean_shared = shared;
    Acc* var_shared = shared + blockDim.x;

    unsigned int group_size = channels_per_group * spatial;
    unsigned int c_start = g * channels_per_group;

    // Shift every element by the group's first value before accumulating.
    // `x - mean` cancels catastrophically when a group sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    Acc ref = AT::load(input + (b * channels + c_start) * spatial, 0);

    // Phase 1: mean of the shifted values
    Acc thread_sum = (Acc)0;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        thread_sum += AT::load(input + offset, 0) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    Acc shifted_mean = block_sum_reduce<Acc>(mean_shared) / group_size;
    __syncthreads();

    // Phase 2: variance
    Acc thread_var = (Acc)0;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        Acc diff = (AT::load(input + offset, 0) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    Acc inv_std = numr_norm_rsqrt(block_sum_reduce<Acc>(var_shared) / group_size + eps);
    __syncthreads();

    // Phase 3: normalize and apply affine transform
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        Acc normalized = ((AT::load(input + offset, 0) - ref) - shifted_mean) * inv_std;
        AT::store(output + offset, 0, normalized * AT::load(weight, (int)c) + AT::load(bias, (int)c));
    }
}

#define DEFINE_GROUP_NORM(suffix, dtype, acc) \
__global__ void group_norm_##suffix( \
    const dtype* input, const dtype* weight, const dtype* bias, dtype* output, \
    unsigned int batch, unsigned int channels, unsigned int spatial, \
    unsigned int num_groups, unsigned int channels_per_group, acc eps \
) { \
    extern __shared__ acc group_smem_##acc[]; \
    group_norm_impl<dtype, acc>( \
        input, weight, bias, output, batch, channels, spatial, \
        num_groups, channels_per_group, eps, group_smem_##acc); \
}

extern "C" {

NORM_DTYPES(DEFINE_GROUP_NORM)

} // extern "C"
