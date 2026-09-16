// Fused Add + LayerNorm CUDA kernels (forward + backward)
//
// Forward:  pre_norm = input + residual
//           output   = (pre_norm - mean) / sqrt(var + eps) * weight + bias
// Backward: d_input_residual, d_weight and d_bias from grad and the saved
//           pre_norm. d_weight and d_bias accumulate across rows with an
//           atomicAdd in the storage dtype, so both must be zeroed before launch.
//
// One templated body per direction, instantiated per dtype through
// DEFINE_FUSED_ADD_LAYER_NORM. The accumulator is F32 for f32, f16, bf16 and
// FP8, F64 for f64.
//
// Shared memory: forward 2 * blockDim.x accumulator elements (mean, var),
// backward 4 * blockDim.x (mean, var, gs, gsn).

#include "norm_common.cuh"

// Each block handles one row (hidden_size elements).
template <typename T, typename Acc>
__device__ __forceinline__ void fused_add_layer_norm_impl(
    const T* input, const T* residual, const T* weight, const T* bias,
    T* output, T* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* shared
) {
    typedef AccumTraits<T, Acc> AT;
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    Acc* mean_shared = shared;
    Acc* var_shared = shared + blockDim.x;

    const T* row_in = input + row * hidden_size;
    const T* row_res = residual + row * hidden_size;
    T* row_pn = pre_norm + row * hidden_size;
    T* row_out = output + row * hidden_size;

    // Phase 1: add residual + mean
    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    // Taken from the inputs, so it needs no barrier against phase 1's own stores.
    Acc ref = AT::load(row_in, 0) + AT::load(row_res, 0);

    Acc thread_sum = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc pn = AT::load(row_in, (int)i) + AT::load(row_res, (int)i);
        AT::store(row_pn, (int)i, pn);
        thread_sum += pn - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce<Acc>(mean_shared);
    Acc shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: variance, from pre_norm read back in the storage dtype
    Acc thread_var = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc diff = (AT::load(row_pn, (int)i) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce<Acc>(var_shared);
    Acc inv_std = numr_norm_rsqrt(var_shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 3: normalize and apply affine
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc normalized = ((AT::load(row_pn, (int)i) - ref) - shifted_mean) * inv_std;
        AT::store(row_out, (int)i, normalized * AT::load(weight, (int)i) + AT::load(bias, (int)i));
    }
}

template <typename T, typename Acc>
__device__ __forceinline__ void fused_add_layer_norm_bwd_impl(
    const T* grad, const T* pre_norm, const T* weight,
    T* d_input_residual, T* d_weight, T* d_bias,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* shared
) {
    typedef AccumTraits<T, Acc> AT;
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    Acc* mean_shared = shared;
    Acc* var_shared = shared + blockDim.x;
    Acc* gs_shared = shared + 2 * blockDim.x;
    Acc* gsn_shared = shared + 3 * blockDim.x;

    const T* row_g = grad + row * hidden_size;
    const T* row_pn = pre_norm + row * hidden_size;
    T* row_dir = d_input_residual + row * hidden_size;

    // Phase 1: mean
    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance from `pre_norm` rather than
    // consuming a saved statistic, so it carries the forward pass's cancellation
    // exactly: subtracting the mean loses the mantissa whenever a row sits far
    // from zero relative to its own spread. Mean, variance and the normalized
    // value are all invariant under the shift in exact arithmetic. `mean_gs` and
    // `mean_gsn` need no shift: they average `g * w` and an already-shifted
    // normalized value.
    Acc ref = AT::load(row_pn, 0);

    Acc thread_sum = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += AT::load(row_pn, (int)i) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce<Acc>(mean_shared);
    Acc shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: variance
    Acc thread_var = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc diff = (AT::load(row_pn, (int)i) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce<Acc>(var_shared);
    Acc var = var_shared[0] / hidden_size;
    Acc inv_std = numr_norm_rsqrt(var + eps);
    __syncthreads();

    // Phase 3: mean_gs and mean_gsn
    Acc thread_gs = (Acc)0, thread_gsn = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc g = AT::load(row_g, (int)i);
        Acc w = AT::load(weight, (int)i);
        Acc normalized = ((AT::load(row_pn, (int)i) - ref) - shifted_mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    block_sum_reduce<Acc>(gs_shared);
    block_sum_reduce<Acc>(gsn_shared);
    Acc mean_gs = gs_shared[0] / hidden_size;
    Acc mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    // Phase 4: gradients
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc g = AT::load(row_g, (int)i);
        Acc w = AT::load(weight, (int)i);
        Acc normalized = ((AT::load(row_pn, (int)i) - ref) - shifted_mean) * inv_std;
        Acc d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        AT::store(row_dir, (int)i, d_ir);
        norm_atomic_add(&d_weight[i], g * normalized);
        norm_atomic_add(&d_bias[i], g);
    }
}

#define DEFINE_FUSED_ADD_LAYER_NORM(suffix, dtype, acc) \
__global__ void fused_add_layer_norm_##suffix( \
    const dtype* input, const dtype* residual, const dtype* weight, const dtype* bias, \
    dtype* output, dtype* pre_norm, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc faln_smem_##acc[]; \
    fused_add_layer_norm_impl<dtype, acc>( \
        input, residual, weight, bias, output, pre_norm, \
        batch_size, hidden_size, eps, faln_smem_##acc); \
} \
\
__global__ void fused_add_layer_norm_bwd_##suffix( \
    const dtype* grad, const dtype* pre_norm, const dtype* weight, \
    dtype* d_input_residual, dtype* d_weight, dtype* d_bias, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc faln_bwd_smem_##acc[]; \
    fused_add_layer_norm_bwd_impl<dtype, acc>( \
        grad, pre_norm, weight, d_input_residual, d_weight, d_bias, \
        batch_size, hidden_size, eps, faln_bwd_smem_##acc); \
}

extern "C" {

NORM_DTYPES(DEFINE_FUSED_ADD_LAYER_NORM)

} // extern "C"
