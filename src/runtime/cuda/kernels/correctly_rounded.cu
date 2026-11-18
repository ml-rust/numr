// Correctly rounded F64 kernels of CorrectlyRoundedOps.
//
// Each lane reads input element i * step (step 0 repeats one element, step 1 walks a
// contiguous buffer) as raw bits and writes output element i as raw bits.

#include "correctly_rounded_lane.cuh"

#define CR_LANE_INDEX                                                 \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;     \
    if (i >= n) return;                                               \
    const double x = cr_from_bits(__ldg(in + (unsigned long long)i * step));

extern "C" {

__global__ void cr_exp_f64(const unsigned long long* in, unsigned int step, unsigned long long* out, unsigned int n) {
    CR_LANE_INDEX
    out[i] = cr_bits(cr_exp(x));
}

__global__ void cr_log_f64(const unsigned long long* in, unsigned int step, unsigned long long* out, unsigned int n) {
    CR_LANE_INDEX
    out[i] = cr_bits(cr_log(x));
}

__global__ void cr_log1p_f64(const unsigned long long* in, unsigned int step, unsigned long long* out, unsigned int n) {
    CR_LANE_INDEX
    out[i] = cr_bits(cr_log1p(x));
}

// The exponent arrives as its bit pattern, so every encoding reaches the lane unchanged.
__global__ void cr_pow_scalar_f64(const unsigned long long* in, unsigned int step, unsigned long long exponent_bits,
                                  unsigned long long* out, unsigned int n) {
    CR_LANE_INDEX
    out[i] = cr_bits(cr_pow(x, cr_from_bits(exponent_bits)));
}

}  // extern "C"
