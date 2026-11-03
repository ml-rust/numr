// Epilogue of the implicit-GEMM conv2d in conv_implicit_gemm.cuh.
//
// Per output element, in F32, in this order:
//   y = acc + bias[co]                      (when bias is not null)
//   y = act(y)
//   y = residual[pixel, co] + scale * y     (when residual is not null)
// then one narrowing store at the output's N/H/W strides. This matches
// `FusedConvOps::conv2d_fused`.
//
// Activation codes are this kernel's own. They are not the GEMM epilogue
// codes of gemm_activation.cuh. The host maps them in
// loader/conv_implicit_gemm.rs (`ConvEpilogueActivation`).

#ifndef NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_CUH
#define NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_CUH

#define CONV_ACT_NONE  0u
#define CONV_ACT_RELU  1u
#define CONV_ACT_LEAKY 2u
#define CONV_ACT_PRELU 3u

/* `neg_slope` is the LeakyRelu slope or the PReLU weight of this channel. */
__device__ __forceinline__ float conv_igemm_activate(float y, unsigned int act, float neg_slope) {
    switch (act) {
        case CONV_ACT_RELU: return fmaxf(y, 0.0f);
        case CONV_ACT_LEAKY:
        case CONV_ACT_PRELU: return y > 0.0f ? y : neg_slope * y;
        default: return y;
    }
}

/* Bias, activation, residual and store for the element at table entry PIX,
   output channel CO, with F32 accumulator ACC. Expands against the kernel
   parameters of DEFINE_CONV_IGEMM. A PReLU weight of length 1 is shared by
   every channel. */
#define CONV_EPILOGUE_STORE(LOAD_F32, STORE_FN, ACC, PIX, CO)                \
{                                                                            \
    float y = (ACC);                                                         \
    if (bias != nullptr) y += LOAD_F32(bias[(CO)]);                          \
    const float neg_slope = (act == CONV_ACT_PRELU)                          \
        ? LOAD_F32(prelu[prelu_len == 1 ? 0 : (CO)])                         \
        : slope;                                                             \
    y = conv_igemm_activate(y, act, neg_slope);                              \
    if (residual != nullptr) {                                               \
        const long long res_off = (long long)(PIX).x * res_sn                \
            + (long long)(PIX).y * res_sh + (long long)(PIX).z * res_sw      \
            + (CO);                                                          \
        y = LOAD_F32(residual[res_off]) + res_scale * y;                     \
    }                                                                        \
    const long long out_off = (long long)(PIX).x * out_sn                    \
        + (long long)(PIX).y * out_sh + (long long)(PIX).z * out_sw + (CO);  \
    out[out_off] = STORE_FN(y);                                              \
}

#endif  // NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_CUH
