// Backward kernels for fused GEMM epilogue: activation(A @ B + bias)
//
// Kernels per dtype:
// 1. gemm_bias_act_bwd_grad_pre: grad_pre = grad * act'(A @ B + bias)
// 2. gemm_bwd_da: d_a = grad_pre @ B^T
// 3. gemm_bwd_db: d_b = A^T @ grad_pre  (write)
// 4. gemm_bwd_db_accum: d_b += A^T @ grad_pre  (accumulate for batched)
// 5. gemm_bwd_dbias: d_bias = sum(grad_pre, dim=0)  (write)
// 6. gemm_bwd_dbias_accum: d_bias += sum(grad_pre, dim=0)  (accumulate for batched)
//
// activation_type: 0=None, 1=ReLU, 2=GELU, 3=SiLU, 4=Sigmoid, 5=Tanh

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "activation_deriv.cuh"

extern "C" {

// ============================================================================
// F32 Backward Kernels
// ============================================================================

__global__ void gemm_bias_act_bwd_grad_pre_f32(
    const float* __restrict__ grad,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ grad_pre,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int activation_type
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int i = idx / N;
    unsigned int j = idx % N;
    double pre_act = (double)bias[j];
    for (unsigned int kk = 0; kk < K; kk++) {
        pre_act += (double)A[i * K + kk] * (double)B[kk * N + j];
    }
    grad_pre[idx] = grad[idx] * (float)activation_deriv_f64(pre_act, activation_type);
}

__global__ void gemm_bwd_da_f32(
    const float* __restrict__ grad_pre,
    const float* __restrict__ B,
    float* __restrict__ d_a,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    unsigned int i = idx / K;
    unsigned int k = idx % K;
    double sum = 0.0;
    for (unsigned int j = 0; j < N; j++) {
        sum += (double)grad_pre[i * N + j] * (double)B[k * N + j];
    }
    d_a[idx] = (float)sum;
}

__global__ void gemm_bwd_db_f32(
    const float* __restrict__ A,
    const float* __restrict__ grad_pre,
    float* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    double sum = 0.0;
    for (unsigned int i = 0; i < M; i++) {
        sum += (double)A[i * K + k] * (double)grad_pre[i * N + j];
    }
    d_b[idx] = (float)sum;
}

__global__ void gemm_bwd_db_accum_f32(
    const float* __restrict__ A,
    const float* __restrict__ grad_pre,
    float* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    double sum = 0.0;
    for (unsigned int i = 0; i < M; i++) {
        sum += (double)A[i * K + k] * (double)grad_pre[i * N + j];
    }
    d_b[idx] += (float)sum;
}

__global__ void gemm_bwd_dbias_f32(
    const float* __restrict__ grad_pre,
    float* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += grad_pre[i * N + j];
    }
    d_bias[j] = sum;
}

__global__ void gemm_bwd_dbias_accum_f32(
    const float* __restrict__ grad_pre,
    float* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += grad_pre[i * N + j];
    }
    d_bias[j] += sum;
}

// ============================================================================
// F64 Backward Kernels
// ============================================================================

__global__ void gemm_bias_act_bwd_grad_pre_f64(
    const double* __restrict__ grad,
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    double* __restrict__ grad_pre,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int activation_type
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int i = idx / N;
    unsigned int j = idx % N;
    double pre_act = bias[j];
    for (unsigned int kk = 0; kk < K; kk++) {
        pre_act += A[i * K + kk] * B[kk * N + j];
    }
    grad_pre[idx] = grad[idx] * activation_deriv_f64(pre_act, activation_type);
}

__global__ void gemm_bwd_da_f64(
    const double* __restrict__ grad_pre,
    const double* __restrict__ B,
    double* __restrict__ d_a,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    unsigned int i = idx / K;
    unsigned int k = idx % K;
    double sum = 0.0;
    for (unsigned int j = 0; j < N; j++) {
        sum += grad_pre[i * N + j] * B[k * N + j];
    }
    d_a[idx] = sum;
}

__global__ void gemm_bwd_db_f64(
    const double* __restrict__ A,
    const double* __restrict__ grad_pre,
    double* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    double sum = 0.0;
    for (unsigned int i = 0; i < M; i++) {
        sum += A[i * K + k] * grad_pre[i * N + j];
    }
    d_b[idx] = sum;
}

__global__ void gemm_bwd_db_accum_f64(
    const double* __restrict__ A,
    const double* __restrict__ grad_pre,
    double* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    double sum = 0.0;
    for (unsigned int i = 0; i < M; i++) {
        sum += A[i * K + k] * grad_pre[i * N + j];
    }
    d_b[idx] += sum;
}

__global__ void gemm_bwd_dbias_f64(
    const double* __restrict__ grad_pre,
    double* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    double sum = 0.0;
    for (unsigned int i = 0; i < M; i++) {
        sum += grad_pre[i * N + j];
    }
    d_bias[j] = sum;
}

__global__ void gemm_bwd_dbias_accum_f64(
    const double* __restrict__ grad_pre,
    double* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    double sum = 0.0;
    for (unsigned int i = 0; i < M; i++) {
        sum += grad_pre[i * N + j];
    }
    d_bias[j] += sum;
}

// ============================================================================
// F16 Backward Kernels (compute in F32)
// ============================================================================

__global__ void gemm_bias_act_bwd_grad_pre_f16(
    const __half* __restrict__ grad,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ grad_pre,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int activation_type
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int i = idx / N;
    unsigned int j = idx % N;
    float pre_act = __half2float(bias[j]);
    for (unsigned int kk = 0; kk < K; kk++) {
        pre_act += __half2float(A[i * K + kk]) * __half2float(B[kk * N + j]);
    }
    grad_pre[idx] = __float2half(__half2float(grad[idx]) * activation_deriv_f32(pre_act, activation_type));
}

__global__ void gemm_bwd_da_f16(
    const __half* __restrict__ grad_pre,
    const __half* __restrict__ B,
    __half* __restrict__ d_a,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    unsigned int i = idx / K;
    unsigned int k = idx % K;
    float sum = 0.0f;
    for (unsigned int j = 0; j < N; j++) {
        sum += __half2float(grad_pre[i * N + j]) * __half2float(B[k * N + j]);
    }
    d_a[idx] = __float2half(sum);
}

__global__ void gemm_bwd_db_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ grad_pre,
    __half* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __half2float(A[i * K + k]) * __half2float(grad_pre[i * N + j]);
    }
    d_b[idx] = __float2half(sum);
}

__global__ void gemm_bwd_db_accum_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ grad_pre,
    __half* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __half2float(A[i * K + k]) * __half2float(grad_pre[i * N + j]);
    }
    d_b[idx] = __float2half(__half2float(d_b[idx]) + sum);
}

__global__ void gemm_bwd_dbias_f16(
    const __half* __restrict__ grad_pre,
    __half* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __half2float(grad_pre[i * N + j]);
    }
    d_bias[j] = __float2half(sum);
}

__global__ void gemm_bwd_dbias_accum_f16(
    const __half* __restrict__ grad_pre,
    __half* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __half2float(grad_pre[i * N + j]);
    }
    d_bias[j] = __float2half(__half2float(d_bias[j]) + sum);
}

// ============================================================================
// BF16 Backward Kernels (compute in F32)
// ============================================================================

__global__ void gemm_bias_act_bwd_grad_pre_bf16(
    const __nv_bfloat16* __restrict__ grad,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ grad_pre,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int activation_type
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int i = idx / N;
    unsigned int j = idx % N;
    float pre_act = __bfloat162float(bias[j]);
    for (unsigned int kk = 0; kk < K; kk++) {
        pre_act += __bfloat162float(A[i * K + kk]) * __bfloat162float(B[kk * N + j]);
    }
    grad_pre[idx] = __float2bfloat16(__bfloat162float(grad[idx]) * activation_deriv_f32(pre_act, activation_type));
}

__global__ void gemm_bwd_da_bf16(
    const __nv_bfloat16* __restrict__ grad_pre,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ d_a,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    unsigned int i = idx / K;
    unsigned int k = idx % K;
    float sum = 0.0f;
    for (unsigned int j = 0; j < N; j++) {
        sum += __bfloat162float(grad_pre[i * N + j]) * __bfloat162float(B[k * N + j]);
    }
    d_a[idx] = __float2bfloat16(sum);
}

__global__ void gemm_bwd_db_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ grad_pre,
    __nv_bfloat16* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __bfloat162float(A[i * K + k]) * __bfloat162float(grad_pre[i * N + j]);
    }
    d_b[idx] = __float2bfloat16(sum);
}

__global__ void gemm_bwd_db_accum_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ grad_pre,
    __nv_bfloat16* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __bfloat162float(A[i * K + k]) * __bfloat162float(grad_pre[i * N + j]);
    }
    d_b[idx] = __float2bfloat16(__bfloat162float(d_b[idx]) + sum);
}

__global__ void gemm_bwd_dbias_bf16(
    const __nv_bfloat16* __restrict__ grad_pre,
    __nv_bfloat16* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __bfloat162float(grad_pre[i * N + j]);
    }
    d_bias[j] = __float2bfloat16(sum);
}

__global__ void gemm_bwd_dbias_accum_bf16(
    const __nv_bfloat16* __restrict__ grad_pre,
    __nv_bfloat16* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += __bfloat162float(grad_pre[i * N + j]);
    }
    d_bias[j] = __float2bfloat16(__bfloat162float(d_bias[j]) + sum);
}

// ============================================================================
// FP8 E4M3 Backward Kernels (compute in F32)
// ============================================================================

__global__ void gemm_bias_act_bwd_grad_pre_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ grad,
    const numr_fp8_e4m3* __restrict__ A,
    const numr_fp8_e4m3* __restrict__ B,
    const numr_fp8_e4m3* __restrict__ bias,
    numr_fp8_e4m3* __restrict__ grad_pre,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int activation_type
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int i = idx / N;
    unsigned int j = idx % N;
    float pre_act = fp8_e4m3_to_f32(bias[j].data);
    for (unsigned int kk = 0; kk < K; kk++) {
        pre_act += fp8_e4m3_to_f32(A[i * K + kk].data) * fp8_e4m3_to_f32(B[kk * N + j].data);
    }
    float g = fp8_e4m3_to_f32(grad[idx].data);
    grad_pre[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * activation_deriv_f32(pre_act, activation_type)));
}

__global__ void gemm_bwd_da_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ grad_pre,
    const numr_fp8_e4m3* __restrict__ B,
    numr_fp8_e4m3* __restrict__ d_a,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    unsigned int i = idx / K;
    unsigned int k = idx % K;
    float sum = 0.0f;
    for (unsigned int j = 0; j < N; j++) {
        sum += fp8_e4m3_to_f32(grad_pre[i * N + j].data) * fp8_e4m3_to_f32(B[k * N + j].data);
    }
    d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sum));
}

__global__ void gemm_bwd_db_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ A,
    const numr_fp8_e4m3* __restrict__ grad_pre,
    numr_fp8_e4m3* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e4m3_to_f32(A[i * K + k].data) * fp8_e4m3_to_f32(grad_pre[i * N + j].data);
    }
    d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sum));
}

__global__ void gemm_bwd_db_accum_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ A,
    const numr_fp8_e4m3* __restrict__ grad_pre,
    numr_fp8_e4m3* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e4m3_to_f32(A[i * K + k].data) * fp8_e4m3_to_f32(grad_pre[i * N + j].data);
    }
    d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fp8_e4m3_to_f32(d_b[idx].data) + sum));
}

__global__ void gemm_bwd_dbias_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ grad_pre,
    numr_fp8_e4m3* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e4m3_to_f32(grad_pre[i * N + j].data);
    }
    d_bias[j] = numr_fp8_e4m3(f32_to_fp8_e4m3(sum));
}

__global__ void gemm_bwd_dbias_accum_fp8_e4m3(
    const numr_fp8_e4m3* __restrict__ grad_pre,
    numr_fp8_e4m3* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e4m3_to_f32(grad_pre[i * N + j].data);
    }
    d_bias[j] = numr_fp8_e4m3(f32_to_fp8_e4m3(fp8_e4m3_to_f32(d_bias[j].data) + sum));
}

// ============================================================================
// FP8 E5M2 Backward Kernels (compute in F32)
// ============================================================================

__global__ void gemm_bias_act_bwd_grad_pre_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ grad,
    const numr_fp8_e5m2* __restrict__ A,
    const numr_fp8_e5m2* __restrict__ B,
    const numr_fp8_e5m2* __restrict__ bias,
    numr_fp8_e5m2* __restrict__ grad_pre,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int activation_type
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int i = idx / N;
    unsigned int j = idx % N;
    float pre_act = fp8_e5m2_to_f32(bias[j].data);
    for (unsigned int kk = 0; kk < K; kk++) {
        pre_act += fp8_e5m2_to_f32(A[i * K + kk].data) * fp8_e5m2_to_f32(B[kk * N + j].data);
    }
    float g = fp8_e5m2_to_f32(grad[idx].data);
    grad_pre[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * activation_deriv_f32(pre_act, activation_type)));
}

__global__ void gemm_bwd_da_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ grad_pre,
    const numr_fp8_e5m2* __restrict__ B,
    numr_fp8_e5m2* __restrict__ d_a,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    unsigned int i = idx / K;
    unsigned int k = idx % K;
    float sum = 0.0f;
    for (unsigned int j = 0; j < N; j++) {
        sum += fp8_e5m2_to_f32(grad_pre[i * N + j].data) * fp8_e5m2_to_f32(B[k * N + j].data);
    }
    d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sum));
}

__global__ void gemm_bwd_db_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ A,
    const numr_fp8_e5m2* __restrict__ grad_pre,
    numr_fp8_e5m2* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e5m2_to_f32(A[i * K + k].data) * fp8_e5m2_to_f32(grad_pre[i * N + j].data);
    }
    d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sum));
}

__global__ void gemm_bwd_db_accum_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ A,
    const numr_fp8_e5m2* __restrict__ grad_pre,
    numr_fp8_e5m2* __restrict__ d_b,
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    unsigned int k = idx / N;
    unsigned int j = idx % N;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e5m2_to_f32(A[i * K + k].data) * fp8_e5m2_to_f32(grad_pre[i * N + j].data);
    }
    d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fp8_e5m2_to_f32(d_b[idx].data) + sum));
}

__global__ void gemm_bwd_dbias_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ grad_pre,
    numr_fp8_e5m2* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e5m2_to_f32(grad_pre[i * N + j].data);
    }
    d_bias[j] = numr_fp8_e5m2(f32_to_fp8_e5m2(sum));
}

__global__ void gemm_bwd_dbias_accum_fp8_e5m2(
    const numr_fp8_e5m2* __restrict__ grad_pre,
    numr_fp8_e5m2* __restrict__ d_bias,
    unsigned int M, unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < M; i++) {
        sum += fp8_e5m2_to_f32(grad_pre[i * N + j].data);
    }
    d_bias[j] = numr_fp8_e5m2(f32_to_fp8_e5m2(fp8_e5m2_to_f32(d_bias[j].data) + sum));
}

} // extern "C"
