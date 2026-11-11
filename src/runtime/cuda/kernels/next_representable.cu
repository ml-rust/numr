// Integer loads preserve signaling NaNs and subnormals under every floating mode.

template <typename U, U SIGN, U INF, bool UPWARD>
__device__ __forceinline__ U neighbor_bits(U bits) {
    const U magnitude = bits & ~SIGN;
    if (magnitude > INF) return bits;
    if (magnitude == 0) return UPWARD ? U(1) : (SIGN | U(1));
    if (UPWARD && bits == INF) return bits;
    if (!UPWARD && bits == (SIGN | INF)) return bits;
    return ((bits & SIGN) != 0) == UPWARD ? bits - U(1) : bits + U(1);
}

extern "C" {

__global__ void next_up_f32(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int numel
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        output[i] = neighbor_bits<unsigned int, 0x80000000u, 0x7f800000u, true>(input[i]);
    }
}

__global__ void next_down_f32(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int numel
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        output[i] = neighbor_bits<unsigned int, 0x80000000u, 0x7f800000u, false>(input[i]);
    }
}

__global__ void next_up_f64(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int numel
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        output[i] = neighbor_bits<unsigned long long, 0x8000000000000000ull, 0x7ff0000000000000ull, true>(input[i]);
    }
}

__global__ void next_down_f64(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int numel
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        output[i] = neighbor_bits<unsigned long long, 0x8000000000000000ull, 0x7ff0000000000000ull, false>(input[i]);
    }
}

}
