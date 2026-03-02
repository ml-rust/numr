// Strided copy CUDA kernel
// Copies non-contiguous (strided) tensor data to contiguous memory
//
// Shape and strides are passed as fixed-size kernel arguments (not device pointers)
// to be compatible with CUDA graph capture/replay. Device pointers to temporary
// host-allocated data would become stale on graph replay.
//
// Algorithm:
// 1. Thread 0 loads shape/strides from kernel args into shared memory (once per block)
// 2. Each thread handles one element
// 3. Convert linear destination index to multi-dimensional indices (row-major)
// 4. Calculate source byte offset using strides from shared memory
// 5. Copy element bytes from source to destination

extern "C" {

// Maximum number of dimensions supported
// Must match MAX_DIMS in strided_copy.rs
#define MAX_DIMS 8

// Generic strided copy kernel - copies element_size bytes per element
// Shape and strides are passed by value as fixed-size arrays in kernel args.
// Thread 0 in each block loads them into shared memory to avoid per-thread
// register pressure from 16 scalar args.
__global__ void strided_copy(
    const char* __restrict__ src,
    char* __restrict__ dst,
    unsigned long long shape0,
    unsigned long long shape1,
    unsigned long long shape2,
    unsigned long long shape3,
    unsigned long long shape4,
    unsigned long long shape5,
    unsigned long long shape6,
    unsigned long long shape7,
    long long stride0,
    long long stride1,
    long long stride2,
    long long stride3,
    long long stride4,
    long long stride5,
    long long stride6,
    long long stride7,
    unsigned int numel,
    unsigned int ndim,
    unsigned int elem_size,
    unsigned long long src_byte_offset
) {
    // Shared memory: shape and strides loaded once per block by thread 0
    __shared__ unsigned long long s_shape[MAX_DIMS];
    __shared__ long long s_strides[MAX_DIMS];

    if (threadIdx.x == 0) {
        s_shape[0] = shape0; s_shape[1] = shape1; s_shape[2] = shape2; s_shape[3] = shape3;
        s_shape[4] = shape4; s_shape[5] = shape5; s_shape[6] = shape6; s_shape[7] = shape7;
        s_strides[0] = stride0; s_strides[1] = stride1; s_strides[2] = stride2; s_strides[3] = stride3;
        s_strides[4] = stride4; s_strides[5] = stride5; s_strides[6] = stride6; s_strides[7] = stride7;
    }
    __syncthreads();

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;

    // Convert linear index to strided source offset (row-major)
    long long offset = 0;
    unsigned int remaining = gid;
    for (int d = (int)ndim - 1; d >= 0; d--) {
        unsigned int dim_size = (unsigned int)s_shape[d];
        unsigned int idx = remaining % dim_size;
        remaining = remaining / dim_size;
        offset += (long long)idx * s_strides[d];
    }

    // Calculate byte addresses
    unsigned long long src_byte_addr = src_byte_offset + (unsigned long long)((long long)offset * (long long)elem_size);
    unsigned long long dst_byte_addr = (unsigned long long)gid * (unsigned long long)elem_size;

    // Copy with size-specific optimization
    if (elem_size == 4) {
        *((unsigned int*)(dst + dst_byte_addr)) = *((const unsigned int*)(src + src_byte_addr));
    } else if (elem_size == 8) {
        *((unsigned long long*)(dst + dst_byte_addr)) = *((const unsigned long long*)(src + src_byte_addr));
    } else if (elem_size == 2) {
        *((unsigned short*)(dst + dst_byte_addr)) = *((const unsigned short*)(src + src_byte_addr));
    } else if (elem_size == 1) {
        dst[dst_byte_addr] = src[src_byte_addr];
    } else {
        for (unsigned int i = 0; i < elem_size; i++) {
            dst[dst_byte_addr + i] = src[src_byte_addr + i];
        }
    }
}

// Optimized version for common case: contiguous dimensions can be collapsed
// This reduces the number of divisions/modulos needed
// For tensors with many contiguous trailing dimensions, this is faster
__global__ void strided_copy_2d(
    const char* __restrict__ src,
    char* __restrict__ dst,
    unsigned long long outer_size,
    unsigned long long inner_size,
    long long outer_stride,
    unsigned int elem_size,
    unsigned long long src_byte_offset
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long numel = outer_size * inner_size;
    if (gid >= numel) return;

    // 2D decomposition
    unsigned long long outer_idx = gid / inner_size;
    unsigned long long inner_idx = gid % inner_size;

    // Source offset: outer_idx * outer_stride + inner_idx (inner is contiguous, stride=1)
    long long src_elem_offset = (long long)outer_idx * outer_stride + (long long)inner_idx;

    unsigned long long src_byte_addr = src_byte_offset + (unsigned long long)((long long)src_elem_offset * (long long)elem_size);
    unsigned long long dst_byte_addr = (unsigned long long)gid * (unsigned long long)elem_size;

    // Copy with size-specific optimization
    if (elem_size == 4) {
        *((unsigned int*)(dst + dst_byte_addr)) = *((const unsigned int*)(src + src_byte_addr));
    } else if (elem_size == 8) {
        *((unsigned long long*)(dst + dst_byte_addr)) = *((const unsigned long long*)(src + src_byte_addr));
    } else if (elem_size == 2) {
        *((unsigned short*)(dst + dst_byte_addr)) = *((const unsigned short*)(src + src_byte_addr));
    } else if (elem_size == 1) {
        dst[dst_byte_addr] = src[src_byte_addr];
    } else {
        for (unsigned int i = 0; i < elem_size; i++) {
            dst[dst_byte_addr + i] = src[src_byte_addr + i];
        }
    }
}

} // extern "C"
