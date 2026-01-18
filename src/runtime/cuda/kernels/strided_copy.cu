// Strided copy CUDA kernel
// Copies non-contiguous (strided) tensor data to contiguous memory
//
// Algorithm:
// 1. Each thread handles one element
// 2. Convert linear destination index to multi-dimensional indices (row-major)
// 3. Calculate source byte offset using strides
// 4. Copy element bytes from source to destination

extern "C" {

// Maximum number of dimensions supported
// 8 dimensions covers most practical tensor use cases
#define MAX_DIMS 8

// Device function: Convert linear index to strided source offset
// Uses row-major indexing (C-style) - iterate dimensions from high to low
__device__ __forceinline__ long long get_strided_offset(
    unsigned int linear_idx,
    unsigned int ndim,
    const unsigned long long* shape,
    const long long* strides
) {
    long long offset = 0;
    unsigned int remaining = linear_idx;

    // Iterate through dimensions in reverse order (row-major)
    for (int d = (int)ndim - 1; d >= 0; d--) {
        unsigned int dim_size = (unsigned int)shape[d];
        unsigned int idx = remaining % dim_size;
        remaining = remaining / dim_size;
        offset += (long long)idx * strides[d];
    }

    return offset;
}

// Generic strided copy kernel - copies element_size bytes per element
// This works for any dtype (f32=4, f64=8, f16=2, etc.)
__global__ void strided_copy(
    const char* __restrict__ src,
    char* __restrict__ dst,
    const unsigned long long* __restrict__ shape,
    const long long* __restrict__ strides,
    unsigned int numel,
    unsigned int ndim,
    unsigned int elem_size,
    unsigned long long src_byte_offset
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;

    // Calculate source element offset (in elements)
    long long src_elem_offset = get_strided_offset(gid, ndim, shape, strides);

    // Calculate byte addresses
    // src_byte_offset is the initial offset into source buffer
    // src_elem_offset is the strided offset in elements
    unsigned long long src_byte_addr = src_byte_offset + (unsigned long long)((long long)src_elem_offset * (long long)elem_size);
    unsigned long long dst_byte_addr = (unsigned long long)gid * (unsigned long long)elem_size;

    // Copy element bytes
    // For common element sizes, use optimized paths
    if (elem_size == 4) {
        // 4-byte elements (f32, i32, u32)
        *((unsigned int*)(dst + dst_byte_addr)) = *((const unsigned int*)(src + src_byte_addr));
    } else if (elem_size == 8) {
        // 8-byte elements (f64, i64, u64)
        *((unsigned long long*)(dst + dst_byte_addr)) = *((const unsigned long long*)(src + src_byte_addr));
    } else if (elem_size == 2) {
        // 2-byte elements (f16, bf16, i16, u16)
        *((unsigned short*)(dst + dst_byte_addr)) = *((const unsigned short*)(src + src_byte_addr));
    } else if (elem_size == 1) {
        // 1-byte elements (i8, u8, bool)
        dst[dst_byte_addr] = src[src_byte_addr];
    } else {
        // Generic byte-by-byte copy for unusual element sizes
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
