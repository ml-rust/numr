// Strided-to-strided copy CUDA kernel
// Copies a strided source view into a strided destination view of the same shape.
//
// Shape and both stride sets are passed as fixed-size kernel arguments (not
// device pointers), the same way strided_copy does, so the launch captures
// into a CUDA graph: no allocation and no host-to-device copy per call.
//
// Algorithm:
// 1. Thread 0 loads shape and strides from kernel args into shared memory (once per block)
// 2. Each thread handles one logical element
// 3. Convert the linear index to multi-dimensional indices (row-major)
// 4. Accumulate the source and destination element offsets with their strides
// 5. Copy element bytes from the source address to the destination address

extern "C" {

// Maximum number of dimensions supported
// Must match MAX_DIMS in strided_copy.rs
#define MAX_DIMS 8

__global__ void strided_copy_into(
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
    long long src_stride0,
    long long src_stride1,
    long long src_stride2,
    long long src_stride3,
    long long src_stride4,
    long long src_stride5,
    long long src_stride6,
    long long src_stride7,
    long long dst_stride0,
    long long dst_stride1,
    long long dst_stride2,
    long long dst_stride3,
    long long dst_stride4,
    long long dst_stride5,
    long long dst_stride6,
    long long dst_stride7,
    unsigned int numel,
    unsigned int ndim,
    unsigned int elem_size,
    unsigned long long src_byte_offset,
    unsigned long long dst_byte_offset
) {
    __shared__ unsigned long long s_shape[MAX_DIMS];
    __shared__ long long s_src_strides[MAX_DIMS];
    __shared__ long long s_dst_strides[MAX_DIMS];

    if (threadIdx.x == 0) {
        s_shape[0] = shape0; s_shape[1] = shape1; s_shape[2] = shape2; s_shape[3] = shape3;
        s_shape[4] = shape4; s_shape[5] = shape5; s_shape[6] = shape6; s_shape[7] = shape7;
        s_src_strides[0] = src_stride0; s_src_strides[1] = src_stride1;
        s_src_strides[2] = src_stride2; s_src_strides[3] = src_stride3;
        s_src_strides[4] = src_stride4; s_src_strides[5] = src_stride5;
        s_src_strides[6] = src_stride6; s_src_strides[7] = src_stride7;
        s_dst_strides[0] = dst_stride0; s_dst_strides[1] = dst_stride1;
        s_dst_strides[2] = dst_stride2; s_dst_strides[3] = dst_stride3;
        s_dst_strides[4] = dst_stride4; s_dst_strides[5] = dst_stride5;
        s_dst_strides[6] = dst_stride6; s_dst_strides[7] = dst_stride7;
    }
    __syncthreads();

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;

    // Convert linear index to strided source and destination offsets (row-major)
    long long src_offset = 0;
    long long dst_offset = 0;
    unsigned int remaining = gid;
    for (int d = (int)ndim - 1; d >= 0; d--) {
        unsigned int dim_size = (unsigned int)s_shape[d];
        unsigned int idx = remaining % dim_size;
        remaining = remaining / dim_size;
        src_offset += (long long)idx * s_src_strides[d];
        dst_offset += (long long)idx * s_dst_strides[d];
    }

    unsigned long long src_byte_addr = src_byte_offset + (unsigned long long)(src_offset * (long long)elem_size);
    unsigned long long dst_byte_addr = dst_byte_offset + (unsigned long long)(dst_offset * (long long)elem_size);

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
