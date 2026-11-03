// Gather kernels for pixel_shuffle, pixel_unshuffle and upsample_nearest2d.
//
// This file is PTX module "image" (IMAGE_MODULE in image.rs).
//
// Kernel naming: {op}_b{bytes}, for op in pixel_shuffle, pixel_unshuffle,
// upsample_nearest2d and element sizes 1, 2, 4, 8 and 16 bytes. The ops move
// elements and compute nothing, so the element size selects the kernel and
// every dtype of that size shares it.
//
// One thread owns one output element, in a grid-stride loop with 64-bit
// indices. The linear index decodes into canonical (n, c, h, w) output
// coordinates: W fastest for NCHW order, C fastest for NHWC order
// (`channels_last`). The op maps them to canonical source coordinates, and
// both sides are addressed through canonical [N, C, H, W] strides in
// elements. Strides can be negative. Every kernel takes the same argument
// list and ignores what its op does not use.
//
// Nearest source index per axis, matching PyTorch's CPU `nearest_idx`:
//   mode 0: src = dst
//   mode 1: src = dst >> 1
//   mode 2: src = min(floor(f32(dst) * scale), in - 1), one rounded f32 multiply

struct numr_image_b16 {
    unsigned long long lo;
    unsigned long long hi;
};

struct ImageShuffleMap {
    unsigned long long r;
    __device__ __forceinline__ void operator()(
        unsigned long long c, unsigned long long h, unsigned long long w,
        unsigned long long* sc, unsigned long long* sh, unsigned long long* sw) const {
        *sc = c * r * r + (h % r) * r + (w % r);
        *sh = h / r;
        *sw = w / r;
    }
};

struct ImageUnshuffleMap {
    unsigned long long r;
    __device__ __forceinline__ void operator()(
        unsigned long long c, unsigned long long h, unsigned long long w,
        unsigned long long* sc, unsigned long long* sh, unsigned long long* sw) const {
        unsigned long long r2 = r * r;
        unsigned long long rem = c % r2;
        *sc = c / r2;
        *sh = h * r + rem / r;
        *sw = w * r + rem % r;
    }
};

__device__ __forceinline__ unsigned long long numr_nearest_src(
    unsigned long long dst, unsigned long long in_size, unsigned int mode, float scale) {
    if (mode == 0u) {
        return dst;
    }
    if (mode == 1u) {
        return dst >> 1;
    }
    float f = floorf(__fmul_rn(__ull2float_rn(dst), scale));
    unsigned long long src = (unsigned long long)f;
    return src < in_size - 1ull ? src : in_size - 1ull;
}

struct ImageNearestMap {
    unsigned long long in_h;
    unsigned long long in_w;
    unsigned int mode_h;
    unsigned int mode_w;
    float scale_h;
    float scale_w;
    __device__ __forceinline__ void operator()(
        unsigned long long c, unsigned long long h, unsigned long long w,
        unsigned long long* sc, unsigned long long* sh, unsigned long long* sw) const {
        *sc = c;
        *sh = numr_nearest_src(h, in_h, mode_h, scale_h);
        *sw = numr_nearest_src(w, in_w, mode_w, scale_w);
    }
};

template <typename T, typename Map>
__device__ __forceinline__ void numr_image_gather(
    const T* __restrict__ src, T* __restrict__ dst, unsigned long long total,
    unsigned long long dim_c, unsigned long long dim_h, unsigned long long dim_w,
    long long ss0, long long ss1, long long ss2, long long ss3,
    long long ds0, long long ds1, long long ds2, long long ds3,
    unsigned int channels_last, Map map) {
    unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += step) {
        unsigned long long n, c, h, w, rest;
        if (channels_last) {
            c = i % dim_c;
            rest = i / dim_c;
            w = rest % dim_w;
            rest /= dim_w;
            h = rest % dim_h;
            n = rest / dim_h;
        } else {
            w = i % dim_w;
            rest = i / dim_w;
            h = rest % dim_h;
            rest /= dim_h;
            c = rest % dim_c;
            n = rest / dim_c;
        }
        unsigned long long sc, sh, sw;
        map(c, h, w, &sc, &sh, &sw);
        long long s_off = (long long)n * ss0 + (long long)sc * ss1 + (long long)sh * ss2 +
                          (long long)sw * ss3;
        long long d_off = (long long)n * ds0 + (long long)c * ds1 + (long long)h * ds2 +
                          (long long)w * ds3;
        dst[d_off] = src[s_off];
    }
}

// Shared argument list. Dims are the canonical output extents.
#define NUMR_IMAGE_PARAMS(T)                                                         \
    const T* __restrict__ src, T* __restrict__ dst, unsigned long long total,        \
    unsigned long long dim_c, unsigned long long dim_h, unsigned long long dim_w,    \
    long long ss0, long long ss1, long long ss2, long long ss3,                      \
    long long ds0, long long ds1, long long ds2, long long ds3,                      \
    unsigned int channels_last, unsigned long long factor,                           \
    unsigned long long in_h, unsigned long long in_w,                                \
    unsigned int mode_h, unsigned int mode_w, float scale_h, float scale_w

#define NUMR_IMAGE_FORWARD                                                           \
    src, dst, total, dim_c, dim_h, dim_w, ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3,    \
    channels_last

#define DEFINE_IMAGE_KERNELS(bytes, T)                                               \
__global__ void pixel_shuffle_b##bytes(NUMR_IMAGE_PARAMS(T)) {                       \
    ImageShuffleMap map = {factor};                                                  \
    numr_image_gather<T>(NUMR_IMAGE_FORWARD, map);                                   \
}                                                                                    \
__global__ void pixel_unshuffle_b##bytes(NUMR_IMAGE_PARAMS(T)) {                     \
    ImageUnshuffleMap map = {factor};                                                \
    numr_image_gather<T>(NUMR_IMAGE_FORWARD, map);                                   \
}                                                                                    \
__global__ void upsample_nearest2d_b##bytes(NUMR_IMAGE_PARAMS(T)) {                  \
    ImageNearestMap map = {in_h, in_w, mode_h, mode_w, scale_h, scale_w};            \
    numr_image_gather<T>(NUMR_IMAGE_FORWARD, map);                                   \
}

extern "C" {

DEFINE_IMAGE_KERNELS(1, unsigned char)
DEFINE_IMAGE_KERNELS(2, unsigned short)
DEFINE_IMAGE_KERNELS(4, unsigned int)
DEFINE_IMAGE_KERNELS(8, unsigned long long)
DEFINE_IMAGE_KERNELS(16, numr_image_b16)

} // extern "C"
