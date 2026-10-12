# Backends

numr runs the same generic code on every backend. A `Tensor<R>` is generic over a `Runtime`,
so switching backends changes the type parameter, not the algorithm.

| Runtime | Hardware | Feature | Kernels |
| ------- | -------- | ------- | ------- |
| `CpuRuntime` | x86-64 and ARM64 CPUs | `cpu` (default, always available) | Native SIMD (AVX-512, AVX2, NEON) |
| `CudaRuntime` | NVIDIA GPUs | `cuda` | Native CUDA kernels in multi-arch fatbins, with a caching allocator |
| `WgpuRuntime` | Vulkan, Metal, and DX12 GPUs through WebGPU | `wgpu` | WGSL compute shaders |

No backend depends on cuBLAS, MKL, or another vendor math library. ROCm is not implemented.

## Backend Dtypes

The `DType` enum lists 15 types. Feature flags gate some of them.

| Type | Size | CPU | CUDA | Feature |
| ---- | ---- | --- | ---- | ------- |
| `F64` | 8 B | ✓ | ✓ | - |
| `F32` | 4 B | ✓ | ✓ | - |
| `F16` | 2 B | ✓ | ✓ | `f16` |
| `BF16` | 2 B | ✓ | ✓ | `f16` |
| `FP8E4M3` | 1 B | ✓ | ✓ | `fp8` |
| `FP8E5M2` | 1 B | ✓ | ✓ | `fp8` |
| `I64` | 8 B | ✓ | ✓ | - |
| `I32` | 4 B | ✓ | ✓ | - |
| `I16` | 2 B | ✓ | ✓ | - |
| `I8` | 1 B | ✓ | ✓ | - |
| `U64` | 8 B | ✓ | ✓ | - |
| `U32` | 4 B | ✓ | ✓ | - |
| `U16` | 2 B | ✓ | ✓ | - |
| `U8` | 1 B | ✓ | ✓ | - |
| `Bool` | 1 B | ✓ | ✓ | - |

The WebGPU backend computes in 32-bit only. Its element-wise, scalar, comparison, and activation
operations accept `F32` only. Its `cast` converts between `F32`, `I32`, and `U32`. Other dtypes
return `Error::UnsupportedDType`. Per-operation dtype coverage on CPU and CUDA is not listed here.
Check the operation before you rely on a dtype.

## CPU SIMD

The CPU backend detects the best instruction set at run time.

| Architecture | Instruction set | Vector width | Elements per op |
| ------------ | --------------- | ------------ | --------------- |
| x86-64 | AVX-512F + FMA | 512 bits | 16 `f32` / 8 `f64` |
| x86-64 | AVX2 + FMA | 256 bits | 8 `f32` / 4 `f64` |
| ARM64 | NEON | 128 bits | 4 `f32` / 2 `f64` |

Vectorized operations include:

- Element-wise: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, and more
- Reductions: `sum`, `max`, `min`, `prod`, with horizontal SIMD reductions
- Activations: `sigmoid`, `silu`, `gelu`, `leaky_relu`, `elu`
- Normalization: `softmax`, `rms_norm`, `layer_norm`, `logsumexp`
- Matrix multiplication: tiled GEMM with FMA microkernels
- Distances: dot product, squared Euclidean, Manhattan, and cosine for `f32` and `f64`, behind `cdist`, `pdist`, and `numr::distance`
- Special functions: `erf`, `erfc`, Bessel, and gamma, with polynomial approximations

## Graph Capture

The `Graph` trait captures a sequence of operations and replays it without re-launch overhead.

- CUDA: full CUDA Graph capture. Replay uses fixed-address buffers for inference loops and training steps.
- CPU and WebGPU: a no-op path (`NoOpGraph`).

Check `R::supports_graph_capture()` before you depend on capture. Write backend-agnostic code with it.

## Native Kernels

numr uses native kernels (SIMD, PTX, WGSL) by default. Native kernels keep the code inspectable and
portable across NVIDIA, AMD, Intel, and Apple GPUs. You can replace individual kernels with
vendor-library calls. See [Extending numr](extending-numr.md).

## CUDA Build

The `cuda` feature needs CUDA 12.8 or newer, or CUDA 13.x. `build.rs` always emits `compute_120` PTX.
The GPU needs compute capability 7.5 (Turing) or newer.

`build.rs` compiles every kernel into a multi-arch fatbin and embeds it in the binary. The binary reads
no kernel files from disk at run time.

`NUMR_CUDA_ARCH` selects which architectures get native SASS:

| Value | SASS built for |
| ----- | -------------- |
| unset | The GPU or GPUs on the build host, detected with `nvidia-smi` |
| unset, no GPU detected | Every supported arch, same as `all` |
| `86`, `sm_86`, `8.6`, or `86,89,90` | Exactly the listed archs |
| `all` or `portable` | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_120` |

Every mode also embeds `compute_75` and `compute_120` PTX. A GPU with no matching SASS JIT-compiles that PTX on first load.

Set `NUMR_CUDA_ARCH=all` for any binary you redistribute. That covers release builds, Docker images,
and binaries built on one machine and run on another. The unset default fits only the build host's GPU.

`NUMR_CUDA_TUNE` set to `0`, `false`, or `off` turns off per-device schedule tuning at run time.
The fixed defaults apply instead.
