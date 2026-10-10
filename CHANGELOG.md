# Changelog

All notable changes to numr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
numr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The 0.7.0 entry covers everything up to and including 0.7.0. Tags `v0.1.0`
through `v0.6.1` predate this file and are folded into it rather than
reconstructed, so nothing in that entry is stated as a delta against an earlier
version. Every later entry is a delta against the one below it.

---

## [Unreleased]

A public `numr::distance` API on plain slices. SIMD kernels back it, and CPU `cdist` and `pdist` now use them.

### Added

- **`numr::distance`** — a public API on plain slices. It needs no `Tensor` or `Runtime`.
  - Float functions: `dot`, `l2_squared`, `manhattan`, and `cosine_distance`. Each has an `_f32` and an `_f64` twin, such as `dot_f32` and `cosine_distance_f64`.
  - One query against many rows: the `*_many_*` twins read a contiguous row-major buffer, such as `dot_many_f32(query, rows, d, out)`.
  - Int8: `dot_i8`, `dot_i8_scaled`, and `dot_i8_many`. The sum is exact and a total outside the i32 range saturates once at the end.
  - A length mismatch panics. The message names the function and both lengths.
  - Cosine distance of a zero vector returns 0. For unit-length vectors, cosine distance equals `1 - dot`.
- **`Kernels` handle** — `Kernels::detect()`, `Kernels::scalar()`, and `Kernels::with_level(level) -> Option<Kernels>`. `with_level` returns `None` when the CPU does not support the level. A caller can hoist dispatch or force a level without `unsafe`. `numr::distance::SimdLevel` is re-exported. It is `#[non_exhaustive]` and has no ordering, so a later release can add a level without a breaking change.
- **SIMD distance kernels** — f32 and f64 `dot`, squared Euclidean, Manhattan, and cosine. Each has an AVX-512 path with a masked tail, an AVX2+FMA path, a NEON path, and a scalar fallback. NEON also serves NEON+FP16. Dispatch runs once per call on the cached `SimdLevel`. Each ISA keeps several independent accumulators to hide FMA latency.
- **`benches/distance.rs`** — a `cdist` benchmark for squared Euclidean, cosine, and Manhattan at d = 128, 384, 768, and 1536.

### Changed

- **Minimum Rust version** — 1.95, up from 1.89. A build with `--features f16` on aarch64 does not compile on 1.89. The aarch64 `f16` conversion kernels use NEON half-precision intrinsics that are stable only from 1.94.
- **CPU `cdist` and `pdist`** — Euclidean, squared Euclidean, Manhattan, and cosine route through the new kernels for F32 and F64.
  - They run in parallel across fixed work units. The unit size depends only on `d`, so the output bits are identical for every thread count.
  - A single query against many rows (`n = 1`) also runs in parallel.
  - F16, BF16, and FP8 inputs convert to F32 once, run on the F32 path, and convert back once.
- **Numerics** — the summation order of these metrics differs from the old sequential loop and differs between SIMD levels. Results can differ in the last bits.
  - Cosine keeps the `denom == 0 -> 0` rule.
  - The CPU accumulator width still matches CUDA: f32 for narrow floats and f64 for F64. The CPU no longer sums term for term in the same order as CUDA.
  - The CUDA and WebGPU parity tests for distance pass against the CPU path on an RTX 3060 (CUDA and Vulkan/WGPU).
- **Internal** — block and row kernels replace the serial crate-internal `cdist_kernel` and `pdist_kernel`. The public API does not change.

### Fixed

- **Cosine and correlation distance at extreme magnitudes** — the denominator was `sqrt(a * b)` of two squared norms, which overflowed f32 once `|a| * |b|` passed about 1.8e19 and flushed to zero below about 3.7e-23. `cosine_distance_f32(&[1e10], &[1e10])` returned 1 instead of 0, and `[1e-12]` against `[-1e-12]` returned 0 instead of 2. The denominator is now `sqrt(a) * sqrt(b)`. The CPU, CUDA, and WebGPU kernels all changed, for `cdist`, `pdist`, and `numr::distance`. Each squared norm must still fit the float type, so a component above about 1e19 (f32) or 1e154 (f64) still gives NaN. Results in the normal range differ from before only in the last bits.
- **Shape overflow** — a tensor shape whose element count overflows `usize` returned a wrapped, too-small allocation. `pdist` on a `[2^32 + 1, 0]` tensor then wrote past the end of the output. `Tensor::empty`, `Tensor::from_slice`, and the scalar and typed fill constructors now return `Error::InvalidArgument` when the count overflows. `pdist` and `squareform_inverse` do the same for the pair count.
- **x86-64 SIMD detection** — `SimdLevel::Avx512` now also requires AVX2. A virtual machine that hides AVX2 but reports AVX-512 no longer reaches AVX2 code.
- **Build without `rayon`** — `--no-default-features` no longer warns about two matmul items that only the rayon path uses.
- **aarch64 clippy** — `erf_f64` uses `FRAC_2_SQRT_PI` instead of a literal.

### Performance

User-space instructions per `cdist` call, measured with `perf stat -e instructions:u`. Setup: n = 1, m = 64, one thread, AVX2+FMA host. Each count includes about 3,000 instructions of tensor allocation and dispatch. These are instruction counts, not wall-clock times.

| Metric            | d    | Before    | After   |
| ----------------- | ---- | --------- | ------- |
| Squared Euclidean | 128  | 43,453    | 10,625  |
| Squared Euclidean | 384  | 121,265   | 19,317  |
| Squared Euclidean | 768  | 238,011   | 34,506  |
| Squared Euclidean | 1536 | 471,485   | 61,168  |
| Cosine            | 128  | 103,473   | 15,464  |
| Cosine            | 384  | 300,093   | 29,033  |
| Cosine            | 768  | 595,005   | 51,721  |
| Cosine            | 1536 | 1,184,828 | 93,534  |
| Manhattan         | 128  | 53,628    | 11,561  |
| Manhattan         | 384  | 151,932   | 22,313  |
| Manhattan         | 768  | 299,388   | 41,088  |
| Manhattan         | 1536 | 594,301   | 73,121  |

### Known limits

- The AVX-512 kernels compile and pass clippy. They have never run. The development host has no AVX-512, and QEMU TCG does not emulate it.
- A separate read of the code traced the AVX-512 loop bounds, tail masks, and feature requirements by hand.
- The NEON kernels ran only under `qemu-aarch64` (user-mode emulation). They have not run on real ARM hardware.
- Instruction counts for NEON and AVX-512 are not measured.

---

## [0.9.0] — 2026-10-09

The WebGPU backend moves to wgpu 30.

### Breaking

- **wgpu 30** — the public WebGPU API carries wgpu 30 types. This covers `WgpuClient::wgpu_device`, `wgpu_device_arc`, `wgpu_queue`, and `submit_and_wait`, plus `WgpuDevice::backend` and `limits`. A caller that passes or receives these types must depend on wgpu 30.

### Changed

- **Dependencies** — wgpu 30 and pollster 1.0.

### Fixed

- **WebGPU readback** — a failed buffer mapping during a device-to-host read returns an error instead of panicking. `masked_select` no longer panics when the GPU count readback fails.

---

## [0.8.0] — 2026-10-09

Destination-passing ops and a capture-safe CUDA stream for graph replay. CUDA kernels ship as multi-arch fatbins inside the binary. New activation, padding, grouped-matmul, and transform ops.

### Breaking

- **`CapturedGraph::new_with_arena`** takes a fifth parameter, `arena_bytes_used: usize`: the arena's peak footprint at the end of capture.
- **`CudaClient::stream()`** returns `&GuardedStream` instead of `&CudaStream`. Build launches with `GuardedStream::launch_builder`. `GuardedStream::raw()` returns the bare `CudaStream`.
- **New required trait methods** — `BinaryOps::copy_into`, `ShapeOps::pad_mode`, and `MatmulOps::matmul_wide` have no default body. A downstream implementor of these traits must add them.
- **`numr::runtime::cuda::kernels::launch_gemv_kernel_bt`** is removed. F32 and integer small-M matmul go through `MatmulOps::matmul`.
- **`kernel_names::NORM_MODULE` and `FUSED_ADD_NORM_MODULE`** are removed. Norm kernels ship as one module per op: `NORM_RMS_MODULE`, `NORM_LAYER_MODULE`, `NORM_GROUP_MODULE`, `FUSED_ADD_RMS_NORM_MODULE`, `FUSED_ADD_LAYER_NORM_MODULE`.
- **`kernel_names::GEMV_INT_MODULE`** is removed with the integer GEMV kernel.

### Added

- **Destination-passing ops** — `BinaryOps::copy_into`, `HostCopyOps::write_host_slice`, and `RandomOps::{rand_into, rand_seeded_into, randn_into, randn_seeded_into}` write into a caller-owned tensor. The buffer keeps its device address, so a captured CUDA graph replays against it. On CUDA each one records as a graph node.
- **CUDA graph capture lock** — `numr::runtime::cuda::capture` adds `DeviceCaptureLock`, `CapturePermit`, `EnqueuePermit`, `thread_is_capturing`, `GuardedStream`, and `GuardedLaunchBuilder`. A capture holds the device's write lock for its whole region. Each enqueue holds the read side for that one enqueue. Threads that share a device no longer leak launches into another thread's graph.
- **Capture queries** — `CudaClient::is_capturing()` answers for the calling thread. `CudaClient::stream_capture_active()` reports the driver's capture state on the stream.
- **Typed arena exhaustion** — `Error::ArenaExhausted { requested, used, capacity }` replaces a formatted string. `Error::as_arena_exhausted()` returns the three byte counts, unwrapping one `AllocFailed` layer. `CapturedGraph::arena_bytes_used()` reports the arena's peak, so a caller can size the next capture.
- **Activations** — `ActivationOps::gelu_erf`, `gelu_erf_mul`, and `gelu_erf_mul_bwd` compute exact erf GELU, matching PyTorch `F.gelu(approximate="none")`. `snake_beta` and `snake_beta_bwd` compute Snake with per-channel `alpha` and `beta`. All five run on CPU, CUDA, and WebGPU, with autograd wrappers.
- **Padding** — `PadMode::{Constant, Reflect}` and `ShapeOps::pad_mode`. `Reflect` matches PyTorch `F.pad(mode="reflect")`. CPU, CUDA, and WebGPU each have a native reflect kernel.
- **Matmul** — `MatmulOps::matmul_wide` returns the F32 accumulator for F16 and BF16 inputs, so a caller that keeps summing rounds once. `GroupedMatmulOps::{grouped_matmul, grouped_matmul_activation}` run one matmul per row group, with group offsets held on the device. Grouped matmul covers F32, F16, and BF16 on CPU and CUDA.
- **Transforms and norms** — `FwhtOps::fwht` is a block-wise, normalized Walsh-Hadamard transform on CPU, CUDA, and WebGPU. `NormalizationOps::l2_normalize` divides by `max(‖x‖₂, eps)` along one dim.
- **Device profiles** — `DeviceProfile`, `DeviceArch`, and `DeviceCaps` describe SM count, shared memory, and tensor-core support (`bf16`, `int8`, `fp8`, and the int8 MMA shape). `Device::profile()` defaults to a conservative unknown profile. `CudaDevice` queries the driver once per device and caches the result.
- **Context-free CUDA queries** — `CudaDevice::count()`, `CudaDevice::product_name()`, and `CudaDevice::total_memory()` need no CUDA context. `count()` returns 0 when the driver library is absent.
- **CUDA schedule tuning** — `numr::runtime::cuda::tune` (`tuned`, `time_launches`) picks between launch alternatives that produce identical bits. It probes once per device and key, then caches the winner. Probes never run inside a graph capture. Set `NUMR_CUDA_TUNE` to `0`, `false`, or `off` to use the fixed defaults.
- **Embedded multi-arch fatbins** — `build.rs` compiles every kernel to a fatbin with native SASS per arch, plus `compute_75` and `compute_120` PTX for JIT. The fatbins are embedded with `include_bytes!`, so the binary reads no kernel file at run time.
- **`NUMR_CUDA_ARCH`** — selects the SASS archs: unset builds for the host GPU, a list such as `86,89` builds those archs, and `all` builds every supported arch. Use `all` for redistributable binaries.
- **CUDA kernels** — tensor-core WMMA GEMM with fused bias, activation, and residual epilogues, at several tile sizes. im2col + GEMM paths for `conv1d`, `conv2d`, and `conv_transpose1d`. A column-blocked `depthwise_conv2d`. A tiled transpose for strided copies. Serial and split kernels for dim reductions. A small-M F32 kernel for transposed weights.

### Changed

- **CUDA requirement** — the `cuda` feature needs CUDA 12.8 or newer, because the build always emits `compute_120` PTX. CUDA 12.8+ and 13.x both work.
- **CPU transcendental accuracy** — f64 `exp`, `exp2`, `expm1`, `sin`, `cos`, `tan`, and the hyperbolic family reach double precision. The f64 log family stays within 2 ulps. f32 `exp`, `exp2`, `expm1`, `cbrt`, `log`, and the hyperbolic family reach single precision. f32 `sin`, `cos`, `tan`, `atan`, `asin`, and `acos` stay within 2 ulps. f64 `exp`, `sinh`, and `cosh` cover the full representable domain.
- **CPU performance** — Rayon dispatch is gated on work size. Broadcast binary ops take a row-wise SIMD path. Strided copies coalesce dims and walk rows. FFT twiddle tables are cached. Strided `conv1d` is vectorized. The tiled F32 matmul reads a transposed B in place instead of copying it.
- **Dependencies** — the archived `paste` macro crate is replaced by its maintained fork `pastey`. The benchmarks use fluxbench 0.2.
- **CUDA performance** — RMSNorm uses a register-cached single pass with quad-packed loads. Norm kernels compile one fatbin per op. Softmax uses a flattened grid. Batched F32 GEMM and grouped F16/BF16 GEMM run on tiled and tensor-core kernels.

### Fixed

- **Determinism** — CPU element-wise kernels run per row, so a row's result no longer depends on how many rows share a call. The CPU matmul column split depends on shape only, never on thread count. CUDA conv kernel selection no longer depends on batch size.
- **Graph capture** — the allocator routes only the capturing thread to the arena. A failed `begin_capture` unfreezes the allocator. The CUDA index-bounds check and the Sobol warmup check no longer synchronize or misread state during another thread's capture. A missing Sobol cache entry returns an error instead of panicking.
- **Tuning races** — concurrent probes for one key run once, and the first cached value stays.
- **CUDA launch limits** — element-wise launches that overflow the grid and matmul launches over the shared-memory budget return an error instead of computing wrong elements.
- **Batch broadcasting** — `matmul_bias`, `matmul_bias_residual`, `fp8_matmul`, and `semiring_matmul` broadcast each operand's batch dims per index instead of wrapping.
- **Narrow-float accumulation** — CPU fused GEMM epilogues and every direct convolution kernel accumulate F16 and BF16 in F32, and F64 in F64.
- **Norm stability** — `layer_norm`, `group_norm`, and `fused_add_layer_norm` shift by a per-row reference before accumulating, on every backend. The WebGPU shaders keep that shift through compilation.
- **Special functions** — `bessel_i0` and `bessel_i1` no longer overflow early in the asymptotic branch. SIMD `pow` honors IEEE 754 identities for non-finite exponent products. f64 `pow` falls back to scalar for non-positive bases. CPU f64 softmax and logsumexp, and the CUDA softmax merge, handle all-masked rows. NEON f64 `gelu_mul` saturates `tanh` instead of dividing `exp(2x)`.
- **SIMD dispatch** — f64 AVX-512 kernels that use bitwise ops require AVX-512DQ, so AVX-512F-only CPUs no longer trap.
- **Autograd** — `conv1d` and `conv2d` input gradients are correct when input and output channel counts differ.
- **WMMA selection** — BF16 WMMA runs only on devices with native BF16 tensor cores (sm_80+). Padding and launch read the same predicate.
- **`dispatch_dtype!`** — a wildcard arm returns `Error::UnsupportedDType`, so downstream crates compile against the `#[non_exhaustive]` `DType`.
- **Feature gating** — builds without `f16` keep the non-half code in the clamp and `where` kernels.

---

## [0.7.0] — 2026-08-30

Tensors, linear algebra, FFT, and autograd behind one API on CPU, CUDA, and WebGPU.
Every kernel is written in-house — no cuBLAS, cuSOLVER, or MKL.

### Added

- **Tensors** — `Tensor<R>` generic over the backend. Broadcasting, zero-copy views (`reshape`, `transpose`, `slice`, `permute`), fallible constructors that never panic on allocation failure.
- **Backends** — CPU (`#[target_feature]` AVX-512 / AVX2+FMA / NEON, optional Rayon), CUDA (graph capture, arena allocator, kernel cache), WebGPU (WGSL per dtype). CPU and CUDA cover every dtype; WebGPU is F32 / I32 / U32 / Bool.
- **DTypes** — F64, F32, F16, BF16, FP8E4M3, FP8E5M2, I64, I32, I16, I8, U64, U32, U16, U8, Bool. Narrow dtypes accumulate in a wider type. WebGPU is F32 / I32 / U32 / Bool.
- **Element-wise and reductions** — unary, binary, scalar, compare, logical, conditional. Reductions over any axis set, cumulative ops, sorting with a NaN-aware total order, gather/scatter indexing.
- **Matmul** — batched and broadcast, tiled CPU kernel with a transposed-B path, fused GEMM epilogue, FP8, min-plus/max-plus semirings, einsum, 2:4 structured sparsity.
- **Linear algebra** — LU, QR, SVD, Cholesky, Schur, QZ, polar, eigen. Solvers, `lstsq`, inverse, `slogdet`, `cond`, `matrix_rank`. Matrix functions (`expm`, `logm`, `sqrtm`, `signm`, `funm`). Tucker, HOSVD, CP, tensor-train.
- **Iterative solvers** — CG, BiCGSTAB, CGS, GMRES, LGMRES, MINRES, QMR, Jacobi, SOR, sparse eigensolvers, `svds`, AMG.
- **FFT** — 1D / 2D / ND, forward and inverse. Bluestein covers arbitrary sizes on all three backends.
- **Convolution** — `conv1d`, `conv2d`, `depthwise_conv2d`, `conv_transpose1d` with native kernels and autograd on each.
- **Autograd** — reverse mode via `Var<R>` and `GradFn`, forward mode via `DualTensor`, gradient checkpointing. `backward` takes a needed-gradient mask so ops skip gradients the driver discards.
- **Random** — uniform, normal, seeded normal, advanced and multivariate distributions, Sobol sequences, Philox on GPU. Seeds reproduce per backend, not across backends.
- **Statistics and special functions** — quantiles, histograms, correlation, distance metrics, polynomials. Gamma, Bessel, Airy, Fresnel, elliptic, hypergeometric, orthogonal families.
- **Sparse** (`sparse`) — CSR, CSC, COO with conversions, element-wise ops, SpMM, SpMV, and sparse LU / QR with COLAMD ordering.
- **Distributed** (`distributed`, `nccl`) — a `Communicator` trait over NCCL, nexar, hierarchical, and no-op backends, plus process groups.

### Semantics

Behaviour a caller must know. These are contracts, not defaults.

- **Output dtype is a function of input dtypes, never input values.** `pow_scalar` on an integer tensor returns F64 unless the exponent is a whole non-negative number. `matmul_output_dtype` is public so a caller can size a bias against it.
- **Integer accumulators saturate** at the dtype's bound: `sum`, `prod`, `mean`, `cumsum`, `cumprod`, `matmul`, `scatter_reduce`.
- **Integer element-wise ops wrap**: `add`, `sub`, `mul`, and the fused forms.
- **Integer division by zero returns 0**, and `INT_MIN / -1` returns `INT_MIN`. Neither panics.
- **I8 matmul widens to I32.** `matmul_bias` on I8 takes an I32 bias and returns I32. The GEMM epilogue ops (`matmul_bias_activation`, `matmul_bias_residual`, and the backward form) reject I8 rather than read a wider bias buffer as I8.

### Notes

- `tests/backend_parity/` checks CUDA and WebGPU against CPU for every operation, at dtype-appropriate tolerances.
- WebGPU stays 32-bit by design. WGSL has no native F64.
- ROCm is planned.
