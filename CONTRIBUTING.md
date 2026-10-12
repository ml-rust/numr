# Contributing

Thanks for contributing to [numr](https://crates.io/crates/numr). This guide covers the architecture conventions and quality gates.

## Before you start

- Rust 1.95 or newer (edition 2024).
- Optional: CUDA 12.8+ or 13.x, or a WebGPU-capable device, to run GPU tests locally.
- A clean working tree before you open a pull request.

Useful contributions:

- **Missing primitives**: a tensor operation, dtype, or linear-algebra routine that NumPy has and numr lacks.
- **Backend coverage**: a CUDA or WebGPU implementation of an operation that runs only on CPU.
- Bug fixes and numerical-accuracy improvements.

Open an issue before you write a non-trivial operation. Describe the addition and the crate it belongs in. Small, self-contained fixes can go straight to a pull request.

### Which crate

Place work by what it is:

| Crate | Scope |
| --- | --- |
| **numr** (this crate) | Foundational primitives: tensor ops, dtypes, the `Runtime`/backend abstraction and new backends, FFT, core linear algebra (matmul, LU/QR/SVD/eigen, `solve`), special functions, basic descriptive statistics. |
| [solvr](https://github.com/ml-rust/solvr) | Scientific and solving algorithms built on numr: optimization, ODE/DAE/BVP/PDE, interpolation, advanced statistics, signal processing, spatial, clustering. |
| [boostr](https://github.com/ml-rust/boostr) | AI/ML building blocks: attention, positional encodings, mixture-of-experts, quantization, neural-network layers, training and inference machinery. |

Quick test:

- Reused across domains, or touches a hardware backend: **numr**.
- A domain solver a scientist or engineer reaches for: **solvr**.
- Makes sense only for neural networks or LLMs: **boostr**.

When in doubt, propose the work in an issue.

## Architecture

numr returns the same numbers on every backend. Each backend reaches those numbers its own way.

### Primitive and composite operations

**Primitive** operations are atomic. The kernel is the algorithm, such as `add`, `exp`, `sum`, or `matmul`. Each backend gets its own kernel, and the kernels differ:

```
ops/traits/unary.rs   trait definition
ops/cpu/unary.rs      impl → calls cpu/kernels/unary.rs   (SIMD)
ops/cuda/unary.rs     impl → launches cuda/kernels/unary.cu (PTX)
ops/wgpu/unary.rs     impl → dispatches wgpu/shaders/unary.wgsl (WGSL)
```

**Composite** operations are built from primitives, such as `softmax`, `layernorm`, or `gelu`. Write each one once in `ops/impl_generic/`. Every backend delegates to it. Use this pattern first.

```rust
// ops/impl_generic/activation.rs
pub fn softmax_impl<R, C>(client: &C, input: &Tensor<R>, dim: i64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    let max = client.max(input, &[dim], true)?;
    let shifted = client.sub(input, &max)?;
    let exp = client.exp(&shifted)?;
    let sum = client.sum(&exp, &[dim], true)?;
    client.div(&exp, &sum)
}
```

Replace `impl_generic` with a fused kernel only when profiling shows the decomposed version is too slow. This usually means a hot path that makes several passes over memory. A fused kernel does not match `impl_generic` bit-for-bit. It must pass the backend parity tests.

### Backend parity tests

Every operation needs a test in `tests/backend_parity/`. The test uses CPU as the reference and compares every other backend against it.

```rust
let cpu_result = cpu_client.softmax(&input_cpu, dim)?;
let cuda_result = cuda_client.softmax(&input_cuda, dim)?;
assert_tensor_allclose(&cuda_result, &cpu_result, dtype, "softmax cuda vs cpu");
```

Cover the edge cases that diverge. These include empty tensors, single-element and size-1 dimensions, very large and very small magnitudes, NaN, and infinities. A test that checks only well-behaved values passes while the operation is wrong.

A value-only test misses a shape divergence, such as `[1]` where CPU returns a scalar `[]`. Assert shapes for rank-reducing operations.

### File organization

- `mod.rs` contains only `pub mod` and `pub use`. A `mod.rs` over about 30 lines holds code in the wrong place.
- One operation is one file. Use the same file name under `traits/` and each backend directory.
- Backends live in directories (`cpu/`, `cuda/`, `wgpu/`), never in a single `cpu.rs`. Kernels go in `cpu/kernels/`, `cuda/kernels/`, and `wgpu/shaders/`.
- Add files for a new operation. Do not grow existing files.

| File type | Soft limit | Hard limit |
| --- | --- | --- |
| Trait files | 100 | 200 |
| Backend impls | 200 | 400 |
| CPU kernels and `.cu` files | 300 | 500 |
| WGSL shaders | 150 | 300 |
| Integration tests | 400 | 600 |

Split a file at its soft limit when a logical boundary exists.

### Backend rules

- **No GPU↔CPU transfers.** Host/device transfers cost far more than the computation. In GPU code, never call `tensor.to_vec()`, loop over device data on the host, or hide a transfer in a helper.
- If a GPU kernel does not exist, return `Err(Error::Unsupported { .. })`. Do not fall back to CPU.
- **No vendor libraries.** numr builds and runs without cuBLAS, cuDNN, MKL, or other vendor libraries. Kernels are native: PTX for CUDA, WGSL for WebGPU, SIMD intrinsics for CPU.
- **Every supported dtype.** Implement each dtype the backend supports, not only `F32`. Use `dispatch_dtype!` for runtime dispatch.
- WebGPU supports only `F32`, `I32`, and `U32`. Return an `UnsupportedDType` error for other dtypes. Do not degrade precision silently.

### CPU SIMD

Do not call `is_x86_feature_detected!()` inside a hot loop. Detect once, then dispatch to functions annotated for each ISA:

```rust
pub unsafe fn my_kernel(data: *const f32, len: usize, level: SimdLevel) -> f32 {
    match level {
        SimdLevel::Avx512 => my_kernel_avx512(data, len),
        SimdLevel::Avx2Fma => my_kernel_avx2(data, len),
        _ => my_kernel_scalar(data, len),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn my_kernel_avx512(data: *const f32, len: usize) -> f32 { /* ... */ }
```

Use two or more accumulators in FMA loops to hide the 4–5 cycle latency. Always provide a scalar fallback.

## Build

```bash
cargo build --release                    # CPU (default)
cargo build --release --features cuda    # CUDA (requires a CUDA 12.8+ or 13.x toolchain)
cargo build --release --features wgpu    # WebGPU
cargo build --release --features sparse  # sparse tensors
```

## Testing

- Put unit tests in the same file as the code, in `#[cfg(test)] mod tests`. They can reach private functions.
- Put public-API tests in `tests/`. Each subdirectory needs a `mod.rs`, so Cargo does not compile it as a separate test binary.
- Test numerical correctness against a reference value, not only that the call returns `Ok`.
- A backend-specific test skips gracefully when no device is present.
- Register each new module under `tests/backend_parity/` in `tests/backend_parity/mod.rs`. An unregistered module never compiles, and the suite still reports green.

```bash
cargo test --release
cargo test --release --features cuda
cargo test --release --features wgpu
```

## Quality checks

Run these before you submit. CI runs clippy with `-D warnings`, so treat a warning as an error.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```

If you touch a GPU backend, run clippy and the tests with `--features cuda` and `--features wgpu`.

## Pull requests

- Keep each pull request focused and scoped.
- Keep the module structure. Add files instead of growing existing ones.
- Include backend parity tests for each new or changed operation.
- Avoid `.unwrap()` and `.expect()` in library code. Return a typed error with the operation, shapes, and dtype.
- Update docs when public APIs or features change.

## Commit messages

Use Conventional Commits with a clear, imperative summary:

```
feat(linalg): add banded LU factorization for CPU and CUDA
fix(sort): apply a NaN-aware total order across all backends
```
