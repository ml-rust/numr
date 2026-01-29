# numr

**High-performance numerical computing for Rust with multi-backend GPU acceleration.**

numr is a numerical computing library that provides n-dimensional arrays (tensors), linear algebra, FFT, and automatic differentiation - with the same API across CPU, CUDA, and WebGPU backends.

## Why numr?

### vs ndarray

[ndarray](https://github.com/rust-ndarray/ndarray) is CPU-only. numr runs on CPU, CUDA, and WebGPU with the same code.

```rust
// Same code, different hardware
let result_cpu = a.matmul(&b)?;           // CPU
let result_gpu = a_cuda.matmul(&b_cuda)?; // NVIDIA GPU
let result_web = a_wgpu.matmul(&b_wgpu)?; // Any GPU via WebGPU
```

### vs faer / nalgebra

[faer](https://github.com/sarah-ek/faer-rs) and [nalgebra](https://nalgebra.org/) are excellent CPU linear algebra libraries. numr provides:

- **GPU backends**: CUDA, WebGPU - not just CPU
- **Automatic differentiation**: Built-in autograd for gradients
- **Sparse tensors**: CSR, CSC, COO formats with GPU support
- **N-dimensional**: True tensors, not just matrices

### vs numpy bindings (numpy, ndarray-npy)

Python bindings require a Python runtime. numr is pure Rust:

- **No Python dependency**: Single binary deployment
- **No GIL**: True parallelism
- **No FFI overhead**: Native Rust performance
- **Compile-time safety**: Catch errors before runtime

### vs cupy / torch bindings

[cupy](https://cupy.dev/) is CUDA-only. PyTorch bindings require the PyTorch runtime. numr provides:

- **Multiple GPU backends**: CUDA and WebGPU (works on NVIDIA, AMD, Intel, Apple)
- **Native kernels**: Not wrappers around cuBLAS/MKL
- **Lightweight**: No 2GB PyTorch installation
- **Same algorithm everywhere**: Identical results across backends

## Features

- **Tensors**: N-dimensional arrays with broadcasting, slicing, views
- **Linear algebra**: Matmul, LU, QR, SVD, Cholesky, eigendecomposition
- **FFT**: Fast Fourier transforms (1D, 2D, ND)
- **Element-wise operations**: Full set of math functions
- **Reductions**: Sum, mean, max, min, argmax, argmin along axes
- **Autograd**: Reverse-mode automatic differentiation
- **Sparse tensors**: CSR, CSC, COO formats (with `sparse` feature)
- **Multiple dtypes**: f64, f32, f16, bf16, fp8, i64, i32, i16, i8, u64, u32, u16, u8, bool

## Backends

| Hardware   | Backend                  | Feature Flag    | Status  |
| ---------- | ------------------------ | --------------- | ------- |
| Any CPU    | CPU                      | `cpu` (default) | ✅      |
| NVIDIA GPU | CUDA                     | `cuda`          | ✅      |
| NVIDIA GPU | WebGPU                   | `wgpu`          | ✅      |
| AMD GPU    | WebGPU                   | `wgpu`          | ✅      |
| Intel GPU  | WebGPU                   | `wgpu`          | ✅      |
| Apple GPU  | WebGPU                   | `wgpu`          | ✅      |
| AMD GPU    | ROCm (native)            | -               | Planned |
| Apple GPU  | Metal (native)           | -               | Planned |
| CPU        | SIMD (AVX-512/AVX2/NEON) | -               | Planned |

All backends use **native kernels** - no cuBLAS, MKL, or other vendor libraries. This means:

- No proprietary dependencies to install
- Same algorithm produces same results on all hardware
- Works anywhere Rust compiles

## Quick Start

```rust
use numr::prelude::*;

// Create tensors
let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let b = Tensor::<CpuRuntime>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;

// Arithmetic
let c = &a + &b;
let d = &a * &b;
let e = a.matmul(&b)?;

// Reductions
let sum = c.sum()?;
let mean = c.mean()?;
let max = c.max()?;

// Reshaping (zero-copy)
let flat = c.reshape(&[4])?;
let transposed = c.transpose()?;
```

### GPU Example

```rust
use numr::prelude::*;

// Create on GPU
let device = CudaRuntime::default_device()?;
let a = Tensor::<CudaRuntime>::randn(&[1024, 1024], &device)?;
let b = Tensor::<CudaRuntime>::randn(&[1024, 1024], &device)?;

// Operations run on GPU
let c = a.matmul(&b)?;

// Transfer to CPU when needed
let cpu_data = c.to_vec::<f32>()?;
```

### Backend-Generic Code

```rust
use numr::prelude::*;

// Works on any backend
fn compute<R: Runtime>(a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
    let c = a.add(b)?;
    let d = c.matmul(a)?;
    d.sum()
}

// Use the same function on CPU or GPU
let cpu_result = compute(&cpu_a, &cpu_b)?;
let cuda_result = compute(&cuda_a, &cuda_b)?;
```

## Installation

```toml
[dependencies]
numr = "0.0.0"
```

With GPU support:

```toml
# NVIDIA CUDA
numr = { version = "0.0.0", features = ["cuda"] }

# Cross-platform GPU (WebGPU)
numr = { version = "0.0.0", features = ["wgpu"] }
```

## Feature Flags

| Feature           | Description                           |
| ----------------- | ------------------------------------- |
| `cpu` (default)   | CPU backend                           |
| `cuda`            | NVIDIA CUDA backend                   |
| `wgpu`            | Cross-platform GPU via WebGPU         |
| `rayon` (default) | Multi-threaded CPU operations         |
| `f16`             | Half-precision floats (F16, BF16)     |
| `fp8`             | 8-bit floats (FP8E4M3, FP8E5M2)       |
| `sparse`          | Sparse tensor formats (CSR, CSC, COO) |

## Building on numr

numr is designed as a foundation. Libraries can build on numr to get optimized kernels for free:

- **solvr** - Scientific computing (optimization, ODE solvers, statistics)
- **boostr** - ML framework (neural network layers, attention mechanisms)
- **Your library** - Build what you need on numr tensors

When numr's kernels improve, everything built on it improves.

## Building from Source

```bash
cargo build --release                    # CPU only
cargo build --release --features cuda    # With CUDA
cargo build --release --features wgpu    # With WebGPU

cargo test                               # Run tests
cargo test --features cuda               # CUDA tests
cargo bench                              # Benchmarks
```

## License

Apache-2.0
