<div align="center">

# numr

<h3>Numerical computing in Rust. Write once. Run on any backend.</h3>

<p>
N-dimensional tensors, linear algebra, FFT, statistics, and autograd on CPU, CUDA, and WebGPU.
The backends use native kernels, with no cuBLAS, MKL, or other vendor math library.
</p>

<p>
  <a href="https://docs.rs/numr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/numr"><strong>Crate</strong></a>
  ·
  <a href="#operations"><strong>Operations</strong></a>
  ·
  <a href="#backends"><strong>Backends</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/numr/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/numr/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="https://crates.io/crates/numr">
    <img src="https://img.shields.io/crates/v/numr" alt="crates.io">
  </a>
  <a href="https://docs.rs/numr">
    <img src="https://img.shields.io/docsrs/numr" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/numr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/numr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/numr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

numr is the foundation library for numerical code in Rust. Tensors are generic over a `Runtime`.
Switching from CPU to CUDA or WebGPU changes the runtime type. Generic functions keep their body.

## What numr Is

numr provides building blocks for higher-level libraries and applications.

| numr IS | numr is NOT |
| ------- | ----------- |
| Tensor library (like NumPy's ndarray) | A deep learning framework |
| Linear algebra (decompositions, solvers) | A high-level ML API |
| FFT, statistics, random distributions | Domain-specific |
| Native GPU (CUDA + WebGPU) and autograd | |

For optimization, ODE solvers, interpolation, and signal processing, use [solvr](https://github.com/ml-rust/solvr).

## Installation

```bash
cargo add numr
```

For GPU support, add a backend feature:

```bash
cargo add numr --features cuda   # NVIDIA GPUs, CUDA 12.8+ or 13.x
cargo add numr --features wgpu   # WebGPU: Vulkan, Metal, and DX12 GPUs
```

Feature flags:

| Feature | Description | Default |
| ------- | ----------- | ------- |
| `rayon` | Multi-threaded CPU through Rayon | ✓ |
| `cuda` | NVIDIA CUDA backend | ✗ |
| `wgpu` | Cross-platform GPU through WebGPU | ✗ |
| `f16` | F16 and BF16 types | ✗ |
| `fp8` | FP8 types (E4M3, E5M2) | ✗ |
| `sparse` | Sparse tensors (CSR, CSC, COO) and sparse linear algebra | ✗ |
| `nccl` | Multi-GPU collectives through NCCL (implies `cuda`) | ✗ |
| `distributed` | Multi-machine communication through nexar | ✗ |
| `distributed-gpu` | NCCL within a node, nexar between nodes (implies `distributed` and `nccl`) | ✗ |

The CPU backend needs no feature flag. The CUDA build details are in [Backends](docs/backends.md#cuda-build).

## Quick Start

### CPU

```rust
use numr::prelude::*;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

fn main() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2], &device)?;

    // Arithmetic with broadcasting
    let c = a.add(&b)?;
    let d = a.mul(&b)?;

    // Matrix multiplication
    let e = a.matmul(&b)?;

    // Reductions: dims, keepdim
    let sum = c.sum(&[0, 1], false)?;
    let max = c.max(&[0, 1], false)?;

    // Element-wise functions
    let exp = a.exp()?;
    let sqrt = a.sqrt()?;

    // Reshape and transpose
    let flat = c.reshape(&[4])?;
    let transposed = c.transpose(0, 1)?;

    println!("{:?}", c.to_vec::<f32>());
    Ok(())
}
```

### Backend-Generic Code

Write functions over `R: Runtime`. The same function runs on any backend.

```rust
use numr::prelude::*;

fn matrix_operations<R>(a: &Tensor<R>, b: &Tensor<R>, client: &R::Client) -> Result<Tensor<R>>
where
    R: Runtime,
    R::Client: BinaryOps<R> + MatmulOps<R> + ReduceOps<R>,
{
    let c = client.add(a, b)?;
    let d = client.matmul(&c, a)?;
    client.sum(&d, &[0, 1], false)
}
```

### GPU (CUDA)

```rust
use numr::ops::RandomOps;
use numr::prelude::*;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let device = CudaRuntime::default_device();
    let client = CudaClient::new(device.clone())?;

    let a = client.randn(&[1024, 1024], DType::F32)?;
    let b = client.randn(&[1024, 1024], DType::F32)?;

    // Runs on the GPU through native CUDA kernels
    let c = client.matmul(&a, &b)?;

    // Copy the result to the host
    let host = c.try_to_vec::<f32>()?;
    Ok(())
}
```

The module docs cover the rest of the API:

- Linear algebra, FFT, and statistics: [Algorithms](docs/algorithms.md) and [Operations](docs/operations.md)
- Gradients: [Autograd](docs/autograd.md)

## Operations

numr provides tensor operations as traits in `numr::ops`. The same traits cover CPU, CUDA, and WebGPU.
The full list is in [docs/operations.md](docs/operations.md):

- Core arithmetic: unary, binary, scalar, type conversion, utility
- Shape and data movement: `cat`, `stack`, `gather`, `scatter`, `sort`, `topk`, and more
- Reductions, cumulative operations, comparisons, and logic
- Activations, normalization, convolution, einsum, and fused GEMM epilogues
- Linear algebra and complex operations
- Statistics, random distributions, and quasi-random sequences
- Distance metrics on tensors and on plain slices

The algorithm modules are in [docs/algorithms.md](docs/algorithms.md):

- Linear algebra decompositions, matrix functions, and solvers
- FFT and real FFT
- Special functions and polynomials
- Iterative solvers and preconditioners
- Sparse tensors (CSR, CSC, COO) and sparse linear algebra

## Backends

| Runtime | Hardware | Feature | Kernels |
| ------- | -------- | ------- | ------- |
| `CpuRuntime` | x86-64 and ARM64 CPUs | `cpu` (default) | SIMD: AVX-512, AVX2, NEON |
| `CudaRuntime` | NVIDIA GPUs | `cuda` | Native CUDA kernels |
| `WgpuRuntime` | Vulkan, Metal, and DX12 GPUs | `wgpu` | WGSL compute shaders |

The WebGPU backend computes in `F32` only. The dtype table, SIMD details, graph capture, and the
`NUMR_CUDA_ARCH` variable are in [docs/backends.md](docs/backends.md).

## Documentation

| Page | Contents |
| ---- | -------- |
| [docs/operations.md](docs/operations.md) | Tensor operation traits and distance metrics |
| [docs/algorithms.md](docs/algorithms.md) | Decompositions, FFT, special functions, solvers, sparse |
| [docs/autograd.md](docs/autograd.md) | Reverse-mode and forward-mode autodiff, checkpointing |
| [docs/backends.md](docs/backends.md) | Backends, dtypes, SIMD, graph capture, CUDA build |
| [docs/distributed.md](docs/distributed.md) | Communicators and parallel groups |
| [docs/extending-numr.md](docs/extending-numr.md) | Custom kernels and new operations |
| [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | Internal architecture for contributors |

## Building from Source

```bash
cargo build --release                    # CPU (default)
cargo build --release --features cuda    # CUDA
cargo build --release --features wgpu    # WebGPU
cargo test --release                     # Tests
cargo bench                              # Benchmarks
```

## How numr Fits in the Stack

numr is the base layer. [solvr](https://github.com/ml-rust/solvr) builds scientific algorithms on numr.
Code written against numr's `Runtime` gains new backends when numr adds them.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
