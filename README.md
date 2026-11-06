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
  <a href="#backends"><strong>Backends</strong></a>
  ·
  <a href="#installation"><strong>Install</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="#operations"><strong>Operations</strong></a>
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

numr is a tensor and numerical computing library for Rust. It fills the role NumPy has in Python, and it also runs on GPUs.
Tensors are generic over a `Runtime`.
Switching from CPU to CUDA or WebGPU changes the runtime type. Generic functions keep their body, and gain new backends when numr adds them.

- Pure Rust on CPU and WebGPU. No Python is needed. The CUDA backend compiles its `.cu` kernels with `nvcc` at build time.
- numr is the base of [solvr](https://github.com/ml-rust/solvr) (SciPy role) and [boostr](https://github.com/ml-rust/boostr) (ML framework).

Not supported yet:

- ROCm.
- WebGPU computes in `F32` only.

## Backends

| Runtime | Hardware | Feature | Needs | Dtypes |
| ------- | -------- | ------- | ----- | ------ |
| `CpuRuntime` | x86-64 and ARM64 CPUs | None, always built | Nothing | All 15. `F16` and `BF16` need `f16`. FP8 types need `fp8` |
| `CudaRuntime` | NVIDIA GPUs, compute capability 7.5 (Turing) or newer | `cuda` | CUDA toolkit 12.8 or newer, with `nvcc` on `PATH` or `CUDA_PATH` set | Same as CPU |
| `WgpuRuntime` | Vulkan, Metal, and DX12 GPUs | `wgpu` | Platform GPU driver | `F32` for computation. `cast` runs on the GPU among `F32`, `I32`, `U32`, and `I64`. Other dtypes return `Error::UnsupportedDType` |

- Kernels: SIMD on CPU (AVX-512, AVX2, NEON), native CUDA kernels, and WGSL compute shaders. No cuBLAS, MKL, or other vendor library.
- ROCm: not implemented.
- Dtype coverage: the table lists what a backend holds. Check the operation before you rely on a dtype.
- CUDA binaries: a build targets the GPUs on the build host. Set `NUMR_CUDA_ARCH=all` for a binary that runs on another machine.
- Details: the dtype table, SIMD, graph capture, and every `NUMR_CUDA_ARCH` value are in [docs/backends.md](docs/backends.md). The CUDA build steps are in [CUDA Build](docs/backends.md#cuda-build).

## Installation

Needs Rust 1.95 or newer.

```bash
cargo add numr                   # CPU
cargo add numr --features cuda   # NVIDIA GPUs
cargo add numr --features wgpu   # WebGPU
```

Other feature flags:

| Feature | Description | Default |
| ------- | ----------- | ------- |
| `rayon` | Multi-threaded CPU through Rayon | ✓ |
| `f16` | F16 and BF16 types | ✗ |
| `fp8` | FP8 types (E4M3, E5M2) | ✗ |
| `sparse` | Sparse tensors (CSR, CSC, COO) and sparse linear algebra | ✗ |
| `nccl` | Multi-GPU collectives through NCCL (implies `cuda`) | ✗ |
| `distributed` | Multi-machine communication through nexar | ✗ |
| `distributed-gpu` | NCCL within a node, nexar between nodes (implies `distributed` and `nccl`) | ✗ |

## What numr Is

numr provides:

- Tensors: N-dimensional arrays
- Linear algebra: decompositions and solvers
- FFT, statistics, and random distributions
- Autograd

numr is not a deep learning framework, a high-level ML API, or domain-specific.

| You need | Use |
| -------- | --- |
| Optimization, ODE solvers, interpolation, signal processing | [solvr](https://github.com/ml-rust/solvr) |
| Attention, model architectures, quantization, training | [boostr](https://github.com/ml-rust/boostr) |

## Quick Start

### CPU

```rust
use numr::prelude::*;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

fn main() -> Result<()> {
    let device = CpuDevice::new();

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
- Activations, normalization, convolution, einsum, and matmul with fused bias and activation
- Linear algebra and complex operations
- Statistics, random distributions, and quasi-random sequences
- Distance metrics on tensors and on plain slices

The algorithm modules are in [docs/algorithms.md](docs/algorithms.md):

- Linear algebra decompositions, matrix functions, and solvers
- FFT and real FFT
- Special functions and polynomials
- Iterative solvers and preconditioners
- Sparse tensors (CSR, CSC, COO) and sparse linear algebra

## For NumPy Users

Most numr operations are methods on the backend client, such as `client.sum(&a, &[0], false)`.
Import `numr::prelude::*`. The traits named below are in the prelude unless a row says otherwise.

| NumPy | numr |
| ----- | ---- |
| `np.array(data).reshape(shape)` | `Tensor::from_slice(&data, &shape, &device)` |
| `np.zeros`, `np.ones` | `Tensor::zeros`, `Tensor::ones` |
| `np.full` | `Tensor::full_scalar` |
| `np.arange`, `np.linspace`, `np.eye` | `UtilityOps`: `arange`, `linspace`, `eye` |
| `np.meshgrid` | `UtilityOps::meshgrid` |
| `a + b`, `a * b` with broadcasting | `a.add(&b)`, `a.mul(&b)`, or `BinaryOps` |
| `np.exp`, `np.sqrt`, `np.sin`, `np.abs` | `a.exp()`, `a.sqrt()`, `a.sin()`, `a.abs()`, or `UnaryOps` |
| `np.clip` | `UtilityOps::clamp` |
| `np.isnan` | `UnaryOps::isnan` |
| `a @ b`, `np.matmul` | `a.matmul(&b)`, or `MatmulOps::matmul` |
| `np.einsum` | `EinsumOps::einsum` |
| `a.reshape`, `a.T`, `np.transpose`, `np.squeeze` | `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze` |
| `np.concatenate`, `np.stack` | `Tensor::cat`, `Tensor::stack` |
| `np.split`, `np.repeat`, `np.roll`, `np.pad` | `ShapeOps`: `split`, `chunk`, `repeat`, `roll`, `pad` |
| `a.sum`, `a.mean`, `a.max`, `a.min` | `ReduceOps`: `sum`, `mean`, `max`, `min` |
| `np.argmax`, `np.argmin` | `IndexingOps::argmax`, `argmin` |
| `np.cumsum`, `np.cumprod`, `scipy.special.logsumexp` | `CumulativeOps`: `cumsum`, `cumprod`, `logsumexp` |
| `np.sort`, `np.argsort`, `np.unique`, `np.searchsorted` | `SortingOps`: `sort`, `argsort`, `unique`, `searchsorted` |
| `np.nonzero`, `np.where(cond, x, y)` | `SortingOps::nonzero`, `ConditionalOps::where_cond` |
| `np.take_along_axis` | `IndexingOps::gather` |
| `np.var`, `np.std`, `np.median`, `np.percentile`, `np.histogram` | `StatisticalOps`: `var`, `std`, `median`, `percentile`, `histogram` |
| `np.cov`, `np.corrcoef` | `StatisticalOps::cov`, `corrcoef` |
| `np.random.rand`, `randn`, `randint`, `permutation` | `RandomOps`: `rand`, `randn`, `randint`, `randperm` |
| `np.random.seed` | `numr::random::manual_seed` |
| `np.linalg.solve`, `inv`, `det`, `lstsq` | `LinalgOps`: `solve`, `inverse`, `det`, `lstsq` |
| `np.trace`, `np.kron`, `np.tril` | `LinalgOps`: `trace`, `kron`, `tril` |
| `np.linalg.svd`, `qr`, `cholesky`, `eig` | `numr::algorithm::linalg::LinearAlgebraAlgorithms`: `svd_decompose`, `qr_decompose`, `cholesky_decompose`, `eig_decompose` |
| `np.fft.fft`, `rfft`, `fftfreq` | `FftAlgorithms`: `fft`, `rfft`, `fftfreq` |

NumPy has no equivalent for autograd, CUDA and WebGPU execution, or the fused ML operations. Those are in [Operations](#operations).

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

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
