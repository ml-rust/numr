# numr

**Foundational numerical computing for Rust**

`numr` provides n-dimensional tensors, linear algebra, FFT, statistics, and automatic differentiation—with native GPU acceleration across CPU, CUDA, and WebGPU backends.

`numr` is like Numpy in Rust but built with gradients, GPUs, and modern dtypes built-in from day one.

## What numr Is

A **foundation library** - Mathematical building blocks for higher-level libraries and applications.

| numr IS                                  | numr is NOT               |
| ---------------------------------------- | ------------------------- |
| Tensor library (like NumPy's ndarray)    | A deep learning framework |
| Linear algebra (decompositions, solvers) | A high-level ML API       |
| FFT, statistics, random distributions    | Domain-specific           |
| Native GPU (CUDA + WebGPU) + autograd    |                           |

**For SciPy-equivalent functionality** (optimization, ODE, interpolation, signal), see [**solvr**](https://github.com/farhan-syah/solvr).

## Why numr?

### vs NumPy

| Capability                    | NumPy                     | numr                           |
| ----------------------------- | ------------------------- | ------------------------------ |
| N-dimensional tensors         | ✓                         | ✓                              |
| Linear algebra, FFT, stats    | ✓                         | ✓                              |
| **Automatic differentiation** | ✗ Need JAX/PyTorch        | ✓ Built-in `numr::autograd`    |
| **GPU acceleration**          | ✗ Need CuPy/JAX           | ✓ Native CUDA + WebGPU         |
| **Non-NVIDIA GPUs**           | ✗ None                    | ✓ AMD, Intel, Apple via WebGPU |
| **FP8 / BF16 dtypes**         | ✗ / Partial               | ✓ Full support                 |
| **Sparse tensors**            | ✗ SciPy separate, 2D only | ✓ Integrated, N-dimensional    |
| **Same code CPU↔GPU**         | ✗ Different libraries     | ✓ `Tensor<R>` abstraction      |

### vs Rust Ecosystem

Fragmented libraries that don't interoperate and lack GPU support. numr consolidates everything:

| Task                      | Old Ecosystem               | numr                         |
| ------------------------- | --------------------------- | ---------------------------- |
| Tensors                   | ndarray                     | Tensor<R>                    |
| Linear algebra            | nalgebra / faer             | numr::linalg                 |
| FFT                       | rustfft                     | numr::fft                    |
| Sparse                    | sprs / ndsparse             | numr::sparse (feature-gated) |
| Statistics                | statrs                      | numr::statistics             |
| Random numbers            | rand + manual distributions | numr::random + multivariate  |
| GPU support               | None                        | CPU, CUDA, WebGPU            |
| Automatic differentiation | None                        | numr::autograd               |

A Rust developer should never need to look elsewhere for numerical computing.

## Architecture

numr is designed with a simple principle: **same code, any backend**.

```
┌──────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│               (any backend-agnostic code)                    │
└──────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐          ┌────▼────┐          ┌───▼────┐
   │ CPU     │          │ CUDA    │          │ WebGPU │
   │ Runtime │          │ Runtime │          │Runtime │
   └────┬────┘          └────┬────┘          └───┬────┘
        │                    │                   │
   ┌────▼──────────┬─────────┴───────┬───────────▼───┐
   │     Trait     │                 │               │
   │  Implemen-    │  Same Algorithm │  Different    │
   │  tations      │  Different Code │  Hardware     │
   └───────────────┴─────────────────┴───────────────┘
```

## Operations

numr implements a comprehensive set of tensor operations across CPU, CUDA, and WebGPU:

### Core Arithmetic

- **UnaryOps**: neg, abs, sqrt, exp, log, sin, cos, tan, sinh, cosh, tanh, floor, ceil, round, and more
- **BinaryOps**: add, sub, mul, div, pow, maximum, minimum (all with NumPy-style broadcasting)
- **ScalarOps**: tensor-scalar arithmetic
- **TypeConversionOps**: cast (convert between dtypes)
- **UtilityOps**: clamp, fill, arange, linspace, eye

### Shape and Data Movement

- **ShapeOps**: cat, stack, split, chunk, repeat, pad, roll
- **IndexingOps**: gather, scatter, gather_nd, scatter_reduce, index_select, masked_select, masked_fill, embedding_lookup, bincount, argmax, argmin
- **SortingOps**: sort, argsort, topk, unique, nonzero, searchsorted

### Reductions

- **ReduceOps**: sum, mean, max, min, prod (with precision variants)
- **CumulativeOps**: cumsum, cumprod, logsumexp

### Comparisons and Logical

- **CompareOps**: eq, ne, lt, le, gt, ge
- **LogicalOps**: logical_and, logical_or, logical_xor, logical_not
- **ConditionalOps**: where (ternary conditional)

### Activation & Normalization Functions

- **ActivationOps**: relu, sigmoid, silu, gelu, leaky_relu, elu, softmax
- **NormalizationOps**: rms_norm, layer_norm
- **ConvOps**: conv1d, conv2d, depthwise_conv2d (with stride, padding, dilation, groups)

_These are mathematical functions commonly used in ML, but numr itself is not an ML framework._

### Linear Algebra

- **MatmulOps**: matmul, matmul_bias (fused GEMM+bias)
- **LinalgOps**: solve, lstsq, pinverse, inverse, det, trace, matrix_rank, diag, matrix_norm, kron, khatri_rao
- **ComplexOps**: conj, real, imag, angle (for complex tensor support)

### Statistics and Probability

- **StatisticalOps**: var, std, skew, kurtosis, quantile, percentile, median, cov, corrcoef
- **RandomOps**: rand, randn, randint, multinomial, bernoulli, poisson, binomial, beta, gamma, exponential, chi_squared, student_t, f_distribution
- **MultivariateRandomOps**: multivariate_normal, wishart, dirichlet
- **QuasirandomOps**: Sobol, Halton sequences

### Distance Metrics

- **DistanceOps**: euclidean, manhattan, cosine, hamming, jaccard, minkowski, chebyshev, correlation

### Algorithm Modules

**Linear Algebra (`numr::linalg`):**

- **Decompositions**: LU, QR, Cholesky, SVD, Schur, full eigendecomposition, generalized eigenvalues
- **Solvers**: solve, lstsq, pinverse
- **Matrix functions**: exp, log, sqrt, sign
- **Utilities**: det, trace, rank, matrix norms

**Fast Fourier Transform (`numr::fft`):**

- FFT/IFFT (1D, 2D, ND) - Stockham algorithm
- Real FFT (RFFT/IRFFT)

**Matrix Multiplication (`numr::matmul`):**

- Tiled GEMM with register blocking
- Bias fusion support

**Special Functions (`numr::special`):**

- Error functions: erf, erfc, erfinv
- Gamma functions: gamma, lgamma, digamma
- Beta functions: beta, betainc
- Incomplete gamma: gammainc, gammaincc
- Bessel functions: J₀, J₁, Y₀, Y₁, I₀, I₁, K₀, K₁
- Elliptic integrals: ellipk, ellipe
- Hypergeometric functions: hyp2f1, hyp1f1
- Airy functions: airy_ai, airy_bi
- Legendre functions: legendre_p, legendre_p_assoc, sph_harm
- Fresnel integrals: fresnel_s, fresnel_c

**Sparse Tensors (`numr::sparse`, feature-gated):**

- Formats: CSR, CSC, COO
- Operations: SpGEMM (sparse matrix multiplication), SpMV (sparse matrix-vector), DSMM (dense-sparse matrix)

## Dtypes

numr supports a wide range of numeric types:

| Type    | Size | CPU | CUDA | WebGPU | Feature |
| ------- | ---- | --- | ---- | ------ | ------- |
| f64     | 8B   | ✓   | ✓    | ✗      | -       |
| f32     | 4B   | ✓   | ✓    | ✓      | -       |
| f16     | 2B   | ✓   | ✓    | ✓      | `f16`   |
| bf16    | 2B   | ✓   | ✓    | ✗      | `f16`   |
| fp8e4m3 | 1B   | ✓   | ✓    | ✗      | `fp8`   |
| fp8e5m2 | 1B   | ✓   | ✓    | ✗      | `fp8`   |
| i64     | 8B   | ✓   | ✓    | ✗      | -       |
| i32     | 4B   | ✓   | ✓    | ✓      | -       |
| i16     | 2B   | ✓   | ✓    | ✗      | -       |
| i8      | 1B   | ✓   | ✓    | ✗      | -       |
| u64     | 8B   | ✓   | ✓    | ✗      | -       |
| u32     | 4B   | ✓   | ✓    | ✓      | -       |
| u16     | 2B   | ✓   | ✓    | ✗      | -       |
| u8      | 1B   | ✓   | ✓    | ✓      | -       |
| bool    | 1B   | ✓   | ✓    | ✓      | -       |

Every operation supports every compatible dtype. No hardcoded f32-only kernels.

## Backends

All backends implement identical algorithms with native kernels—no cuBLAS, MKL, or vendor library dependencies.

| Hardware     | Backend | Feature       | Status  | Notes              |
| ------------ | ------- | ------------- | ------- | ------------------ |
| CPU (x86-64) | CPU     | cpu (default) | ✓       | AVX-512/AVX2 SIMD  |
| CPU (ARM64)  | CPU     | cpu           | ✓       | NEON SIMD          |
| NVIDIA GPU   | CUDA    | cuda          | ✓       | Native PTX kernels |
| AMD GPU      | WebGPU  | wgpu          | ✓       | WGSL shaders       |
| Intel GPU    | WebGPU  | wgpu          | ✓       | WGSL shaders       |
| Apple GPU    | WebGPU  | wgpu          | ✓       | WGSL shaders       |
| AMD GPU      | ROCm    | -             | Planned | Native HIP kernels |

### SIMD Acceleration

The CPU backend automatically detects and uses the best available SIMD instruction set at runtime:

| Architecture | Instruction Set | Vector Width | Elements per Op |
| ------------ | --------------- | ------------ | --------------- |
| x86-64       | AVX-512F + FMA  | 512 bits     | 16 f32 / 8 f64  |
| x86-64       | AVX2 + FMA      | 256 bits     | 8 f32 / 4 f64   |
| ARM64        | NEON            | 128 bits     | 4 f32 / 2 f64   |

**Vectorized operations include:**

- Element-wise: add, sub, mul, div, neg, abs, sqrt, exp, log, sin, cos, tanh, and more
- Reductions: sum, max, min, prod with horizontal SIMD reductions
- Activations: sigmoid, silu, gelu, leaky_relu, elu
- Normalization: softmax, rms_norm, layer_norm, logsumexp
- Matrix multiplication: tiled GEMM with FMA microkernels
- Special functions: erf, erfc, bessel, gamma (with polynomial approximations)

### Why Native Kernels?

numr uses native kernels (SIMD, PTX, WGSL) by default—not cuBLAS/MKL wrappers.

|                     | Vendor Libraries (cuBLAS/MKL)      | numr Native Kernels                       |
| ------------------- | ---------------------------------- | ----------------------------------------- |
| **Transparency**    | Black box                          | Whitebox—inspect, debug, step through     |
| **Portability**     | NVIDIA-only (cuBLAS)               | CPU, NVIDIA, AMD, Intel, Apple            |
| **Reproducibility** | Heuristics change between versions | Bit-exact results, fixed in crate version |
| **Choice**          | Locked in                          | Swap in vendor kernels if needed          |
| **Dependencies**    | 2GB+ CUDA toolkit                  | Minimal                                   |
| **Deployment**      | Complex linking                    | Simple static binaries                    |

**You're not locked out of vendor libraries**—numr's kernel system is extensible. Use native kernels for portability and transparency, or swap in cuBLAS/MKL for maximum vendor-optimized performance. Other libraries don't give you this choice.

## Quick Start

### CPU Example

```rust
use numr::prelude::*;
use numr::runtime::cpu::CpuRuntime;

fn main() -> Result<()> {
    // Create tensors
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    )?;
    let b = Tensor::<CpuRuntime>::from_slice(
        &[5.0, 6.0, 7.0, 8.0],
        &[2, 2],
    )?;

    // Arithmetic (with broadcasting)
    let c = a.add(&b)?;
    let d = a.mul(&b)?;

    // Matrix multiplication
    let e = a.matmul(&b)?;

    // Reductions
    let sum = c.sum()?;
    let mean = c.mean()?;
    let max = c.max()?;

    // Element-wise functions
    let exp = a.exp()?;
    let sqrt = a.sqrt()?;

    // Reshaping (zero-copy)
    let flat = c.reshape(&[4])?;
    let transposed = c.transpose()?;

    Ok(())
}
```

### GPU Example (CUDA)

```rust
use numr::prelude::*;
use numr::runtime::cuda::CudaRuntime;

fn main() -> Result<()> {
    // Create on GPU
    let device = CudaRuntime::default_device()?;
    let a = Tensor::<CudaRuntime>::randn(&[1024, 1024], &device)?;
    let b = Tensor::<CudaRuntime>::randn(&[1024, 1024], &device)?;

    // Operations run on GPU (native CUDA kernels)
    let c = a.matmul(&b)?;

    // Transfer result to CPU when needed
    let cpu_result = c.to_cpu()?;
    let data = cpu_result.to_vec::<f32>()?;

    Ok(())
}
```

### Backend-Generic Code

```rust
use numr::prelude::*;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// Works on CPU, CUDA, or WebGPU
fn matrix_operations<R: Runtime>(
    a: &Tensor<R>,
    b: &Tensor<R>,
    client: &R::Client,
) -> Result<Tensor<R>> {
    // Same code, any backend
    let c = client.add(a, b)?;
    let d = client.matmul(&c, a)?;
    client.sum(&d)
}

// Use the same function on different hardware
fn main() -> Result<()> {
    let a_cpu = Tensor::<CpuRuntime>::randn(&[128, 128], &device_cpu)?;
    let b_cpu = Tensor::<CpuRuntime>::randn(&[128, 128], &device_cpu)?;
    let result_cpu = matrix_operations(&a_cpu, &b_cpu, &client_cpu)?;

    #[cfg(feature = "cuda")]
    {
        let device_cuda = CudaRuntime::default_device()?;
        let a_cuda = Tensor::<CudaRuntime>::randn(&[128, 128], &device_cuda)?;
        let b_cuda = Tensor::<CudaRuntime>::randn(&[128, 128], &device_cuda)?;
        let result_cuda = matrix_operations(&a_cuda, &b_cuda, &client_cuda)?;
    }

    Ok(())
}
```

### Linear Algebra

```rust
use numr::prelude::*;
use numr::algorithm::linalg::{LinalgOps, Decomposition};

fn main() -> Result<()> {
    let a = Tensor::<CpuRuntime>::randn(&[64, 64], &device)?;

    // LU decomposition
    let (p, l, u) = client.lu(&a)?;

    // QR decomposition
    let (q, r) = client.qr(&a)?;

    // SVD
    let (u, s, vt) = client.svd(&a)?;

    // Eigendecomposition
    let (eigenvalues, eigenvectors) = client.eig(&a)?;

    // Solve linear system: Ax = b
    let b = Tensor::<CpuRuntime>::randn(&[64, 32], &device)?;
    let x = client.solve(&a, &b)?;

    // Determinant, trace, rank
    let det = client.det(&a)?;
    let tr = client.trace(&a)?;
    let rank = client.matrix_rank(&a)?;

    Ok(())
}
```

### FFT

```rust
use numr::prelude::*;
use numr::algorithm::fft::FftOps;

fn main() -> Result<()> {
    let x = Tensor::<CpuRuntime>::randn(&[1024], &device)?;

    // Complex FFT
    let fft_result = client.fft(&x)?;
    let inverse = client.ifft(&fft_result)?;

    // Real FFT (more efficient for real-valued inputs)
    let rfft_result = client.rfft(&x)?;
    let irfft_result = client.irfft(&rfft_result, 1024)?;

    // 2D FFT
    let image = Tensor::<CpuRuntime>::randn(&[256, 256], &device)?;
    let fft_2d = client.fft_2d(&image)?;

    Ok(())
}
```

### Statistics and Distributions

```rust
use numr::prelude::*;

fn main() -> Result<()> {
    let data = Tensor::<CpuRuntime>::randn(&[1000], &device)?;

    // Descriptive statistics
    let mean = client.mean(&data)?;
    let std = client.std(&data)?;
    let var = client.var(&data)?;
    let median = client.median(&data)?;
    let q25 = client.quantile(&data, 0.25)?;

    // Statistical measures
    let skewness = client.skew(&data)?;
    let kurtosis = client.kurtosis(&data)?;

    // Covariance and correlation
    let x = Tensor::<CpuRuntime>::randn(&[100, 5], &device)?;
    let y = Tensor::<CpuRuntime>::randn(&[100, 5], &device)?;
    let cov = client.cov(&x)?;
    let corr = client.corrcoef(&x)?;

    // Random distributions
    let normal = Tensor::<CpuRuntime>::randn(&[1000], &device)?; // mean=0, std=1
    let uniform = Tensor::<CpuRuntime>::rand(&[1000], &device)?; // [0, 1)
    let gamma = client.gamma(&[1000], shape, scale, &device)?;
    let poisson = client.poisson(&[1000], lambda, &device)?;

    // Multivariate distributions
    let mvn = client.multivariate_normal(&[100], &mean, &cov)?;
    let wishart = client.wishart(&[10], df, &scale_matrix)?;

    Ok(())
}
```

## Installation

### CPU-only (default)

```toml
[dependencies]
numr = "*"
```

### With GPU Support

```toml
[dependencies]
# NVIDIA CUDA (requires CUDA 12.0+)
numr = { version = "*", features = ["cuda"] }

# Cross-platform GPU (NVIDIA, AMD, Intel, Apple)
numr = { version = "*", features = ["wgpu"] }
```

### With Optional Features

```toml
[dependencies]
numr = { version = "*", features = [
    "cuda",      # NVIDIA GPU support
    "wgpu",      # Cross-platform GPU (WebGPU)
    "f16",       # Half-precision (F16, BF16)
    "fp8",       # 8-bit floating point
    "sparse",    # Sparse tensors
] }
```

## Feature Flags

| Feature  | Description                                         | Default |
| -------- | --------------------------------------------------- | ------- |
| `cpu`    | CPU backend (AVX-512/AVX2 on x86-64, NEON on ARM64) | ✓       |
| `cuda`   | NVIDIA CUDA backend                                 | ✗       |
| `wgpu`   | Cross-platform GPU (WebGPU)                         | ✗       |
| `rayon`  | Multi-threaded CPU via Rayon                        | ✓       |
| `f16`    | Half-precision floats (F16, BF16)                   | ✗       |
| `fp8`    | 8-bit floats (FP8E4M3, FP8E5M2)                     | ✗       |
| `sparse` | Sparse tensor support (CSR, CSC, COO)               | ✗       |

## Building from Source

```bash
# CPU only
cargo build --release

# With CUDA
cargo build --release --features cuda

# With WebGPU
cargo build --release --features wgpu

# With all features
cargo build --release --features cuda,wgpu,f16,fp8,sparse

# Run tests
cargo test --release
cargo test --release --features cuda
cargo test --release --features wgpu

# Run benchmarks
cargo bench
```

## How numr Fits in the Stack

numr is the **foundation** that everything else builds on:

```
┌──────────────────────────────────────────────────┐
│  Your Application                                 │
│  (data science, simulation, finance, ML, etc.)   │
└─────────────────────────┬────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────┐
│  solvr - Scientific Computing (like SciPy)       │
│  Optimization, ODE/PDE, interpolation, signal    │
│  https://github.com/farhan-syah/solvr            │
└─────────────────────────┬────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────┐
│  numr - Foundations (like NumPy)  ◄── YOU ARE HERE│
│  Tensors, linalg, FFT, statistics, random        │
│  Native CPU, CUDA, WebGPU kernels + autograd     │
└──────────────────────────────────────────────────┘
```

**numr : solvr :: NumPy : SciPy**

When numr's kernels improve, everything above improves automatically.

## Kernels and Extensibility

numr provides default kernels for all operations. You can also:

- **Use default kernels**: All operations work out of the box with optimized kernels:
  - **CPU**: SIMD-vectorized kernels (AVX-512/AVX2 on x86-64, NEON on ARM64)
  - **CUDA**: Native PTX kernels (compiled at build time, loaded on first use)
  - **WebGPU**: WGSL compute shaders for cross-platform GPU
- **Replace specific kernels**: Swap in your own optimized kernels for performance-critical paths
- **Add new operations**: Define new traits and implement kernels for all backends

For detailed guidance on writing custom kernels, adding new operations, and backend-specific optimization techniques, see **[docs/extending-numr.md](docs/extending-numr.md)**.

## License

Apache-2.0
