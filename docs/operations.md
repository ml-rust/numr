# Operations

This page lists the tensor operations numr provides. Each operation is a method on a trait
in `numr::ops`. Every backend (CPU, CUDA, WebGPU) implements the same traits.

The traits take the client as `self`. Call them on a `RuntimeClient`, such as `CpuClient`.
Inherent `Tensor` methods cover a subset of these, including `add`, `mul`, `matmul`, `exp`,
`sqrt`, `sum`, `mean`, `max`, `min`, `reshape`, and `transpose`.

## Core Arithmetic

| Trait | Operations |
| ----- | ---------- |
| `UnaryOps` | `neg`, `abs`, `sqrt`, `rsqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round` (ties away from zero), `round_ties_even` (IEEE roundTiesToEven, matching `np.round` and `torch.round`), and more |
| `BinaryOps` | `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, plus fused forms such as `fused_mul_add` and `fused_add_mul`. All use NumPy-style broadcasting. |
| `ScalarOps` | Tensor-scalar arithmetic, including `fused_mul_add_scalar` |
| `TypeConversionOps` | `cast`, which converts between dtypes |
| `UtilityOps` | `clamp`, `fill`, `arange`, `linspace`, `eye` |

## Shape and Data Movement

| Trait | Operations |
| ----- | ---------- |
| `ShapeOps` | `cat`, `stack`, `split`, `chunk`, `repeat`, `repeat_interleave`, `pad`, `roll` |
| `IndexingOps` | `gather`, `scatter`, `gather_nd`, `scatter_reduce`, `index_select`, `masked_select`, `masked_fill`, `embedding_lookup`, `bincount`, `bincount_with_len`, `argmax`, `argmin`, `slice_assign` |
| `SortingOps` | `sort`, `argsort`, `topk`, `unique`, `nonzero`, `searchsorted` |

## Reductions

| Trait | Operations |
| ----- | ---------- |
| `ReduceOps` | `sum`, `mean`, `max`, `min`, `prod`, with precision variants |
| `CumulativeOps` | `cumsum`, `cumprod`, `logsumexp` |

## Comparisons and Logic

| Trait | Operations |
| ----- | ---------- |
| `CompareOps` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` |
| `LogicalOps` | `logical_and`, `logical_or`, `logical_xor`, `logical_not` |
| `ConditionalOps` | `where_cond` (ternary conditional) |

## Activation and Normalization

| Trait | Operations |
| ----- | ---------- |
| `ActivationOps` | `relu`, `sigmoid`, `silu`, `gelu`, `gelu_erf`, `leaky_relu`, `elu`, `softplus`, `softmax`, `dropout`, and fused activation-mul forms: `silu_mul`, `gelu_mul`, `gelu_erf_mul`, `relu_mul` (for gated architectures) |
| `NormalizationOps` | `rms_norm`, `layer_norm`, `group_norm`, fused add-norm (`fused_add_rms_norm`, `fused_add_layer_norm`) |
| `GemmEpilogueOps` | Fused matmul with bias and activation, in one kernel (forward and backward). Includes `matmul_bias_activation` and `matmul_bias_residual`. |
| `ConvOps` | `conv1d`, `conv2d`, `depthwise_conv2d`, with stride, padding, dilation, and groups |
| `EinsumOps` | Einstein summation notation |

numr provides these functions because ML code uses them. numr itself is not an ML framework.

## Linear Algebra

| Trait | Operations |
| ----- | ---------- |
| `MatmulOps` | `matmul`, `matmul_bias` (fused GEMM with bias), `matmul_wide` (widens `i8` and half-precision inputs to a wider output) |
| `LinalgOps` | `solve`, `lstsq`, `pinverse`, `inverse`, `det`, `trace`, `matrix_rank`, `diag`, `matrix_norm`, `kron`, `khatri_rao` |
| `ComplexOps` | `conj`, `real`, `imag`, `angle` |

The decompositions (`lu_decompose`, `qr_decompose`, `cholesky_decompose`, `svd_decompose`,
`schur_decompose`, `eig_decompose`) are on `LinearAlgebraAlgorithms`. See [Algorithms](algorithms.md).

## Statistics and Probability

| Trait | Operations |
| ----- | ---------- |
| `StatisticalOps` | `var`, `std`, `skew`, `kurtosis`, `quantile`, `percentile`, `median`, `cov`, `corrcoef` |
| `RandomOps` | `rand`, `randn`, `randint`, `multinomial`, `bernoulli`, `poisson`, `binomial`, `beta`, `gamma`, `exponential`, `chi_squared`, `student_t`, `f_distribution`. Seeded generation is deterministic. |
| `MultivariateRandomOps` | `multivariate_normal`, `wishart`, `dirichlet` |
| `QuasiRandomOps` | `sobol` and `halton` sequences |

## Distance Metrics

`DistanceOps` provides `cdist`, `pdist`, `squareform`, and `squareform_inverse`. `cdist` and
`pdist` take a `DistanceMetric`:

| `DistanceMetric` variant | Notes |
| ------------------------ | ----- |
| `Euclidean` | |
| `SquaredEuclidean` | |
| `Manhattan` | |
| `Chebyshev` | |
| `Minkowski(f64)` | Takes the order `p` |
| `Cosine` | |
| `Correlation` | |
| `Hamming` | |
| `Jaccard` | |

### Distance Kernels on Slices

The `numr::distance` module computes distances on plain slices. It needs no `Tensor`, runtime, or client.

- Per pair: `dot_f32`, `l2_squared_f32`, `manhattan_f32`, `cosine_distance_f32`, and the `_f64` variants.
- One query against many rows: `dot_many_f32`, `l2_squared_many_f32`, `manhattan_many_f32`,
  `cosine_distance_many_f32`, and the `_f64` variants.
- Signed 8-bit: `dot_i8`, `dot_i8_scaled`, `dot_i8_many`.
- `Kernels::detect()` fixes the SIMD level once. `Kernels::with_level` pins a level the CPU supports.

```rust
use numr::distance;

let a = [1.0_f32, 2.0, 3.0];
let b = [4.0_f32, 5.0, 6.0];
assert_eq!(distance::dot_f32(&a, &b), 32.0);
```

## Related Pages

- [Algorithms](algorithms.md): decompositions, FFT, special functions, iterative and sparse solvers.
- [Autograd](autograd.md): gradients and the differentiable ops.
- [Backends](backends.md): dtypes, SIMD, and GPU backends.
