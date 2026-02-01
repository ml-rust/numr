//! Tensor operations
//!
//! This module defines operation traits and implementations for
//! arithmetic, matrix operations, reductions, and activations.
//!
//! # Design
//!
//! Operations are defined as traits that are implemented by `RuntimeClient`.
//! This gives operations access to device and allocator for creating output tensors.
//!
//! ```text
//! RuntimeClient<R>
//!   └── implements TensorOps<R>
//!         ├── add, sub, mul, div (binary arithmetic)
//!         ├── neg, sqrt, exp, ... (unary operations)
//!         ├── matmul             (matrix multiplication)
//!         ├── sum, mean, max, min (reductions)
//!         └── relu, sigmoid, softmax (activations)
//! ```
//!
//! # Implementing Operations for a New Backend
//!
//! To add operations for a new backend (e.g., CUDA, WebGPU):
//!
//! 1. **Implement `TensorOps<YourRuntime>` for your `Client` type:**
//!    ```ignore
//!    impl TensorOps<CudaRuntime> for CudaClient {
//!        fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
//!            // 1. Validate shapes are broadcastable
//!            let out_shape = broadcast_shape(a.shape(), b.shape())
//!                .ok_or(Error::BroadcastError { ... })?;
//!
//!            // 2. Allocate output tensor
//!            let out = Tensor::empty(&out_shape, a.dtype(), self.device());
//!
//!            // 3. Dispatch kernel
//!            cuda_add_kernel(a.storage().ptr(), b.storage().ptr(), out.storage().ptr(), ...);
//!
//!            Ok(out)
//!        }
//!        // ... other operations
//!    }
//!    ```
//!
//! 2. **Use helper types for operation parameters:**
//!    - [`BinaryOp`], [`UnaryOp`] - Operation kind enums for dispatch
//!    - [`MatmulParams`] - Matrix multiplication configuration
//!    - [`ReduceOp`] - Reduction operation kinds
//!    - [`ActivationKind`] - Activation function kinds
//!
//! 3. **Use validation helpers:**
//!    - [`broadcast_shape`] - Compute broadcast shape for binary ops
//!    - [`validate_matmul_shapes`] - Validate matmul dimensions
//!    - [`reduce_output_shape`] - Compute reduction output shape
//!
//! # Operation Categories
//!
//! ## Element-wise Operations
//! Binary (add, sub, mul, div) and unary (neg, abs, sqrt, exp, log, sin, cos, tanh).
//!
//! **Note:** Broadcasting is implemented for binary arithmetic and comparison ops
//! via strided kernels on CPU.
//!
//! ## Matrix Operations
//! Matrix multiplication with batching support. Inner dimensions must match.
//!
//! ## Reductions
//! Sum, mean, max, min over specified dimensions with optional keepdim.
//!
//! ## Activations
//! ReLU, sigmoid, softmax for neural network layers.

mod activation;
mod arithmetic;
mod dispatch;
mod matmul;
mod reduce;
mod special;
mod traits;

pub use activation::*;
pub use arithmetic::*;
pub use dispatch::*;
pub use matmul::*;
pub use reduce::*;
pub use special::SpecialFunctions;
pub use traits::{
    ActivationOps, CompareOps, ComplexOps, ConditionalOps, CumulativeOps, IndexingOps, Kernel,
    LogicalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, SortingOps, StatisticalOps,
    TypeConversionOps, UtilityOps,
};

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// Kernel trait moved to traits/kernel.rs

// ============================================================================
// High-Level TensorOps Trait (Layer 1)
// ============================================================================

/// Core tensor operations trait
///
/// This trait defines low-level operations that each backend must implement.
/// It is implemented by `RuntimeClient` types, giving operations access to
/// the device and allocator for creating output tensors.
///
/// # Example
///
/// ```ignore
/// use numr::prelude::*;
///
/// let device = CpuDevice::new();
/// let client = CpuRuntime::default_client(&device);
///
/// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
/// let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);
///
/// let c = client.add(&a, &b)?;
/// ```
pub trait TensorOps<R: Runtime>:
    TypeConversionOps<R>
    + ConditionalOps<R>
    + ComplexOps<R>
    + NormalizationOps<R>
    + MatmulOps<R>
    + CumulativeOps<R>
    + ActivationOps<R>
    + UtilityOps<R>
    + ReduceOps<R>
    + IndexingOps<R>
    + SortingOps<R>
    + StatisticalOps<R>
{
    // ===== Element-wise Binary Operations =====

    /// Element-wise addition: a + b
    fn add(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise subtraction: a - b
    fn sub(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise multiplication: a * b
    fn mul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise division: a / b
    fn div(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Element-wise Unary Operations =====

    // --- Sign and Absolute ---

    /// Negation: -a
    fn neg(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Absolute value: |a|
    fn abs(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Sign: returns -1 for negative, 0 for zero, 1 for positive
    fn sign(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // --- Power and Root ---

    /// Square root: sqrt(a)
    fn sqrt(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Reciprocal square root: 1/sqrt(a) - critical for normalization layers
    fn rsqrt(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Square: a²
    fn square(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Cube root: cbrt(a)
    fn cbrt(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Reciprocal: 1/a
    fn recip(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // --- Exponential and Logarithmic ---

    /// Exponential: e^a
    fn exp(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Base-2 exponential: 2^a
    fn exp2(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Exponential minus 1: e^a - 1 (numerically stable for small a)
    fn expm1(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Natural logarithm: ln(a)
    fn log(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Base-2 logarithm: log2(a)
    fn log2(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Base-10 logarithm: log10(a)
    fn log10(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Natural log of 1+a: ln(1+a) (numerically stable for small a)
    fn log1p(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // --- Trigonometric ---

    /// Sine: sin(a)
    fn sin(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Cosine: cos(a)
    fn cos(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Tangent: tan(a)
    fn tan(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Arc sine (inverse sine): asin(a), domain [-1,1], range [-π/2, π/2]
    fn asin(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Arc cosine (inverse cosine): acos(a), domain [-1,1], range [0, π]
    fn acos(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Arc tangent (inverse tangent): atan(a)
    fn atan(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // --- Hyperbolic ---

    /// Hyperbolic sine: sinh(a)
    fn sinh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Hyperbolic cosine: cosh(a)
    fn cosh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Hyperbolic tangent: tanh(a)
    fn tanh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Inverse hyperbolic sine: asinh(a)
    fn asinh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Inverse hyperbolic cosine: acosh(a), domain [1, ∞)
    fn acosh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Inverse hyperbolic tangent: atanh(a), domain (-1, 1)
    fn atanh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // --- Rounding ---

    /// Floor: floor(a)
    fn floor(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Ceiling: ceil(a)
    fn ceil(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Round: round(a) to nearest integer
    fn round(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Truncate toward zero: trunc(a)
    fn trunc(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // --- Special Checks ---

    /// Check for NaN values: returns U8 tensor (1 if NaN, 0 otherwise)
    fn isnan(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Check for Inf values: returns U8 tensor (1 if Inf, 0 otherwise)
    fn isinf(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Element-wise Binary Operations (extended) =====

    /// Element-wise power: a^b
    fn pow(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise maximum: max(a, b)
    fn maximum(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise minimum: min(a, b)
    fn minimum(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Two-argument arctangent: atan2(y, x)
    ///
    /// Computes the angle in radians between the positive x-axis and the point (x, y).
    /// Result is in the range [-π, π]. Essential for polar coordinates and spatial algorithms.
    fn atan2(&self, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Reductions =====
    // Moved to ReduceOps trait in traits/reduce.rs

    // ===== Cumulative Operations =====
    // Moved to CumulativeOps trait in traits/cumulative.rs

    // ===== Activations =====
    // Moved to ActivationOps trait in traits/activation.rs

    // ===== Index Operations =====
    // Moved to IndexingOps trait in traits/indexing.rs

    // ===== Sorting and Search Operations =====
    // Moved to SortingOps trait in traits/sorting.rs

    // ===== Type Conversion =====
    // Moved to TypeConversionOps trait in traits/type_conversion.rs

    // ===== Complex Number Operations =====
    // Moved to ComplexOps trait in traits/complex.rs

    // ===== Conditional Operations =====
    // Moved to ConditionalOps trait in traits/conditional.rs

    // ===== Utility Operations =====
    // Moved to UtilityOps trait in traits/utility.rs

    // ===== Statistical Operations =====
    // Moved to StatisticalOps trait in traits/statistics.rs

    // ===== Random Operations =====

    /// Generate uniform random values in [0, 1)
    ///
    /// Creates a tensor filled with random values uniformly distributed in [0, 1).
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with uniform random values
    fn rand(&self, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>>;

    /// Generate standard normal random values (mean=0, std=1)
    ///
    /// Creates a tensor filled with random values from standard normal distribution N(0, 1).
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with normally distributed random values
    fn randn(&self, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>>;

    /// Generate random integers in the range [low, high)
    ///
    /// Creates a tensor filled with random integers uniformly distributed in [low, high).
    /// The `high` value is exclusive (never included in the output).
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (exclusive), must be > low
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be integer type: I8, I16, I32, I64, U8, U16, U32, U64)
    ///
    /// # Returns
    ///
    /// Tensor filled with random integers
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `high <= low`
    /// - The range `[low, high)` cannot be represented in the specified dtype
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not an integer type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Random integers in [0, 10)
    /// let a = client.randint(0, 10, &[3, 4], DType::I32)?;
    ///
    /// // Random bytes in [0, 256)
    /// let b = client.randint(0, 256, &[1024], DType::U8)?;
    ///
    /// // Random signed integers in [-100, 100)
    /// let c = client.randint(-100, 100, &[10], DType::I32)?;
    /// ```
    ///
    /// # Notes
    ///
    /// - For unsigned types, `low` must be >= 0
    /// - The distribution is uniform over the discrete values in [low, high)
    /// - Each call produces independent random values (not reproducible without seeding)
    fn randint(
        &self,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from a multinomial (categorical) distribution
    ///
    /// Given a tensor of probabilities for each category, samples indices according
    /// to those probabilities. This is the fundamental operation for categorical
    /// sampling in machine learning, including LLM next-token selection.
    ///
    /// # Algorithm
    ///
    /// Uses inverse transform sampling (CDF method):
    /// 1. Compute cumulative sum of probabilities (CDF)
    /// 2. For each sample, draw uniform random u ∈ [0, 1)
    /// 3. Find smallest index i where CDF[i] ≥ u (binary search)
    ///
    /// ```text
    /// probs:  [0.1, 0.2, 0.3, 0.4]
    /// CDF:    [0.1, 0.3, 0.6, 1.0]
    ///          ↑    ↑    ↑    ↑
    /// u=0.05 → 0    │    │    │   (u < 0.1)
    /// u=0.25 ──────→ 1   │    │   (0.1 ≤ u < 0.3)
    /// u=0.55 ─────────→ 2│    │   (0.3 ≤ u < 0.6)
    /// u=0.80 ──────────────────→ 3 (0.6 ≤ u < 1.0)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `probs` - Probability tensor with shape `[..., num_categories]`
    ///   - Probabilities must be non-negative
    ///   - Probabilities are normalized automatically (do not need to sum to 1)
    ///   - Must be floating point dtype (F32, F64, F16, BF16)
    /// * `num_samples` - Number of samples to draw per distribution
    /// * `replacement` - Whether to sample with replacement
    ///   - `true`: Same category can be sampled multiple times
    ///   - `false`: Each category sampled at most once (requires num_samples ≤ num_categories)
    ///
    /// # Returns
    ///
    /// Tensor of sampled indices with shape `[..., num_samples]` and dtype I64 (I32 on WGPU).
    /// Each index is in the range `[0, num_categories)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `probs` is empty or has zero categories
    /// - `num_samples` is 0
    /// - `replacement` is false and `num_samples > num_categories`
    /// - `probs` contains negative values
    /// - All probabilities in a row are zero (no valid category to sample)
    ///
    /// Returns `Error::UnsupportedDType` if `probs` is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Sample one index from a 4-category distribution
    /// let probs = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], &[4], &device);
    /// let sample = client.multinomial(&probs, 1, true)?;  // Shape: [1]
    ///
    /// // Sample 3 indices without replacement
    /// let samples = client.multinomial(&probs, 3, false)?;  // Shape: [3]
    ///
    /// // Batch sampling: 2 distributions, 5 samples each
    /// let batch_probs = Tensor::from_slice(
    ///     &[0.1, 0.9, 0.5, 0.5],  // 2 rows of 2 categories
    ///     &[2, 2], &device
    /// );
    /// let batch_samples = client.multinomial(&batch_probs, 5, true)?;  // Shape: [2, 5]
    /// ```
    ///
    /// # Notes
    ///
    /// - Input probabilities are normalized per distribution (per row)
    /// - Zero-probability categories are never sampled
    /// - When `replacement=false`, samples within each distribution are unique
    /// - Each call produces independent random samples (not reproducible without seeding)
    /// - This is the PyTorch `torch.multinomial` equivalent
    fn multinomial(
        &self,
        probs: &Tensor<R>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<R>>;

    /// Sample from a Bernoulli distribution
    ///
    /// Creates a tensor where each element is 1 with probability p and 0 otherwise.
    /// This is the fundamental binary random variable.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// P(X = 1) = p
    /// P(X = 0) = 1 - p
    /// Mean = p
    /// Variance = p(1 - p)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of success (1), must be in [0, 1]
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with 0s and 1s sampled from Bernoulli(p)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if p is not in [0, 1].
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Fair coin flips
    /// let flips = client.bernoulli(0.5, &[100], DType::F32)?;
    ///
    /// // Biased coin (70% heads)
    /// let biased = client.bernoulli(0.7, &[1000], DType::F32)?;
    /// ```
    fn bernoulli(&self, p: f64, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>>;

    /// Sample from a Beta distribution
    ///
    /// Creates a tensor filled with random values from the Beta distribution,
    /// which is commonly used as a prior for probabilities in Bayesian statistics.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; α, β) = x^(α-1) * (1-x)^(β-1) / B(α, β)
    /// where B(α, β) = Γ(α)Γ(β) / Γ(α+β)
    ///
    /// Support: x ∈ (0, 1)
    /// Mean = α / (α + β)
    /// Variance = αβ / ((α+β)²(α+β+1))
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the relationship: if X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
    /// then X / (X + Y) ~ Beta(α, β).
    ///
    /// # Arguments
    ///
    /// * `alpha` - First shape parameter (α > 0)
    /// * `beta` - Second shape parameter (β > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with values in (0, 1) from Beta(α, β)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if alpha ≤ 0 or beta ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Symmetric beta (same as uniform for α=β=1)
    /// let uniform_like = client.beta(1.0, 1.0, &[1000], DType::F32)?;
    ///
    /// // Skewed towards 0
    /// let left_skewed = client.beta(0.5, 5.0, &[1000], DType::F32)?;
    ///
    /// // Skewed towards 1
    /// let right_skewed = client.beta(5.0, 0.5, &[1000], DType::F32)?;
    ///
    /// // Bell-shaped in middle
    /// let bell = client.beta(5.0, 5.0, &[1000], DType::F32)?;
    /// ```
    fn beta(
        &self,
        alpha: f64,
        beta: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from a Gamma distribution
    ///
    /// Creates a tensor filled with random values from the Gamma distribution,
    /// which is used for modeling waiting times and is the basis for many other distributions.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; k, θ) = x^(k-1) * e^(-x/θ) / (θ^k * Γ(k))
    ///
    /// Support: x > 0
    /// Mean = k * θ
    /// Variance = k * θ²
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses Marsaglia and Tsang's method for shape ≥ 1,
    /// with Ahrens-Dieter acceptance-rejection for shape < 1.
    ///
    /// # Arguments
    ///
    /// * `shape_param` - Shape parameter k (k > 0), also called α in some notations
    /// * `scale` - Scale parameter θ (θ > 0), also called 1/rate or 1/β
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with positive values from Gamma(k, θ)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if shape_param ≤ 0 or scale ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Exponential distribution is Gamma(1, θ)
    /// let exponential_like = client.gamma(1.0, 2.0, &[1000], DType::F32)?;
    ///
    /// // Chi-squared with df=5 is Gamma(2.5, 2)
    /// let chi2_like = client.gamma(2.5, 2.0, &[1000], DType::F32)?;
    /// ```
    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from an Exponential distribution
    ///
    /// Creates a tensor filled with random values from the Exponential distribution,
    /// which models the time between events in a Poisson process.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; λ) = λ * e^(-λx)
    ///
    /// Support: x ≥ 0
    /// Mean = 1/λ
    /// Variance = 1/λ²
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses inverse transform sampling: X = -ln(U) / λ where U ~ Uniform(0, 1).
    ///
    /// # Arguments
    ///
    /// * `rate` - Rate parameter λ (λ > 0), also called 1/scale
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with non-negative values from Exponential(λ)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if rate ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Average wait time of 2 seconds (rate = 0.5)
    /// let wait_times = client.exponential(0.5, &[1000], DType::F32)?;
    ///
    /// // High rate = short wait times
    /// let fast_events = client.exponential(10.0, &[1000], DType::F32)?;
    /// ```
    fn exponential(
        &self,
        rate: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from a Poisson distribution
    ///
    /// Creates a tensor filled with random integer values from the Poisson distribution,
    /// which models the number of events in a fixed interval.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// P(X = k) = λ^k * e^(-λ) / k!
    ///
    /// Support: k ∈ {0, 1, 2, ...}
    /// Mean = λ
    /// Variance = λ
    /// ```
    ///
    /// # Algorithm
    ///
    /// - For small λ (< 30): Direct inversion method
    /// - For large λ (≥ 30): Normal approximation with continuity correction
    ///
    /// # Arguments
    ///
    /// * `lambda` - Rate parameter λ (λ > 0), the expected number of events
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point, stores integer counts)
    ///
    /// # Returns
    ///
    /// Tensor filled with non-negative integer values (stored as floats) from Poisson(λ)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if lambda ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Average of 5 events per interval
    /// let counts = client.poisson(5.0, &[1000], DType::F32)?;
    ///
    /// // Rare events (average 0.1 per interval)
    /// let rare = client.poisson(0.1, &[1000], DType::F32)?;
    /// ```
    ///
    /// # Notes
    ///
    /// Output values are non-negative integers but stored in floating point dtype
    /// for compatibility with GPU operations. Cast to integer type if needed.
    fn poisson(
        &self,
        lambda: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from a Binomial distribution
    ///
    /// Creates a tensor filled with random values from the Binomial distribution,
    /// which models the number of successes in n independent Bernoulli trials.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
    ///
    /// Support: k ∈ {0, 1, 2, ..., n}
    /// Mean = n * p
    /// Variance = n * p * (1 - p)
    /// ```
    ///
    /// # Algorithm
    ///
    /// - For small n (< 25): Direct simulation (sum of Bernoulli trials)
    /// - For large n: BTRD algorithm (Binomial, Triangle, Rectangle, Decomposition)
    ///
    /// # Arguments
    ///
    /// * `n` - Number of trials (n > 0)
    /// * `p` - Probability of success per trial, must be in [0, 1]
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with integer values in [0, n] from Binomial(n, p)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if n ≤ 0 or p is not in [0, 1].
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 10 coin flips with fair coin
    /// let flips = client.binomial(10, 0.5, &[1000], DType::F32)?;
    ///
    /// // 100 trials with 20% success rate
    /// let trials = client.binomial(100, 0.2, &[1000], DType::F32)?;
    /// ```
    fn binomial(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from a Laplace (double exponential) distribution
    ///
    /// Creates a tensor filled with random values from the Laplace distribution,
    /// which has heavier tails than the normal distribution.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; μ, b) = (1/2b) * e^(-|x - μ| / b)
    ///
    /// Support: x ∈ (-∞, +∞)
    /// Mean = μ
    /// Variance = 2b²
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses inverse transform: X = μ - b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
    /// where U ~ Uniform(0, 1).
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter μ (mean)
    /// * `scale` - Scale parameter b (b > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with values from Laplace(μ, b)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if scale ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Standard Laplace (loc=0, scale=1)
    /// let standard = client.laplace(0.0, 1.0, &[1000], DType::F32)?;
    ///
    /// // Shifted and scaled
    /// let shifted = client.laplace(5.0, 2.0, &[1000], DType::F32)?;
    /// ```
    fn laplace(
        &self,
        loc: f64,
        scale: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from a Chi-squared distribution
    ///
    /// Creates a tensor filled with random values from the Chi-squared distribution,
    /// which is the distribution of a sum of squared standard normal variables.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; k) = x^(k/2-1) * e^(-x/2) / (2^(k/2) * Γ(k/2))
    ///
    /// Support: x > 0
    /// Mean = k
    /// Variance = 2k
    /// ```
    ///
    /// # Algorithm
    ///
    /// Implemented as Gamma(k/2, 2), since χ²(k) = Gamma(k/2, 2).
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom k (k > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with positive values from χ²(k)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if df ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Chi-squared with 5 degrees of freedom
    /// let chi2 = client.chi_squared(5.0, &[1000], DType::F32)?;
    ///
    /// // Chi-squared test statistic distribution
    /// let test_stats = client.chi_squared(10.0, &[10000], DType::F32)?;
    /// ```
    fn chi_squared(
        &self,
        df: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Sample from a Student's t distribution
    ///
    /// Creates a tensor filled with random values from Student's t distribution,
    /// which arises in estimating the mean of a normally distributed population
    /// when the sample size is small.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; ν) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
    ///
    /// Support: x ∈ (-∞, +∞)
    /// Mean = 0 (for ν > 1), undefined for ν ≤ 1
    /// Variance = ν/(ν-2) (for ν > 2), infinite for 1 < ν ≤ 2
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the relationship: T = Z / √(V/ν) where Z ~ N(0,1) and V ~ χ²(ν).
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom ν (ν > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with values from t(ν)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if df ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // t distribution with 10 degrees of freedom
    /// let t10 = client.student_t(10.0, &[1000], DType::F32)?;
    ///
    /// // Heavy tails with low df
    /// let heavy = client.student_t(2.0, &[1000], DType::F32)?;
    ///
    /// // Approaches normal as df → ∞
    /// let approx_normal = client.student_t(100.0, &[1000], DType::F32)?;
    /// ```
    fn student_t(&self, df: f64, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>>;

    /// Sample from an F distribution
    ///
    /// Creates a tensor filled with random values from the F distribution,
    /// which arises in the analysis of variance (ANOVA) and regression.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; d₁, d₂) = √[(d₁x)^d₁ * d₂^d₂ / (d₁x + d₂)^(d₁+d₂)] / (x * B(d₁/2, d₂/2))
    ///
    /// Support: x > 0
    /// Mean = d₂/(d₂-2) (for d₂ > 2)
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the relationship: F = (X₁/d₁) / (X₂/d₂) where X₁ ~ χ²(d₁), X₂ ~ χ²(d₂).
    ///
    /// # Arguments
    ///
    /// * `df1` - Numerator degrees of freedom d₁ (d₁ > 0)
    /// * `df2` - Denominator degrees of freedom d₂ (d₂ > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with positive values from F(d₁, d₂)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if df1 ≤ 0 or df2 ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // F distribution for ANOVA with 5 and 20 df
    /// let f_stat = client.f_distribution(5.0, 20.0, &[1000], DType::F32)?;
    ///
    /// // Equal degrees of freedom
    /// let f_equal = client.f_distribution(10.0, 10.0, &[1000], DType::F32)?;
    /// ```
    fn f_distribution(
        &self,
        df1: f64,
        df2: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    // ===== Shape Operations =====

    /// Concatenate tensors along a dimension
    ///
    /// Joins a sequence of tensors along an existing dimension. All tensors must
    /// have the same shape except in the concatenation dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensor references to concatenate
    /// * `dim` - Dimension along which to concatenate (supports negative indexing)
    ///
    /// # Returns
    ///
    /// New tensor containing the concatenated data
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0], &[2], &device);
    /// let b = Tensor::from_slice(&[3.0, 4.0, 5.0], &[3], &device);
    /// let c = client.cat(&[&a, &b], 0)?; // Shape: [5]
    /// ```
    fn cat(&self, tensors: &[&Tensor<R>], dim: isize) -> Result<Tensor<R>>;

    /// Stack tensors along a new dimension
    ///
    /// Joins a sequence of tensors along a new dimension. All tensors must have
    /// exactly the same shape.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensor references to stack
    /// * `dim` - Dimension at which to insert the new stacking dimension
    ///
    /// # Returns
    ///
    /// New tensor with an additional dimension
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0], &[2], &device);
    /// let b = Tensor::from_slice(&[3.0, 4.0], &[2], &device);
    /// let c = client.stack(&[&a, &b], 0)?; // Shape: [2, 2]
    /// ```
    fn stack(&self, tensors: &[&Tensor<R>], dim: isize) -> Result<Tensor<R>>;

    /// Split a tensor into chunks of a given size along a dimension
    ///
    /// Splits the tensor into chunks. The last chunk will be smaller if the
    /// dimension size is not evenly divisible by split_size.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to split
    /// * `split_size` - Size of each chunk (except possibly the last)
    /// * `dim` - Dimension along which to split (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Vector of tensor views (zero-copy) into the original tensor
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    /// let chunks = client.split(&a, 2, 0)?; // [2], [2], [1]
    /// ```
    fn split(&self, tensor: &Tensor<R>, split_size: usize, dim: isize) -> Result<Vec<Tensor<R>>>;

    /// Split a tensor into a specific number of chunks along a dimension
    ///
    /// Splits the tensor into approximately equal chunks. If the dimension
    /// is not evenly divisible, earlier chunks will be one element larger.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to chunk
    /// * `chunks` - Number of chunks to create
    /// * `dim` - Dimension along which to chunk (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Vector of tensor views (zero-copy) into the original tensor
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    /// let chunks = client.chunk(&a, 2, 0)?; // [3], [2]
    /// ```
    fn chunk(&self, tensor: &Tensor<R>, chunks: usize, dim: isize) -> Result<Vec<Tensor<R>>>;

    /// Repeat tensor along each dimension
    ///
    /// Creates a new tensor by repeating the input tensor along each dimension.
    /// The `repeats` slice specifies how many times to repeat along each dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `repeats` - Number of repetitions for each dimension. Length must match tensor ndim.
    ///
    /// # Returns
    ///
    /// New tensor with shape `[dim_0 * repeats[0], dim_1 * repeats[1], ...]`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
    /// let repeated = client.repeat(&a, &[2, 3])?; // Shape: [4, 6]
    /// // Result: [[1,2,1,2,1,2], [3,4,3,4,3,4], [1,2,1,2,1,2], [3,4,3,4,3,4]]
    /// ```
    fn repeat(&self, tensor: &Tensor<R>, repeats: &[usize]) -> Result<Tensor<R>>;

    /// Pad tensor with a constant value
    ///
    /// Adds padding to the tensor along specified dimensions. The `padding` slice
    /// contains pairs of (before, after) padding sizes, starting from the last dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `padding` - Padding sizes as pairs: `[last_before, last_after, second_last_before, ...]`
    /// * `value` - Value to use for padding
    ///
    /// # Returns
    ///
    /// New tensor with padded dimensions
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
    /// // Pad last dim by 1 on each side
    /// let padded = client.pad(&a, &[1, 1], 0.0)?; // Shape: [2, 4]
    /// // Result: [[0,1,2,0], [0,3,4,0]]
    /// ```
    fn pad(&self, tensor: &Tensor<R>, padding: &[usize], value: f64) -> Result<Tensor<R>>;

    /// Roll tensor elements along a dimension
    ///
    /// Shifts elements circularly along a dimension. Elements that roll beyond
    /// the last position wrap around to the first position.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `shift` - Number of positions to shift (negative = shift left, positive = shift right)
    /// * `dim` - Dimension along which to roll (supports negative indexing)
    ///
    /// # Returns
    ///
    /// New tensor with rolled elements
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let rolled = client.roll(&a, 1, 0)?; // [4, 1, 2, 3]
    /// let rolled = client.roll(&a, -1, 0)?; // [2, 3, 4, 1]
    /// ```
    fn roll(&self, tensor: &Tensor<R>, shift: isize, dim: isize) -> Result<Tensor<R>>;

    // ===== Linear Algebra =====

    /// Solve linear system Ax = b using LU decomposition
    ///
    /// Computes the solution x to the linear equation Ax = b, where A is a square
    /// coefficient matrix.
    ///
    /// # Algorithm
    ///
    /// Uses LU decomposition with partial pivoting:
    /// ```text
    /// 1. Compute PA = LU (pivoted LU decomposition)
    /// 2. Solve Ly = Pb (forward substitution)
    /// 3. Solve Ux = y (backward substitution)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix [n, n]
    /// * `b` - Right-hand side vector/matrix [n] or [n, k]
    ///
    /// # Returns
    ///
    /// Solution tensor x [n] or [n, k]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if dimensions are incompatible or A is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if matrix is singular (not invertible)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Solve 2x + 3y = 5
    /// //       4x + 5y = 11
    /// let a = Tensor::from_slice(&[2.0, 3.0, 4.0, 5.0], &[2, 2], &device);
    /// let b = Tensor::from_slice(&[5.0, 11.0], &[2], &device);
    /// let x = client.solve(&a, &b)?;
    /// // x = [2.0, 1.0]
    /// ```
    fn solve(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Least squares solution: minimize ||Ax - b||²
    ///
    /// Computes the solution x that minimizes the 2-norm of the residual ||Ax - b||².
    /// Uses QR decomposition (Householder reflections) followed by back-substitution.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute QR decomposition: A = QR
    /// 2. Transform: y = Q^T @ b
    /// 3. Solve: R @ x = y (back-substitution)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix [m, n] (can be non-square)
    /// * `b` - Right-hand side vector/matrix [m] or [m, k]
    ///
    /// # Returns
    ///
    /// Solution tensor x [n] or [n, k] that minimizes ||Ax - b||²
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if dimensions are incompatible
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Fit line y = mx + c to overdetermined system
    /// let a = Tensor::from_slice(&[1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[3, 2], &device);
    /// let b = Tensor::from_slice(&[2.0, 4.0, 6.0], &[3], &device);
    /// let x = client.lstsq(&a, &b)?; // [m, c]
    /// ```
    fn lstsq(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Moore-Penrose pseudo-inverse via SVD: A^+ = V @ diag(1/S) @ U^T
    ///
    /// Computes the pseudo-inverse of a matrix using SVD. For a matrix A with
    /// SVD decomposition A = U @ diag(S) @ V^T, the pseudo-inverse is:
    ///
    /// ```text
    /// A^+ = V @ diag(1/S_i where S_i > rcond*max(S), else 0) @ U^T
    /// ```
    ///
    /// # Algorithm
    ///
    /// 1. Compute SVD: A = U @ diag(S) @ V^T
    /// 2. Invert non-zero singular values: S_inv[i] = 1/S[i] if S[i] > rcond*max(S), else 0
    /// 3. Compute: A^+ = V @ diag(S_inv) @ U^T
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix [m, n]
    /// * `rcond` - Relative condition number threshold (singular values below rcond*max(S) are treated as zero)
    ///   If None, uses default: max(m,n) * machine_epsilon
    ///
    /// # Returns
    ///
    /// Pseudo-inverse matrix [n, m]
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    /// let a_pinv = client.pinverse(&a, None)?; // Shape: [3, 2]
    /// // Verify: a @ a_pinv @ a ≈ a
    /// ```
    fn pinverse(&self, a: &Tensor<R>, rcond: Option<f64>) -> Result<Tensor<R>>;

    /// Matrix norm
    ///
    /// Computes the matrix norm of the input tensor.
    ///
    /// # Supported Norms
    ///
    /// - **Frobenius**: `sqrt(sum(A[i,j]²))` - Euclidean norm of the matrix
    /// - **Spectral** (2-norm): Maximum singular value (requires SVD)
    /// - **Nuclear** (trace norm): Sum of singular values (requires SVD)
    ///
    /// # Algorithm
    ///
    /// **Frobenius norm:**
    /// ```text
    /// ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2) = sqrt(trace(A^T @ A))
    /// ```
    ///
    /// **Spectral norm:**
    /// ```text
    /// ||A||_2 = max singular value of A
    /// ```
    ///
    /// **Nuclear norm:**
    /// ```text
    /// ||A||_* = sum of singular values of A
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor
    /// * `ord` - Norm order (Frobenius, Spectral, Nuclear)
    ///
    /// # Returns
    ///
    /// Scalar tensor containing the norm value
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 2D
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use numr::algorithm::linalg::MatrixNormOrder;
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
    /// let fro = client.matrix_norm(&a, MatrixNormOrder::Frobenius)?;
    /// let spec = client.matrix_norm(&a, MatrixNormOrder::Spectral)?;
    /// ```
    fn matrix_norm(
        &self,
        a: &Tensor<R>,
        ord: crate::algorithm::linalg::MatrixNormOrder,
    ) -> Result<Tensor<R>>;

    /// Matrix inverse using LU decomposition
    ///
    /// Computes the multiplicative inverse of a square matrix.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute PA = LU (pivoted LU decomposition)
    /// 2. Solve for A^{-1}: each column j of A^{-1} solves A @ x_j = e_j
    ///    where e_j is the j-th standard basis vector
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Inverse matrix [n, n] such that A @ A^{-1} = I
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if matrix is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if matrix is singular (determinant = 0)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[4.0, 7.0, 2.0, 6.0], &[2, 2], &device);
    /// let a_inv = client.inverse(&a)?;
    /// // Verify: a @ a_inv ≈ I
    /// ```
    fn inverse(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix determinant using LU decomposition
    ///
    /// Computes the determinant of a square matrix.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute PA = LU (pivoted LU decomposition)
    /// 2. det(A) = (-1)^{number of row swaps} * product(diag(U))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Scalar tensor containing the determinant
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if matrix is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
    /// let det = client.det(&a)?;
    /// // det = 1*4 - 2*3 = -2
    /// ```
    fn det(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix trace: sum of diagonal elements
    ///
    /// Computes the sum of the diagonal elements of a matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Scalar tensor containing trace(A) = sum_i A[i,i]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if matrix is not square
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
    /// let tr = client.trace(&a)?;
    /// // tr = 1 + 4 = 5
    /// ```
    fn trace(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Extract diagonal elements
    ///
    /// Returns the diagonal elements of a 2D matrix as a 1D tensor.
    ///
    /// # Arguments
    ///
    /// * `a` - 2D matrix [m, n]
    ///
    /// # Returns
    ///
    /// 1D tensor [min(m,n)] containing diagonal elements
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 2D
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    /// let d = client.diag(&a)?;
    /// // d = [1, 5]
    /// ```
    fn diag(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Create diagonal matrix from 1D tensor
    ///
    /// Creates a 2D square matrix with the input elements on the diagonal.
    ///
    /// # Arguments
    ///
    /// * `a` - 1D tensor [n]
    ///
    /// # Returns
    ///
    /// 2D diagonal matrix [n, n]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 1D
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &device);
    /// let d = client.diagflat(&a)?;
    /// // d = [[1, 0, 0],
    /// //      [0, 2, 0],
    /// //      [0, 0, 3]]
    /// ```
    fn diagflat(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix rank via SVD
    ///
    /// Computes the numerical rank of a matrix (number of non-zero singular values).
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute SVD: A = U @ diag(S) @ V^T
    /// 2. Count singular values: rank = #{i : S[i] > tol}
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix [m, n]
    /// * `tol` - Singular value threshold (values below this are treated as zero)
    ///   If None, uses default: max(m,n) * eps * max(S)
    ///
    /// # Returns
    ///
    /// Scalar integer tensor containing the rank
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 2.0, 4.0], &[2, 2], &device);
    /// let rank = client.matrix_rank(&a, None)?;
    /// // rank = 1 (rank-deficient: rows are linearly dependent)
    /// ```
    fn matrix_rank(&self, a: &Tensor<R>, tol: Option<f64>) -> Result<Tensor<R>>;
}

// ScalarOps, CompareOps, LogicalOps moved to traits/ module
