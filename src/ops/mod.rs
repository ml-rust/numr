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

pub use activation::*;
pub use arithmetic::*;
pub use dispatch::*;
pub use matmul::*;
pub use reduce::*;

use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Low-Level Kernel Trait (Layer 0)
// ============================================================================

/// Low-level typed kernels for compute operations
///
/// This trait defines the actual compute kernels that operate on typed pointers.
/// It is generic over `T: Element` for code reuse and specialization via
/// monomorphization.
///
/// Backend implementations (CPU, CUDA, WGPU) implement this trait with
/// optimized kernels for each operation and element type.
///
/// # Design (per TDD §3.0.2)
///
/// ```text
/// TensorOps (dtype-agnostic API)
///     │
///     │ match dtype { F32 => ..., F16 => ... }
///     ▼
/// Kernel<T: Element> (typed operations)
///     │
///     ▼
/// Backend-specific SIMD/GPU code
/// ```
///
/// # Example Implementation
///
/// ```ignore
/// impl Kernel<CpuRuntime> for CpuClient {
///     fn binary_op<T: Element>(
///         &self,
///         op: BinaryOp,
///         a: *const T,
///         b: *const T,
///         out: *mut T,
///         len: usize,
///     ) {
///         // Current: scalar loops for all types
///         // Future: SIMD-optimized paths for f32/f64
///         for i in 0..len {
///             unsafe {
///                 *out.add(i) = apply_op(op, *a.add(i), *b.add(i));
///             }
///         }
///     }
/// }
/// ```
pub trait Kernel<R: Runtime>: Send + Sync {
    /// Element-wise binary operation
    ///
    /// # Safety
    /// - `a`, `b`, and `out` must be valid pointers to `len` elements
    /// - `out` must not overlap with `a` or `b` unless they are the same pointer
    unsafe fn binary_op<T: Element>(
        &self,
        op: BinaryOp,
        a: *const T,
        b: *const T,
        out: *mut T,
        len: usize,
    );

    /// Element-wise unary operation
    ///
    /// # Safety
    /// - `a` and `out` must be valid pointers to `len` elements
    unsafe fn unary_op<T: Element>(&self, op: UnaryOp, a: *const T, out: *mut T, len: usize);

    /// Matrix multiplication: C = A @ B
    ///
    /// Computes C[m, n] = sum_k(A[m, k] * B[k, n])
    ///
    /// # Arguments
    /// * `a` - Pointer to matrix A (m × k)
    /// * `b` - Pointer to matrix B (k × n)
    /// * `out` - Pointer to output matrix C (m × n)
    /// * `m`, `n`, `k` - Matrix dimensions
    /// * `lda`, `ldb`, `ldc` - Leading dimensions (strides)
    ///
    /// # Safety
    /// - All pointers must be valid for the specified dimensions
    #[allow(clippy::too_many_arguments)] // Matrix ops inherently need dimension params
    unsafe fn matmul<T: Element>(
        &self,
        a: *const T,
        b: *const T,
        out: *mut T,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    );

    /// Reduction along contiguous dimension
    ///
    /// # Arguments
    /// * `op` - Reduction operation (Sum, Mean, Max, Min, Prod)
    /// * `a` - Input pointer
    /// * `out` - Output pointer
    /// * `reduce_size` - Size of the dimension being reduced
    /// * `outer_size` - Product of all other dimensions
    ///
    /// # Safety
    /// - `a` must point to `reduce_size * outer_size` elements
    /// - `out` must point to `outer_size` elements
    unsafe fn reduce<T: Element>(
        &self,
        op: ReduceOp,
        a: *const T,
        out: *mut T,
        reduce_size: usize,
        outer_size: usize,
    );

    /// Fill buffer with a constant value
    ///
    /// # Safety
    /// - `out` must be a valid pointer to `len` elements
    unsafe fn fill<T: Element>(&self, out: *mut T, value: T, len: usize);

    /// Copy elements from src to dst
    ///
    /// # Safety
    /// - `src` and `dst` must be valid pointers to `len` elements
    /// - `dst` must not overlap with `src`
    unsafe fn copy<T: Element>(&self, src: *const T, dst: *mut T, len: usize);
}

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
pub trait TensorOps<R: Runtime> {
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

    /// Negation: -a
    fn neg(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Absolute value: |a|
    fn abs(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Square root: sqrt(a)
    fn sqrt(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Exponential: e^a
    fn exp(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Natural logarithm: ln(a)
    fn log(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Sine: sin(a)
    fn sin(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Cosine: cos(a)
    fn cos(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Hyperbolic tangent: tanh(a)
    fn tanh(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Tangent: tan(a)
    fn tan(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Reciprocal: 1/a
    fn recip(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Square: a²
    fn square(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Floor: floor(a)
    fn floor(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Ceiling: ceil(a)
    fn ceil(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Round: round(a) to nearest integer
    fn round(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Sign: returns -1 for negative, 0 for zero, 1 for positive
    fn sign(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

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

    // ===== Matrix Operations =====

    /// Matrix multiplication: a @ b
    ///
    /// Supports batched matmul for tensors with more than 2 dimensions.
    fn matmul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Reductions =====

    /// Sum along specified dimensions
    fn sum(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Sum along specified dimensions with explicit accumulation precision.
    ///
    /// See [`AccumulationPrecision`] for details on precision options.
    fn sum_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    /// Mean along specified dimensions
    fn mean(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Maximum along specified dimensions
    fn max(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Maximum along specified dimensions with explicit accumulation precision.
    ///
    /// See [`AccumulationPrecision`] for details on precision options.
    fn max_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    /// Minimum along specified dimensions
    fn min(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Minimum along specified dimensions with explicit accumulation precision.
    ///
    /// See [`AccumulationPrecision`] for details on precision options.
    fn min_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    // ===== Activations =====

    /// Rectified linear unit: max(0, a)
    fn relu(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Sigmoid: 1 / (1 + e^(-a))
    fn sigmoid(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// SiLU (Swish): a * sigmoid(a) = a / (1 + e^(-a))
    ///
    /// Used in LLaMA, Mistral, and other modern transformer architectures.
    fn silu(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// GELU (Gaussian Error Linear Unit): 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    ///
    /// Uses the tanh approximation. Used in GPT, BERT, and other transformer architectures.
    fn gelu(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Leaky ReLU: max(negative_slope * a, a)
    ///
    /// Allows small gradients for negative inputs, helping prevent "dying ReLU" problem.
    /// Default negative_slope is typically 0.01.
    fn leaky_relu(&self, a: &Tensor<R>, negative_slope: f64) -> Result<Tensor<R>>;

    /// ELU (Exponential Linear Unit): a if a > 0, else alpha * (exp(a) - 1)
    ///
    /// Smooth approximation to ReLU with negative values saturating to -alpha.
    /// Default alpha is typically 1.0.
    fn elu(&self, a: &Tensor<R>, alpha: f64) -> Result<Tensor<R>>;

    /// Softmax along a dimension
    fn softmax(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>;

    // ===== Normalization =====

    /// RMS Normalization: output = input * rsqrt(mean(input^2) + eps) * weight
    ///
    /// RMSNorm is used in LLaMA and other modern transformer architectures.
    /// It normalizes over the last dimension.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [..., hidden_size]
    /// * `weight` - Weight tensor of shape [hidden_size]
    /// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
    fn rms_norm(&self, input: &Tensor<R>, weight: &Tensor<R>, eps: f32) -> Result<Tensor<R>>;

    /// Layer Normalization: output = (input - mean) / sqrt(variance + eps) * weight + bias
    ///
    /// LayerNorm normalizes across the last dimension for each batch element.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [..., hidden_size]
    /// * `weight` - Weight (gamma) tensor of shape [hidden_size]
    /// * `bias` - Bias (beta) tensor of shape [hidden_size]
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    fn layer_norm(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: &Tensor<R>,
        eps: f32,
    ) -> Result<Tensor<R>>;

    // ===== Index Operations =====

    /// Argmax: returns indices of maximum values along a dimension.
    ///
    /// Returns a tensor of I64 indices indicating the position of the maximum
    /// value along the specified dimension. The output shape is the input shape
    /// with the specified dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to find the maximum index
    /// * `keepdim` - If true, the reduced dimension is retained with size 1
    ///
    /// # Returns
    ///
    /// Tensor of I64 containing indices of maximum values
    fn argmax(&self, a: &Tensor<R>, dim: usize, keepdim: bool) -> Result<Tensor<R>>;

    /// Argmin: returns indices of minimum values along a dimension.
    ///
    /// Returns a tensor of I64 indices indicating the position of the minimum
    /// value along the specified dimension. The output shape is the input shape
    /// with the specified dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to find the minimum index
    /// * `keepdim` - If true, the reduced dimension is retained with size 1
    ///
    /// # Returns
    ///
    /// Tensor of I64 containing indices of minimum values
    fn argmin(&self, a: &Tensor<R>, dim: usize, keepdim: bool) -> Result<Tensor<R>>;

    /// Gather elements along a dimension using an index tensor.
    ///
    /// For a 3D tensor with dim=1:
    /// `out[i][j][k] = input[i][index[i][j][k]][k]`
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to gather
    /// * `index` - Index tensor (I64) with same number of dimensions as input
    ///
    /// # Returns
    ///
    /// Tensor with same shape as index tensor, same dtype as input
    fn gather(&self, a: &Tensor<R>, dim: usize, index: &Tensor<R>) -> Result<Tensor<R>>;

    /// Scatter values into a tensor at positions specified by an index tensor.
    ///
    /// Creates a new tensor (copy of `a`) with values from `src` scattered at positions
    /// specified by `index` along dimension `dim`.
    ///
    /// For a 3D tensor with dim=1:
    /// `out[i][index[i][j][k]][k] = src[i][j][k]`
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (values to scatter into)
    /// * `dim` - Dimension along which to scatter
    /// * `index` - Index tensor (I64) specifying scatter positions
    /// * `src` - Source tensor with values to scatter
    ///
    /// # Returns
    ///
    /// New tensor with scattered values
    fn scatter(
        &self,
        a: &Tensor<R>,
        dim: usize,
        index: &Tensor<R>,
        src: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Select elements along a dimension using a 1D index tensor.
    ///
    /// Simpler than gather - the index tensor is 1D and applies to all positions
    /// in the specified dimension.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to select
    /// * `index` - 1D index tensor (I64) of length m
    ///
    /// # Returns
    ///
    /// Tensor with dimension `dim` having size m (length of index)
    fn index_select(&self, a: &Tensor<R>, dim: usize, index: &Tensor<R>) -> Result<Tensor<R>>;

    /// Select elements where mask is true, returning a flattened 1D tensor.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `mask` - Boolean mask tensor (U8: 0=false, non-zero=true), must be broadcastable to `a`
    ///
    /// # Returns
    ///
    /// 1D tensor containing only elements where mask is true
    fn masked_select(&self, a: &Tensor<R>, mask: &Tensor<R>) -> Result<Tensor<R>>;

    /// Fill elements where mask is true with a scalar value.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `mask` - Boolean mask tensor (U8: 0=false, non-zero=true), must be broadcastable to `a`
    /// * `value` - Value to fill where mask is true
    ///
    /// # Returns
    ///
    /// New tensor with masked positions filled with value
    fn masked_fill(&self, a: &Tensor<R>, mask: &Tensor<R>, value: f64) -> Result<Tensor<R>>;

    // ===== Type Conversion =====

    /// Cast tensor to a different data type.
    ///
    /// Converts all elements of the input tensor to the target dtype.
    /// The output tensor has the same shape as the input.
    ///
    /// # Supported Conversions
    ///
    /// - **Widening** (lossless): I8→I16→I32→I64, F16→F32→F64
    /// - **Narrowing** (may lose precision): F64→F32→F16, I64→I32→I16→I8
    /// - **Float↔Int**: Truncates toward zero for float→int
    /// - **FP8 conversions**: F32↔FP8E4M3, F32↔FP8E5M2 (with saturation)
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dtype` - Target data type
    ///
    /// # Returns
    ///
    /// New tensor with the specified dtype
    ///
    /// # Errors
    ///
    /// Returns `UnsupportedDType` if the conversion is not supported
    /// (e.g., Bool↔numeric without explicit handling).
    fn cast(&self, a: &Tensor<R>, dtype: crate::dtype::DType) -> Result<Tensor<R>>;

    // ===== Conditional Operations =====

    /// Conditional select: where(cond, x, y) = cond ? x : y
    ///
    /// Performs element-wise conditional selection. For each position,
    /// returns x if condition is true (non-zero), otherwise y.
    ///
    /// # Arguments
    ///
    /// * `cond` - Condition tensor (U8: 0 = false, non-zero = true)
    /// * `x` - Values to select when condition is true
    /// * `y` - Values to select when condition is false
    ///
    /// # Returns
    ///
    /// Tensor with same shape and dtype as x and y
    fn where_cond(&self, cond: &Tensor<R>, x: &Tensor<R>, y: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Utility Operations =====

    /// Clamp tensor values to a range: clamp(x, min, max) = min(max(x, min), max)
    ///
    /// Element-wise clamps each value to be within [min_val, max_val].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `min_val` - Minimum value (inclusive)
    /// * `max_val` - Maximum value (inclusive)
    ///
    /// # Returns
    ///
    /// Tensor with same shape and dtype as input, with values clamped to range
    fn clamp(&self, a: &Tensor<R>, min_val: f64, max_val: f64) -> Result<Tensor<R>>;

    /// Fill tensor with a constant value
    ///
    /// Creates a new tensor with the specified shape and dtype, filled with the given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `value` - Value to fill the tensor with
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// New tensor filled with the constant value
    fn fill(&self, shape: &[usize], value: f64, dtype: crate::dtype::DType) -> Result<Tensor<R>>;

    /// Create a 1D tensor with evenly spaced values within a half-open interval [start, stop)
    ///
    /// Values are generated using the formula: start + step * i for i in 0..n
    /// where n = ceil((stop - start) / step)
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the interval (inclusive)
    /// * `stop` - End of the interval (exclusive)
    /// * `step` - Spacing between values (must be positive if start < stop, negative if start > stop)
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// 1D tensor with evenly spaced values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = client.arange(0.0, 5.0, 1.0, DType::F32)?; // [0, 1, 2, 3, 4]
    /// let t = client.arange(0.0, 5.0, 2.0, DType::F32)?; // [0, 2, 4]
    /// let t = client.arange(5.0, 0.0, -1.0, DType::F32)?; // [5, 4, 3, 2, 1]
    /// ```
    fn arange(
        &self,
        start: f64,
        stop: f64,
        step: f64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Create a 1D tensor with evenly spaced values over a specified interval
    ///
    /// Returns `steps` evenly spaced values from `start` to `stop` (inclusive).
    /// Values are: start + (stop - start) * i / (steps - 1) for i in 0..steps
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the interval
    /// * `stop` - End of the interval (inclusive)
    /// * `steps` - Number of values to generate (must be >= 2)
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// 1D tensor with evenly spaced values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = client.linspace(0.0, 10.0, 5, DType::F32)?; // [0, 2.5, 5, 7.5, 10]
    /// let t = client.linspace(0.0, 1.0, 3, DType::F64)?; // [0, 0.5, 1]
    /// ```
    fn linspace(
        &self,
        start: f64,
        stop: f64,
        steps: usize,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Create a 2D identity matrix (or batch of identity matrices)
    ///
    /// Creates a tensor where the diagonal elements are 1 and all others are 0.
    /// For rectangular matrices, the diagonal is the main diagonal.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows
    /// * `m` - Number of columns (if None, defaults to n for square matrix)
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// 2D tensor of shape [n, m] with ones on the diagonal
    ///
    /// # Example
    ///
    /// ```ignore
    /// let eye = client.eye(3, None, DType::F32)?;    // 3x3 identity matrix
    /// let rect = client.eye(2, Some(4), DType::F32)?; // 2x4 matrix with diagonal ones
    /// ```
    fn eye(&self, n: usize, m: Option<usize>, dtype: crate::dtype::DType) -> Result<Tensor<R>>;

    // ===== Statistical Operations =====

    /// Variance along specified dimensions
    ///
    /// Computes the variance of elements along the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    /// * `correction` - Degrees of freedom correction (0 for population, 1 for sample)
    ///
    /// # Returns
    ///
    /// Tensor containing variance values
    fn var(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<R>>;

    /// Standard deviation along specified dimensions
    ///
    /// Computes the standard deviation (sqrt of variance) along the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    /// * `correction` - Degrees of freedom correction (0 for population, 1 for sample)
    ///
    /// # Returns
    ///
    /// Tensor containing standard deviation values
    fn std(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<R>>;

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
}

/// Scalar operations trait for tensor-scalar operations
pub trait ScalarOps<R: Runtime>: TensorOps<R> {
    /// Add scalar to tensor: a + scalar
    fn add_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Subtract scalar from tensor: a - scalar
    fn sub_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Multiply tensor by scalar: a * scalar
    fn mul_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Divide tensor by scalar: a / scalar
    fn div_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Raise tensor to scalar power: a^scalar
    fn pow_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;
}

/// Comparison operations trait
pub trait CompareOps<R: Runtime>: TensorOps<R> {
    /// Element-wise equality: a == b
    fn eq(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise inequality: a != b
    fn ne(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise less than: a < b
    fn lt(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise less than or equal: a <= b
    fn le(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise greater than: a > b
    fn gt(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise greater than or equal: a >= b
    fn ge(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
}

/// Logical operations trait for boolean tensors
///
/// All operations work on U8 tensors where 0 = false, non-zero = true.
pub trait LogicalOps<R: Runtime>: TensorOps<R> {
    /// Element-wise logical AND: a && b
    fn logical_and(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise logical OR: a || b
    fn logical_or(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise logical XOR: a ^ b
    fn logical_xor(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise logical NOT: !a
    fn logical_not(&self, a: &Tensor<R>) -> Result<Tensor<R>>;
}
