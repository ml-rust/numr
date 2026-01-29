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

pub use activation::*;
pub use arithmetic::*;
pub use dispatch::*;
pub use matmul::*;
pub use reduce::*;
pub use special::SpecialFunctions;

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

    /// Fused matrix multiplication with bias addition: C = A @ B + bias
    ///
    /// This is a fused operation that combines matrix multiplication and bias addition
    /// into a single kernel, avoiding an extra memory round-trip compared to separate
    /// `matmul` followed by `add`. This is the core operation for neural network linear
    /// layers: `output = input @ weight.T + bias`.
    ///
    /// # Algorithm (Epilogue Fusion)
    ///
    /// The bias addition is fused into the GEMM epilogue:
    /// ```text
    /// 1. Load tiles of A and B into shared memory
    /// 2. Compute partial products, accumulate in registers
    /// 3. Repeat for all K tiles
    /// 4. EPILOGUE: For each output element C[i][j]:
    ///    C[i][j] = accumulated_value[i][j] + bias[j]
    /// 5. Write final result to global memory
    /// ```
    ///
    /// This saves one global memory read/write cycle vs the naive:
    /// ```text
    /// temp = A @ B       // Write temp to global memory
    /// C = temp + bias    // Read temp, write C
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of shape `[..., M, K]`
    /// * `b` - Weight tensor of shape `[..., K, N]`
    /// * `bias` - Bias tensor of shape `[N]` (1D, broadcast across all M rows)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., M, N]` where `C[..., i, j] = sum_k(A[..., i, k] * B[..., k, j]) + bias[j]`
    ///
    /// # Errors
    ///
    /// Returns `Error::ShapeMismatch` if:
    /// - Inner dimensions don't match (A's last dim != B's second-to-last dim)
    /// - Bias shape doesn't match output columns (bias.len() != N)
    /// - Bias is not 1D
    ///
    /// Returns `Error::DTypeMismatch` if A, B, and bias don't have the same dtype.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Linear layer: output = input @ weight.T + bias
    /// let input = Tensor::randn(&[batch, seq_len, hidden], DType::F32, &device);
    /// let weight = Tensor::randn(&[out_features, hidden], DType::F32, &device);
    /// let bias = Tensor::randn(&[out_features], DType::F32, &device);
    ///
    /// // Using fused operation (faster):
    /// let output = client.matmul_bias(&input, &weight.transpose(-1, -2)?, &bias)?;
    ///
    /// // Equivalent to (slower - extra memory round-trip):
    /// let temp = client.matmul(&input, &weight.transpose(-1, -2)?)?;
    /// let output = client.add(&temp, &bias.unsqueeze(0)?.unsqueeze(0)?)?;
    /// ```
    ///
    /// # Performance
    ///
    /// Fusing bias into the GEMM epilogue provides:
    /// - ~2x memory bandwidth reduction for the bias addition
    /// - Better cache utilization (output stays in registers)
    /// - Reduced kernel launch overhead (one kernel instead of two)
    ///
    /// For large matrices where GEMM is compute-bound, the speedup is modest.
    /// For smaller matrices (typical in LLM inference), the speedup is more significant.
    ///
    /// # Backend Support
    ///
    /// | Backend | Supported DTypes | Tensor Dims | Notes |
    /// |---------|------------------|-------------|-------|
    /// | CPU     | All dtypes       | 2D, 3D+     | Full support via generic kernels |
    /// | CUDA    | F32, F64, F16, BF16 | 2D, 3D+ | Returns `UnsupportedDType` for integers |
    /// | WebGPU  | F32, I32, U32, F16 | 2D, 3D only | Returns error for >3D tensors |
    ///
    /// Integer dtypes (I32, I64, U32, U64) are only supported on CPU.
    /// CUDA returns `Error::UnsupportedDType` for integer matmul_bias operations.
    /// WebGPU is limited to 3D workgroup dispatches and returns an error for >3D tensors.
    fn matmul_bias(&self, a: &Tensor<R>, b: &Tensor<R>, bias: &Tensor<R>) -> Result<Tensor<R>>;

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

    /// Product along specified dimensions
    ///
    /// Computes the product of elements along the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor containing product values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let result = client.prod(&a, &[0], false)?; // 24.0
    /// ```
    fn prod(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Product along specified dimensions with explicit accumulation precision.
    ///
    /// Accumulation precision is especially important for products as values
    /// can grow or shrink exponentially, causing overflow or underflow.
    ///
    /// See [`AccumulationPrecision`] for details on precision options.
    fn prod_with_precision(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<R>>;

    /// Test if any element is true (non-zero) along specified dimensions.
    ///
    /// Returns true (1) if any element is non-zero along the specified dimensions,
    /// false (0) otherwise. This operation performs logical OR reduction.
    ///
    /// # Truth Value Semantics by DType
    ///
    /// The "truthiness" of a value depends on its dtype:
    ///
    /// | DType | False | True |
    /// |-------|-------|------|
    /// | Bool | `false` / 0 | `true` / 1 |
    /// | F32, F64, F16, BF16 | `0.0`, `-0.0` | Any non-zero value, including **NaN** and **±Inf** |
    /// | I8, I16, I32, I64 | `0` | Any non-zero value (positive or negative) |
    /// | U8, U16, U32, U64 | `0` | Any non-zero value |
    /// | FP8 variants | `0.0` | Any non-zero value |
    ///
    /// # Important: NaN Handling
    ///
    /// **NaN is considered truthy (non-zero).** This follows the convention that
    /// `any` checks if values are non-zero, not whether they are valid numbers.
    /// If you need to exclude NaN values, filter them first with `nan_to_num` or
    /// check with `isnan`.
    ///
    /// ```ignore
    /// // NaN is truthy
    /// let a = Tensor::from_slice(&[0.0, f32::NAN, 0.0], &[3], &device);
    /// let result = client.any(&a, &[0], false)?; // 1.0 (true, because NaN ≠ 0)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of any supported dtype
    /// * `dims` - Dimensions to reduce over (empty = reduce over all dimensions)
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor with the same dtype as input, containing:
    /// - `1` (or `1.0` for floats) where any element is non-zero
    /// - `0` (or `0.0` for floats) where all elements are zero
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Float tensor - standard case
    /// let a = Tensor::from_slice(&[0.0, 0.0, 1.0, 0.0], &[4], &device);
    /// let result = client.any(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // Integer tensor
    /// let a = Tensor::from_slice(&[0i32, 0, -5, 0], &[4], &device);
    /// let result = client.any(&a, &[0], false)?; // 1 (true, -5 ≠ 0)
    ///
    /// // All zeros
    /// let a = Tensor::from_slice(&[0.0, 0.0, 0.0], &[3], &device);
    /// let result = client.any(&a, &[0], false)?; // 0.0 (false)
    ///
    /// // With infinity
    /// let a = Tensor::from_slice(&[0.0, f32::INFINITY], &[2], &device);
    /// let result = client.any(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // 2D tensor - reduce along rows
    /// let a = Tensor::from_slice(&[0.0, 1.0, 0.0, 0.0], &[2, 2], &device);
    /// let result = client.any(&a, &[1], false)?; // [1.0, 0.0]
    /// ```
    ///
    /// # See Also
    ///
    /// * [`all`] - Test if all elements are true (logical AND)
    /// * [`sum`] - For counting non-zero elements, consider `sum(a != 0)`
    fn any(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Test if all elements are true (non-zero) along specified dimensions.
    ///
    /// Returns true (1) if all elements are non-zero along the specified dimensions,
    /// false (0) otherwise. This operation performs logical AND reduction.
    ///
    /// # Truth Value Semantics by DType
    ///
    /// The "truthiness" of a value depends on its dtype:
    ///
    /// | DType | False | True |
    /// |-------|-------|------|
    /// | Bool | `false` / 0 | `true` / 1 |
    /// | F32, F64, F16, BF16 | `0.0`, `-0.0` | Any non-zero value, including **NaN** and **±Inf** |
    /// | I8, I16, I32, I64 | `0` | Any non-zero value (positive or negative) |
    /// | U8, U16, U32, U64 | `0` | Any non-zero value |
    /// | FP8 variants | `0.0` | Any non-zero value |
    ///
    /// # Important: NaN Handling
    ///
    /// **NaN is considered truthy (non-zero).** This follows the convention that
    /// `all` checks if values are non-zero, not whether they are valid numbers.
    /// A tensor of all NaN values will return true. If you need to check for
    /// valid (non-NaN) values, use `isnan` first.
    ///
    /// ```ignore
    /// // All NaN values → true (all are non-zero)
    /// let a = Tensor::from_slice(&[f32::NAN, f32::NAN], &[2], &device);
    /// let result = client.all(&a, &[0], false)?; // 1.0 (true, because NaN ≠ 0)
    ///
    /// // NaN mixed with zero → false
    /// let a = Tensor::from_slice(&[f32::NAN, 0.0], &[2], &device);
    /// let result = client.all(&a, &[0], false)?; // 0.0 (false, because 0.0 == 0)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of any supported dtype
    /// * `dims` - Dimensions to reduce over (empty = reduce over all dimensions)
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor with the same dtype as input, containing:
    /// - `1` (or `1.0` for floats) where all elements are non-zero
    /// - `0` (or `0.0` for floats) where any element is zero
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Float tensor - all non-zero
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let result = client.all(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // Float tensor - contains zero
    /// let a = Tensor::from_slice(&[1.0, 0.0, 3.0, 4.0], &[4], &device);
    /// let result = client.all(&a, &[0], false)?; // 0.0 (false)
    ///
    /// // Integer tensor - negative values are truthy
    /// let a = Tensor::from_slice(&[-1i32, -2, -3], &[3], &device);
    /// let result = client.all(&a, &[0], false)?; // 1 (true)
    ///
    /// // With infinity - Inf is truthy
    /// let a = Tensor::from_slice(&[f32::INFINITY, f32::NEG_INFINITY], &[2], &device);
    /// let result = client.all(&a, &[0], false)?; // 1.0 (true)
    ///
    /// // 2D tensor - reduce along rows
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 0.0], &[2, 2], &device);
    /// let result = client.all(&a, &[1], false)?; // [1.0, 0.0]
    ///
    /// // Empty dimension reduction - edge case
    /// let a = Tensor::from_slice(&[1.0, 2.0], &[2, 1], &device);
    /// let result = client.all(&a, &[1], false)?; // [1.0, 1.0] (single element is truthy)
    /// ```
    ///
    /// # See Also
    ///
    /// * [`any`] - Test if any element is true (logical OR)
    /// * [`prod`] - Product reduction (different from logical AND)
    fn all(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    // ===== Cumulative Operations =====

    /// Cumulative sum along a dimension
    ///
    /// Returns the cumulative sum of elements along the specified dimension.
    /// For input [a, b, c, d], output is [a, a+b, a+b+c, a+b+c+d].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to compute cumulative sum (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input containing cumulative sums
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let result = client.cumsum(&a, 0)?; // [1, 3, 6, 10]
    /// ```
    fn cumsum(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>;

    /// Cumulative product along a dimension
    ///
    /// Returns the cumulative product of elements along the specified dimension.
    /// For input [a, b, c, d], output is [a, a*b, a*b*c, a*b*c*d].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to compute cumulative product (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input containing cumulative products
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let result = client.cumprod(&a, 0)?; // [1, 2, 6, 24]
    /// ```
    fn cumprod(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>;

    /// Log-sum-exp along specified dimensions (numerically stable)
    ///
    /// Computes log(sum(exp(x))) in a numerically stable way:
    /// logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    ///
    /// This is commonly used in softmax computation and log-probability calculations.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor containing log-sum-exp values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &device);
    /// let result = client.logsumexp(&a, &[0], false)?;
    /// // result ≈ log(exp(1) + exp(2) + exp(3)) ≈ 3.4076
    /// ```
    fn logsumexp(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

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

    /// Look up embeddings from an embedding table using indices.
    ///
    /// This is the standard embedding lookup operation used in neural networks
    /// for word embeddings, entity embeddings, etc. It is equivalent to
    /// `index_select(embeddings, 0, indices)` but optimized for the common case
    /// where the embedding table is 2D and indices index into the first dimension.
    ///
    /// # Algorithm
    ///
    /// For each index value i in the indices tensor:
    /// ```text
    /// output[..., i, :] = embeddings[indices[..., i], :]
    /// ```
    ///
    /// The output shape is `indices.shape() + [embedding_dim]` where `embedding_dim`
    /// is `embeddings.shape()[1]`.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - 2D embedding table of shape `[vocab_size, embedding_dim]`
    /// * `indices` - Index tensor of any shape containing indices into the embedding table.
    ///   Must be I64 (or I32 on WebGPU). Values must be in range `[0, vocab_size)`.
    ///
    /// # Returns
    ///
    /// Tensor of shape `indices.shape() + [embedding_dim]` containing the looked-up embeddings.
    ///
    /// # Example
    ///
    /// ```text
    /// embeddings = [[1.0, 2.0],   # word 0
    ///               [3.0, 4.0],   # word 1
    ///               [5.0, 6.0]]   # word 2
    /// indices = [2, 0, 1]
    ///
    /// output = [[5.0, 6.0],   # word 2
    ///           [1.0, 2.0],   # word 0
    ///           [3.0, 4.0]]   # word 1
    /// ```
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if embeddings is not 2D
    /// * `DTypeMismatch` - if indices is not I64 (or I32 on WebGPU)
    /// * Index out of bounds results in undefined behavior (implementation may return zeros)
    ///
    /// # Performance
    ///
    /// On GPU, this operation is memory-bound and optimized for coalesced reads
    /// from the embedding table. Each thread handles one index lookup and writes
    /// a full embedding vector.
    fn embedding_lookup(&self, embeddings: &Tensor<R>, indices: &Tensor<R>) -> Result<Tensor<R>>;

    // ===== Sorting and Search Operations =====

    /// Sort tensor along a dimension.
    ///
    /// Returns the sorted values. For both values and indices, use `sort_with_indices`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to sort (supports negative indexing)
    /// * `descending` - If true, sort in descending order
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input, containing sorted values along the specified dimension.
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], &device);
    /// let sorted = client.sort(&a, 0, false)?; // [1.0, 1.0, 3.0, 4.0, 5.0]
    /// ```
    fn sort(&self, a: &Tensor<R>, dim: isize, descending: bool) -> Result<Tensor<R>>;

    /// Sort tensor along a dimension, returning both sorted values and indices.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to sort (supports negative indexing)
    /// * `descending` - If true, sort in descending order
    ///
    /// # Returns
    ///
    /// Tuple of (sorted_values, indices) where:
    /// - `sorted_values`: Tensor with same shape and dtype as input, containing sorted values
    /// - `indices`: I64 tensor with same shape, containing original indices
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[3.0, 1.0, 4.0], &[3], &device);
    /// let (values, indices) = client.sort_with_indices(&a, 0, false)?;
    /// // values = [1.0, 3.0, 4.0]
    /// // indices = [1, 0, 2]
    /// ```
    fn sort_with_indices(
        &self,
        a: &Tensor<R>,
        dim: isize,
        descending: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Return indices that would sort the tensor along a dimension.
    ///
    /// Equivalent to `sort_with_indices(...).1`, but more efficient when only
    /// indices are needed.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to compute sort indices (supports negative indexing)
    /// * `descending` - If true, return indices for descending order
    ///
    /// # Returns
    ///
    /// I64 tensor with same shape as input, containing indices that would sort the tensor.
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[3.0, 1.0, 4.0], &[3], &device);
    /// let indices = client.argsort(&a, 0, false)?; // [1, 0, 2]
    /// // a[indices] would give [1.0, 3.0, 4.0]
    /// ```
    fn argsort(&self, a: &Tensor<R>, dim: isize, descending: bool) -> Result<Tensor<R>>;

    /// Return top K largest (or smallest) values and their indices along a dimension.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `k` - Number of top elements to return
    /// * `dim` - Dimension along which to find top-k (supports negative indexing)
    /// * `largest` - If true, return largest elements; if false, return smallest
    /// * `sorted` - If true, return in sorted order; if false, maintain relative input order
    ///
    /// # Returns
    ///
    /// Tuple of (values, indices) where:
    /// - `values`: Tensor with shape [..., k, ...] (dim replaced with k), same dtype as input
    /// - `indices`: I64 tensor with same shape as values, containing original indices
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if k > dim_size.
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], &device);
    /// let (values, indices) = client.topk(&a, 2, 0, true, true)?;
    /// // values = [5.0, 4.0] (largest 2, sorted)
    /// // indices = [4, 2]
    /// ```
    fn topk(
        &self,
        a: &Tensor<R>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Return unique elements of the input tensor.
    ///
    /// Flattens the tensor and returns unique elements in sorted order.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `sorted` - If true (default), return unique elements in sorted order
    ///
    /// # Returns
    ///
    /// 1D tensor containing unique elements. Length may vary depending on input.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 2.0, 3.0, 1.0], &[5], &device);
    /// let unique = client.unique(&a, true)?; // [1.0, 2.0, 3.0]
    /// ```
    fn unique(&self, a: &Tensor<R>, sorted: bool) -> Result<Tensor<R>>;

    /// Return unique elements with inverse indices and counts.
    ///
    /// More complete version of `unique` that also returns:
    /// - Inverse indices: for each element in input, index into unique output
    /// - Counts: how many times each unique element appears
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// Tuple of (unique, inverse_indices, counts) where:
    /// - `unique`: 1D tensor of unique values (sorted)
    /// - `inverse_indices`: I64 tensor with same shape as flattened input
    /// - `counts`: I64 tensor with same length as unique, containing occurrence counts
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 2.0, 3.0, 1.0], &[5], &device);
    /// let (unique, inverse, counts) = client.unique_with_counts(&a)?;
    /// // unique = [1.0, 2.0, 3.0]
    /// // inverse = [0, 1, 1, 2, 0] (maps each input to index in unique)
    /// // counts = [2, 2, 1] (1.0 appears 2x, 2.0 appears 2x, 3.0 appears 1x)
    /// ```
    fn unique_with_counts(&self, a: &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// Return indices of non-zero elements.
    ///
    /// Returns a 2D tensor where each row contains the indices of a non-zero element.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// I64 tensor of shape [N, ndim] where N is the number of non-zero elements
    /// and ndim is the number of dimensions in the input tensor. Each row contains
    /// the multi-dimensional index of a non-zero element.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[0.0, 1.0, 0.0, 2.0], &[2, 2], &device);
    /// let indices = client.nonzero(&a)?;
    /// // indices = [[0, 1], [1, 1]] (positions of 1.0 and 2.0)
    /// ```
    fn nonzero(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Find insertion points for values in a sorted sequence.
    ///
    /// For each value in `values`, finds the index in `sorted_sequence` where the value
    /// should be inserted to maintain sorted order.
    ///
    /// # Arguments
    ///
    /// * `sorted_sequence` - 1D sorted tensor
    /// * `values` - Values to search for (any shape)
    /// * `right` - If true, find rightmost insertion point; if false, leftmost
    ///
    /// # Returns
    ///
    /// I64 tensor with same shape as `values`, containing insertion indices.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sorted = Tensor::from_slice(&[1.0, 3.0, 5.0, 7.0], &[4], &device);
    /// let values = Tensor::from_slice(&[2.0, 4.0, 6.0], &[3], &device);
    /// let indices = client.searchsorted(&sorted, &values, false)?;
    /// // indices = [1, 2, 3] (insert positions to maintain order)
    /// ```
    fn searchsorted(
        &self,
        sorted_sequence: &Tensor<R>,
        values: &Tensor<R>,
        right: bool,
    ) -> Result<Tensor<R>>;

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

    // ===== Complex Number Operations =====

    /// Complex conjugate: conj(a + bi) = a - bi
    ///
    /// Returns the complex conjugate of the input tensor.
    /// For real tensors, returns the input unchanged.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (Complex64, Complex128, or real types)
    ///
    /// # Returns
    ///
    /// * Complex types: Tensor with same shape and dtype, imaginary part negated
    /// * Real types: Returns input tensor unchanged (real numbers equal their conjugate)
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends (identity operation)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let conj_z = client.conj(&z)?;
    /// // Result: [1.0 - 2.0i, 3.0 + 4.0i]
    /// ```
    fn conj(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Extract real part of complex tensor: real(a + bi) = a
    ///
    /// Extracts the real component from a complex tensor.
    /// For real tensors, returns a copy of the input.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with same shape
    /// * Complex128 input → F64 tensor with same shape
    /// * Real input → Copy of input tensor
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let re = client.real(&z)?;  // F32 tensor: [1.0, 3.0]
    /// ```
    fn real(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Extract imaginary part of complex tensor: imag(a + bi) = b
    ///
    /// Extracts the imaginary component from a complex tensor.
    /// For real tensors, returns a zero tensor with the same shape.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with same shape
    /// * Complex128 input → F64 tensor with same shape
    /// * Real input → Zero tensor with same shape and dtype
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let im = client.imag(&z)?;  // F32 tensor: [2.0, 4.0]
    /// ```
    fn imag(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute phase angle of complex tensor: angle(a + bi) = atan2(b, a)
    ///
    /// Returns the phase angle (argument) of complex numbers in radians.
    /// The result is in the range [-π, π].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with angles in radians
    /// * Complex128 input → F64 tensor with angles in radians
    /// * Real input → Zero tensor (real numbers have phase angle 0 for positive, π for negative)
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 1.0), Complex64::new(-1.0, 0.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let angles = client.angle(&z)?;  // F32 tensor: [π/4, π]
    /// ```
    ///
    /// # Mathematical Notes
    ///
    /// For complex z = a + bi, returns atan2(b, a) in radians [-π, π].
    /// For real x, returns 0 if x ≥ 0, π if x < 0.
    /// To compute magnitude, use abs(z) = sqrt(re² + im²) separately.
    fn angle(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

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

    /// Compute the q-th quantile along a dimension
    ///
    /// Returns the value at the given quantile (0.0 to 1.0) along the specified dimension.
    /// The input is sorted along the dimension, then the value at the quantile position
    /// is computed using the specified interpolation method.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `q` - Quantile to compute, must be in [0.0, 1.0]
    /// * `dim` - Dimension to reduce (None = flatten first)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    /// * `interpolation` - Method for interpolating between data points:
    ///   - "linear" (default): Linear interpolation between adjacent values
    ///   - "lower": Use lower index value
    ///   - "higher": Use higher index value
    ///   - "nearest": Use nearest index value
    ///   - "midpoint": Average of lower and higher values
    ///
    /// # Returns
    ///
    /// Tensor with quantile values. Shape is the same as input with the specified
    /// dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Sort input along dimension
    /// 2. Compute index: idx = q * (n - 1) where n = dimension size
    /// 3. Interpolate based on method:
    ///    - linear: result = sorted[floor] * (1 - frac) + sorted[ceil] * frac
    ///    - lower: result = sorted[floor(idx)]
    ///    - higher: result = sorted[ceil(idx)]
    ///    - nearest: result = sorted[round(idx)]
    ///    - midpoint: result = (sorted[floor] + sorted[ceil]) / 2
    /// ```
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    /// let median = client.quantile(&a, 0.5, Some(0), false, "linear")?;  // 3.0
    /// let q25 = client.quantile(&a, 0.25, Some(0), false, "linear")?;    // 2.0
    /// let q75 = client.quantile(&a, 0.75, Some(0), false, "linear")?;    // 4.0
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidArgument` if q is outside [0.0, 1.0]
    /// - `InvalidArgument` if interpolation is not a valid method
    /// - `InvalidAxis` if dim is out of bounds
    fn quantile(
        &self,
        a: &Tensor<R>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<R>>;

    /// Compute the p-th percentile along a dimension
    ///
    /// Convenience wrapper for `quantile(a, p/100, dim, keepdim, "linear")`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `p` - Percentile to compute, must be in [0.0, 100.0]
    /// * `dim` - Dimension to reduce (None = flatten first)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    /// let p50 = client.percentile(&a, 50.0, Some(0), false)?;  // 3.0 (median)
    /// let p25 = client.percentile(&a, 25.0, Some(0), false)?;  // 2.0
    /// ```
    fn percentile(
        &self,
        a: &Tensor<R>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<R>>;

    /// Compute median (50th percentile) along a dimension
    ///
    /// Returns the median value along the specified dimension.
    /// Equivalent to `quantile(a, 0.5, dim, keepdim, "linear")`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension to reduce (None = flatten first)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 3.0, 2.0, 5.0, 4.0], &[5], &device);
    /// let med = client.median(&a, Some(0), false)?;  // 3.0
    ///
    /// let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let med = client.median(&b, Some(0), false)?;  // 2.5 (interpolated)
    /// ```
    fn median(&self, a: &Tensor<R>, dim: Option<isize>, keepdim: bool) -> Result<Tensor<R>>;

    /// Compute histogram of input values
    ///
    /// Counts the number of values that fall into each of the specified bins.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (flattened internally)
    /// * `bins` - Number of equal-width bins
    /// * `range` - Optional (min, max) range for bins. Defaults to (a.min(), a.max())
    ///
    /// # Returns
    ///
    /// Tuple of (histogram, bin_edges):
    /// - histogram: I64 tensor of shape [bins] with counts
    /// - bin_edges: Tensor of shape [bins + 1] with bin boundaries
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Determine range: [min, max] (from input or provided)
    /// 2. Compute bin_width = (max - min) / bins
    /// 3. For each value x:
    ///    bin_idx = floor((x - min) / bin_width)
    ///    Clamp to [0, bins-1]
    ///    counts[bin_idx]++
    /// 4. bin_edges = [min, min+w, min+2w, ..., max]
    /// ```
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[0.5, 1.5, 2.5, 1.0, 2.0], &[5], &device);
    /// let (hist, edges) = client.histogram(&a, 3, None)?;
    /// // hist = [1, 2, 2] for bins [0.5-1.17), [1.17-1.83), [1.83-2.5]
    /// // edges = [0.5, 1.17, 1.83, 2.5]
    /// ```
    fn histogram(
        &self,
        a: &Tensor<R>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Covariance matrix of observations
    ///
    /// Computes the covariance matrix from a matrix where each row is an observation
    /// and each column is a variable (feature).
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [n_samples, n_features]
    /// * `ddof` - Delta degrees of freedom. Default: 1 (sample covariance)
    ///
    /// # Returns
    ///
    /// Covariance matrix [n_features, n_features]
    ///
    /// # Properties
    ///
    /// - Symmetric: cov[i,j] = cov[j,i]
    /// - Diagonal elements are variances: cov[i,i] = var(X[:,i])
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 3 samples, 2 features
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
    /// let c = client.cov(&x, None)?;  // [2, 2] covariance matrix
    /// ```
    fn cov(&self, a: &Tensor<R>, ddof: Option<usize>) -> Result<Tensor<R>>;

    /// Pearson correlation coefficient matrix
    ///
    /// Computes the correlation coefficient matrix from observations.
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [n_samples, n_features]
    ///
    /// # Returns
    ///
    /// Correlation matrix [n_features, n_features] with values in [-1, 1]
    ///
    /// # Properties
    ///
    /// - Diagonal elements are 1.0 (unless feature has zero variance)
    /// - Off-diagonal in [-1, 1]: correlation coefficient
    /// - Symmetric: corr[i,j] = corr[j,i]
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
    /// let corr = client.corrcoef(&x)?;  // [2, 2] correlation matrix
    /// ```
    fn corrcoef(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Skewness (third standardized moment)
    ///
    /// Measures the asymmetry of the distribution. Positive skew indicates a tail
    /// extending toward positive values; negative skew indicates a tail toward
    /// negative values.
    ///
    /// # Formula
    ///
    /// ```text
    /// skew = E[(X - mean)³] / std³
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over (empty = all dimensions)
    /// * `keepdim` - If true, keep reduced dimensions as size 1
    /// * `correction` - Degrees of freedom correction (default 0)
    ///
    /// # Returns
    ///
    /// Tensor containing skewness values
    ///
    /// # Interpretation
    ///
    /// - skew ≈ 0: Symmetric distribution
    /// - skew > 0: Right-skewed (tail toward positive)
    /// - skew < 0: Left-skewed (tail toward negative)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Symmetric distribution
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    /// let s = client.skew(&a, &[], false, 0)?;  // ≈ 0.0
    /// ```
    fn skew(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<R>>;

    /// Kurtosis (fourth standardized moment, excess)
    ///
    /// Measures the "tailedness" of the distribution. Higher kurtosis indicates
    /// heavier tails and sharper peak; lower kurtosis indicates lighter tails.
    ///
    /// Returns excess kurtosis (kurtosis - 3), so a normal distribution has
    /// kurtosis of 0.
    ///
    /// # Formula
    ///
    /// ```text
    /// kurtosis = E[(X - mean)⁴] / std⁴ - 3
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over (empty = all dimensions)
    /// * `keepdim` - If true, keep reduced dimensions as size 1
    /// * `correction` - Degrees of freedom correction (default 0)
    ///
    /// # Returns
    ///
    /// Tensor containing excess kurtosis values
    ///
    /// # Interpretation
    ///
    /// - kurtosis ≈ 0: Normal-like tails (mesokurtic)
    /// - kurtosis > 0: Heavy tails, sharp peak (leptokurtic)
    /// - kurtosis < 0: Light tails, flat peak (platykurtic)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Normal distribution has excess kurtosis ≈ 0
    /// let normal_data = client.randn(&[10000], DType::F32)?;
    /// let k = client.kurtosis(&normal_data, &[], false, 0)?;  // ≈ 0.0
    /// ```
    fn kurtosis(
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
