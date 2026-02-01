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
    LogicalOps, MatmulOps, NormalizationOps, RandomOps, ReduceOps, ScalarOps, SortingOps,
    StatisticalOps, TypeConversionOps, UtilityOps,
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
    + RandomOps<R>
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
    // Moved to RandomOps trait in traits/random.rs

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
