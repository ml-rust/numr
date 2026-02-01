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
    LinalgOps, LogicalOps, MatmulOps, NormalizationOps, RandomOps, ReduceOps, ScalarOps, ShapeOps,
    SortingOps, StatisticalOps, TypeConversionOps, UtilityOps,
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
    + LinalgOps<R>
    + ShapeOps<R>
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
    // Moved to ShapeOps trait in traits/shape.rs

    // ===== Linear Algebra Operations =====
    // Moved to LinalgOps trait in traits/linalg.rs
}

// ScalarOps, CompareOps, LogicalOps moved to traits/ module
