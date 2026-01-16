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

    /// Mean along specified dimensions
    fn mean(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Maximum along specified dimensions
    fn max(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

    /// Minimum along specified dimensions
    fn min(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;

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
