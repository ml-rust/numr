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

pub(crate) mod activation;
mod arithmetic;
pub(crate) mod common;
mod dispatch;
pub(crate) mod distance_common;
mod matmul;
pub(crate) mod reduce;
mod special;
pub mod traits;

pub use activation::*;
pub use arithmetic::*;
pub use dispatch::*;
pub use matmul::*;
pub use reduce::*;
pub use special::SpecialFunctions;
pub use traits::{
    ActivationOps, AdvancedRandomOps, BinaryOps, CompareOps, ComplexOps, ConditionalOps,
    CumulativeOps, DistanceMetric, DistanceOps, IndexingOps, Kernel, LinalgOps, LogicalOps,
    MatmulOps, NormalizationOps, QuasiRandomOps, RandomOps, ReduceOps, ScalarOps, ShapeOps,
    SortingOps, StatisticalOps, TypeConversionOps, UnaryOps, UtilityOps,
};

use crate::runtime::Runtime;

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
    + DistanceOps<R>
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
    + UnaryOps<R>
    + BinaryOps<R>
{
    // ===== Element-wise Binary Operations =====
    // Moved to BinaryOps trait in traits/binary.rs

    // ===== Element-wise Unary Operations =====
    // Moved to UnaryOps trait in traits/unary.rs

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
