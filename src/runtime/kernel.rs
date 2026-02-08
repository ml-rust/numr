//! Compile-time enforced typed kernel traits
//!
//! This module provides `TypedKernel<T>` trait that MUST be implemented for each
//! dtype a backend supports. Missing implementations cause compile errors, not
//! runtime UnsupportedDType errors.
//!
//! # Architecture
//!
//! ```text
//! dispatch_dtype!(dtype, T => {
//!     client.binary_op::<T>(...)  // Calls TypedKernel<T>::binary_op
//! })
//!                  │
//!                  ▼
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                     TypedKernel<T: Element>                               │
//! │  Compile-time trait that each backend implements PER DTYPE               │
//! └──────────────────────────────────────────────────────────────────────────┘
//!                  │
//!     ┌────────────┼────────────┬────────────┐
//!     ▼            ▼            ▼            ▼
//! TypedKernel<f32> TypedKernel<i32> TypedKernel<u32> TypedKernel<f64>
//!     │            │            │            │
//!     ▼            ▼            ▼            ▼
//! Backend-specific implementation (WGSL shader, CUDA kernel, SIMD)
//! ```
//!
//! # Why This Design?
//!
//! Previously, backends could silently fail at runtime by returning `UnsupportedDType`.
//! This led to bugs like WGPU only supporting F32 while claiming to support all types.
//!
//! With `TypedKernel<T>`:
//! - Each backend MUST implement the trait for each dtype it supports
//! - If `TypedKernel<i32>` is missing, code using I32 won't compile
//! - The compiler enforces complete dtype coverage
//!
//! # Backend Requirements
//!
//! Each backend must implement `TypedKernel<T>` for ALL dtypes it can handle:
//!
//! | Backend | Required TypedKernel<T> implementations |
//! |---------|----------------------------------------|
//! | CPU     | f32, f64, i32, i64, u32, u64, i16, i8, u16, u8, (f16, bf16, fp8) |
//! | CUDA    | f32, f64, i32, i64, u32, u64, i16, i8, u16, u8, (f16, bf16, fp8) |
//! | WebGPU  | f32, i32, u32, (f16 with extension) |
//!
//! If a backend doesn't implement a dtype, code using that dtype won't compile.
//!
//! # Example
//!
//! ```ignore
//! // Backend MUST implement for each dtype
//! impl TypedKernel<f32> for WgpuClient {
//!     fn binary_op(&self, op: BinaryOp, a: u64, b: u64, out: u64, len: usize) {
//!         // Launch WGSL shader for f32
//!     }
//! }
//!
//! impl TypedKernel<i32> for WgpuClient {
//!     fn binary_op(&self, op: BinaryOp, a: u64, b: u64, out: u64, len: usize) {
//!         // Launch WGSL shader for i32
//!     }
//! }
//!
//! // If this is missing, any code using u32 tensors won't compile:
//! // impl TypedKernel<u32> for WgpuClient { ... }  // COMPILE ERROR if missing
//! ```

use crate::dtype::Element;
use crate::error::Result;
use crate::ops::{BinaryOp, ReduceOp, UnaryOp};

/// Typed kernel trait for element-wise and reduction operations.
///
/// Each backend MUST implement this trait for EVERY dtype it supports.
/// Missing implementations cause compile errors when code uses that dtype.
///
/// # Type Parameter
///
/// `T: Element` - The element type this implementation handles (f32, i32, etc.)
///
/// # Implementation Notes
///
/// - All pointers (`a`, `b`, `out`) are device handles (u64 for GPU, pointer as u64 for CPU)
/// - Operations are dispatched through `dispatch_dtype!` which calls the appropriate impl
/// - Backends should cache compiled kernels/pipelines by `(op, dtype)` key
pub trait TypedKernel<T: Element>: Send + Sync {
    /// Element-wise binary operation: out[i] = a[i] op b[i]
    ///
    /// # Arguments
    /// * `op` - Binary operation to perform (Add, Sub, Mul, Div, Pow, Max, Min)
    /// * `a` - Handle to first input buffer
    /// * `b` - Handle to second input buffer
    /// * `out` - Handle to output buffer
    /// * `len` - Number of elements
    fn binary_op(&self, op: BinaryOp, a: u64, b: u64, out: u64, len: usize) -> Result<()>;

    /// Element-wise unary operation: out[i] = op(a[i])
    ///
    /// # Arguments
    /// * `op` - Unary operation to perform (Neg, Abs, Sqrt, Exp, Log, Sin, Cos, etc.)
    /// * `a` - Handle to input buffer
    /// * `out` - Handle to output buffer
    /// * `len` - Number of elements
    fn unary_op(&self, op: UnaryOp, a: u64, out: u64, len: usize) -> Result<()>;

    /// Scalar operation: out[i] = a[i] op scalar
    ///
    /// # Arguments
    /// * `op` - Binary operation to apply with scalar
    /// * `a` - Handle to input buffer
    /// * `scalar` - Scalar value (as f64, cast to T internally)
    /// * `out` - Handle to output buffer
    /// * `len` - Number of elements
    fn scalar_op(&self, op: BinaryOp, a: u64, scalar: f64, out: u64, len: usize) -> Result<()>;

    /// Reduction operation along contiguous dimension
    ///
    /// # Arguments
    /// * `op` - Reduction operation (Sum, Mean, Max, Min, Prod)
    /// * `a` - Handle to input buffer
    /// * `out` - Handle to output buffer
    /// * `reduce_size` - Size of dimension being reduced
    /// * `outer_size` - Product of all other dimensions
    fn reduce(
        &self,
        op: ReduceOp,
        a: u64,
        out: u64,
        reduce_size: usize,
        outer_size: usize,
    ) -> Result<()>;

    /// Fill buffer with a constant value
    ///
    /// # Arguments
    /// * `out` - Handle to output buffer
    /// * `value` - Value to fill with (as f64, cast to T internally)
    /// * `len` - Number of elements
    fn fill(&self, out: u64, value: f64, len: usize) -> Result<()>;

    /// Copy elements from src to dst
    ///
    /// # Arguments
    /// * `src` - Handle to source buffer
    /// * `dst` - Handle to destination buffer
    /// * `len` - Number of elements
    fn copy(&self, src: u64, dst: u64, len: usize) -> Result<()>;
}

/// Typed matrix multiplication kernel.
///
/// Separate trait because matmul has different parameters than element-wise ops.
pub trait TypedMatmul<T: Element>: Send + Sync {
    /// Matrix multiplication: C = A @ B
    ///
    /// # Arguments
    /// * `a` - Handle to matrix A (m × k)
    /// * `b` - Handle to matrix B (k × n)
    /// * `out` - Handle to output matrix C (m × n)
    /// * `m`, `n`, `k` - Matrix dimensions
    /// * `lda`, `ldb`, `ldc` - Leading dimensions (row strides)
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        a: u64,
        b: u64,
        out: u64,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    ) -> Result<()>;

    /// Batched matrix multiplication: C[b] = A[b] @ B[b]
    ///
    /// # Arguments
    /// * `a` - Handle to batched matrix A (batch × m × k)
    /// * `b` - Handle to batched matrix B (batch × k × n)
    /// * `out` - Handle to output C (batch × m × n)
    /// * `batch_size` - Number of matrices in batch
    /// * `m`, `n`, `k` - Matrix dimensions per batch
    #[allow(clippy::too_many_arguments)]
    fn batched_matmul(
        &self,
        a: u64,
        b: u64,
        out: u64,
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
}

/// Typed normalization kernels.
pub trait TypedNorm<T: Element>: Send + Sync {
    /// RMS Normalization
    ///
    /// # Arguments
    /// * `input` - Handle to input buffer
    /// * `weight` - Handle to weight buffer
    /// * `out` - Handle to output buffer
    /// * `batch_size` - Number of rows to normalize
    /// * `hidden_size` - Size of last dimension
    /// * `eps` - Epsilon for numerical stability
    fn rms_norm(
        &self,
        input: u64,
        weight: u64,
        out: u64,
        batch_size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()>;

    /// Layer Normalization
    ///
    /// # Arguments
    /// * `input` - Handle to input buffer
    /// * `weight` - Handle to weight (gamma) buffer
    /// * `bias` - Handle to bias (beta) buffer
    /// * `out` - Handle to output buffer
    /// * `batch_size` - Number of rows to normalize
    /// * `hidden_size` - Size of last dimension
    /// * `eps` - Epsilon for numerical stability
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        input: u64,
        weight: u64,
        bias: u64,
        out: u64,
        batch_size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()>;
}

/// Typed comparison kernels.
///
/// Returns F32 tensor (0.0 for false, 1.0 for true).
/// Note: Uses F32 because WGSL bool cannot be used in storage buffers.
pub trait TypedCompare<T: Element>: Send + Sync {
    /// Element-wise comparison
    ///
    /// # Arguments
    /// * `op` - Comparison operation (Eq, Ne, Lt, Le, Gt, Ge)
    /// * `a` - Handle to first input buffer
    /// * `b` - Handle to second input buffer
    /// * `out` - Handle to output buffer (F32: 0.0=false, 1.0=true)
    /// * `len` - Number of elements
    fn compare(&self, op: CompareOp, a: u64, b: u64, out: u64, len: usize) -> Result<()>;
}

/// Comparison operations for TypedCompare trait
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal: a == b
    Eq,
    /// Not equal: a != b
    Ne,
    /// Less than: a < b
    Lt,
    /// Less than or equal: a <= b
    Le,
    /// Greater than: a > b
    Gt,
    /// Greater than or equal: a >= b
    Ge,
}

// ============================================================================
// Backend Requirement Traits - COMPILE-TIME ENFORCEMENT
// ============================================================================

/// Marker trait for backends that support all WebGPU-compatible dtypes.
///
/// WebGPU backends MUST implement this trait, which requires:
/// - TypedKernel<f32>
/// - TypedKernel<i32>
/// - TypedKernel<u32>
/// - TypedMatmul<f32>
/// - TypedNorm<f32>
///
/// If any of these are missing, the backend won't compile.
///
/// # Example
///
/// ```ignore
/// // This MUST be present for WgpuClient to compile:
/// impl WgpuKernels for WgpuClient {}
///
/// // This causes a compile error if any TypedKernel<T> is missing
/// ```
pub trait WgpuKernels:
    TypedKernel<f32>
    + TypedKernel<i32>
    + TypedKernel<u32>
    + TypedMatmul<f32>
    + TypedMatmul<i32>
    + TypedMatmul<u32>
    + TypedNorm<f32>
    + TypedCompare<f32>
    + TypedCompare<i32>
    + TypedCompare<u32>
{
}

/// Marker trait for backends that support all CPU/CUDA dtypes.
///
/// CPU and CUDA backends MUST implement this trait, which requires TypedKernel
/// for ALL dtypes: f32, f64, i32, i64, u32, u64, i16, i8, u16, u8.
///
/// Feature-gated dtypes (F16, BF16, FP8) are checked separately.
pub trait FullKernels:
    TypedKernel<f32>
    + TypedKernel<f64>
    + TypedKernel<i32>
    + TypedKernel<i64>
    + TypedKernel<u32>
    + TypedKernel<u64>
    + TypedKernel<i16>
    + TypedKernel<i8>
    + TypedKernel<u16>
    + TypedKernel<u8>
    + TypedMatmul<f32>
    + TypedMatmul<f64>
    + TypedNorm<f32>
    + TypedNorm<f64>
    + TypedCompare<f32>
    + TypedCompare<f64>
    + TypedCompare<i32>
    + TypedCompare<i64>
    + TypedCompare<u32>
    + TypedCompare<u64>
{
}

// ============================================================================
// Dispatch Helpers
// ============================================================================

/// Dispatch a typed kernel call based on runtime dtype.
///
/// This macro is used internally to dispatch from `DType` enum to the
/// appropriate `TypedKernel<T>` implementation.
///
/// # Example
///
/// ```ignore
/// fn binary_op_dispatch<R: Runtime>(
///     client: &impl TypedKernel<f32> + TypedKernel<i32>,
///     op: BinaryOp,
///     a: u64,
///     b: u64,
///     out: u64,
///     len: usize,
///     dtype: DType,
/// ) -> Result<()> {
///     dispatch_typed_kernel!(client, dtype, T => {
///         client.binary_op::<T>(op, a, b, out, len)
///     })
/// }
/// ```
#[macro_export]
macro_rules! dispatch_typed_kernel {
    ($client:expr, $dtype:expr, $T:ident => $body:expr, $op:expr) => {
        match $dtype {
            $crate::dtype::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
            $crate::dtype::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::dtype::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::dtype::DType::U32 => {
                type $T = u32;
                $body
            }
            $crate::dtype::DType::U64 => {
                type $T = u64;
                $body
            }
            $crate::dtype::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::dtype::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::dtype::DType::U16 => {
                type $T = u16;
                $body
            }
            $crate::dtype::DType::U8 => {
                type $T = u8;
                $body
            }
            #[cfg(feature = "f16")]
            $crate::dtype::DType::F16 => {
                type $T = half::f16;
                $body
            }
            #[cfg(feature = "f16")]
            $crate::dtype::DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            #[cfg(not(feature = "f16"))]
            $crate::dtype::DType::F16 | $crate::dtype::DType::BF16 => {
                return Err($crate::error::Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $op,
                })
            }
            $crate::dtype::DType::FP8E4M3 => {
                type $T = $crate::dtype::FP8E4M3;
                $body
            }
            $crate::dtype::DType::FP8E5M2 => {
                type $T = $crate::dtype::FP8E5M2;
                $body
            }
            $crate::dtype::DType::Bool => {
                return Err($crate::error::Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $op,
                })
            }
        }
    };
}

pub use dispatch_typed_kernel;

#[cfg(test)]
mod tests {
    use super::*;

    // Test that CompareOp is properly defined
    #[test]
    fn test_compare_op_variants() {
        let ops = [
            CompareOp::Eq,
            CompareOp::Ne,
            CompareOp::Lt,
            CompareOp::Le,
            CompareOp::Gt,
            CompareOp::Ge,
        ];
        assert_eq!(ops.len(), 6);
    }
}
