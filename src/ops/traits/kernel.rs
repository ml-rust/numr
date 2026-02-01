//! Low-level kernel trait for compute operations.

use crate::dtype::Element;
use crate::ops::{BinaryOp, ReduceOp, UnaryOp};
use crate::runtime::Runtime;

/// Low-level typed kernels for compute operations
///
/// This trait defines the actual compute kernels that operate on typed pointers.
/// It is generic over `T: Element` for code reuse and specialization via
/// monomorphization.
///
/// # Safety Contract
///
/// All kernel methods are unsafe because they operate on raw pointers.
/// Callers must ensure:
/// - Pointers are valid and properly aligned
/// - Lengths accurately describe the buffer sizes
/// - No aliasing violations (output doesn't overlap with inputs)
///
/// # Example Implementation
///
/// ```ignore
/// struct CpuKernel;
///
/// impl Kernel<CpuRuntime> for CpuKernel {
///     unsafe fn binary_op<T: Element>(
///         &self,
///         op: BinaryOp,
///         a: *const T,
///         b: *const T,
///         out: *mut T,
///         len: usize,
///     ) {
///         for i in 0..len {
///             let av = *a.add(i);
///             let bv = *b.add(i);
///             *out.add(i) = match op {
///                 BinaryOp::Add => av + bv,
///                 BinaryOp::Sub => av - bv,
///                 // ...
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
