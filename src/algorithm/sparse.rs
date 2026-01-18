//! Sparse algorithm contracts for backend consistency
//!
//! This module defines traits that ensure all backends implement the same
//! mathematical algorithms for sparse operations. This guarantees numerical
//! parity across CPU, CUDA, WebGPU, and other backends.
//!
//! # Design Principles
//!
//! 1. **Algorithm-Level Contract**: Each trait method represents a specific algorithm
//! 2. **Backend Parity**: Same algorithm must produce same results (within FP tolerance)
//! 3. **Explicit Contracts**: Missing implementations cause compile errors
//! 4. **Testability**: Easy to verify all backends implement the same algorithm

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::{CscData, CsrData};
use crate::tensor::Tensor;

/// Algorithmic contract for sparse matrix operations
///
/// All backends implementing sparse operations MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
///
/// # Algorithms Defined
///
/// - **ESC SpGEMM**: Exact Symbolic Computation with Hash Accumulation
///   - Phase 1: Symbolic (count NNZ per output row using HashSet)
///   - Phase 2: Numeric (compute values with pre-sized HashMap, filter by zero_tolerance)
///
/// - **Column-Parallel DSMM**: Dense × Sparse Matrix Multiplication
///   - For each column j of sparse matrix B (CSC format)
///   - For each non-zero B[k,j] in column j
///   - Compute C[:,j] += A[:,k] * B[k,j]
///
/// # Implementation Requirements
///
/// Backends may differ in:
/// - Parallelization strategy (threads, SIMD, GPU blocks)
/// - Memory access patterns (coalescing, vectorization)
/// - Loop unrolling and compiler optimizations
///
/// Backends MUST match in:
/// - Mathematical formula (same operations in same order)
/// - Accumulator precision (e.g., F32 accumulator for F16 inputs)
/// - Special case handling (NaN, Inf, zero_tolerance filtering)
/// - Result ordering (sorted by column index for SpGEMM)
pub trait SparseAlgorithms<R: Runtime> {
    /// ESC SpGEMM: Sparse × Sparse matrix multiplication using ESC algorithm
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Phase 1 (Symbolic):
    ///   For each output row i:
    ///     Use HashSet to find unique columns in C[i,:]
    ///     Store count in row_nnz[i]
    ///   Build row_ptrs via exclusive scan
    ///
    /// Phase 2 (Numeric):
    ///   For each output row i:
    ///     Create HashMap<col_index, value> with capacity = row_nnz[i]
    ///     For each non-zero A[i,k]:
    ///       For each non-zero B[k,j]:
    ///         Accumulate: map[j] += A[i,k] * B[k,j]
    ///     Sort entries by column index
    ///     Filter values below zero_tolerance
    ///     Write to output arrays
    /// ```
    fn esc_spgemm_csr(&self, a_csr: &CsrData<R>, b_csr: &CsrData<R>) -> Result<CsrData<R>>;

    /// Column-Parallel DSMM: Dense × Sparse matrix multiplication
    ///
    /// # Algorithm
    ///
    /// ```text
    /// For each column j in B (0..N):
    ///   For each non-zero B[k,j] in column j:
    ///     C[:,j] += A[:,k] * B[k,j]
    /// ```
    fn column_parallel_dsmm(
        &self,
        dense_a: &Tensor<R>,
        sparse_b_csc: &CscData<R>,
    ) -> Result<Tensor<R>>;
}

// ============================================================================
// Helper Functions (Shared across backends)
// ============================================================================

/// Zero tolerance threshold for filtering small values
pub use crate::runtime::sparse_utils::zero_tolerance;

/// Validate CSR matrix dimensions for SpGEMM
pub fn validate_spgemm_shapes(
    a_shape: [usize; 2],
    b_shape: [usize; 2],
) -> Result<([usize; 2], usize)> {
    let [m, k_a] = a_shape;
    let [k_b, n] = b_shape;

    if k_a != k_b {
        return Err(crate::error::Error::ShapeMismatch {
            expected: vec![k_a],
            got: vec![k_b],
        });
    }

    Ok(([m, n], k_a))
}

/// Validate Dense × Sparse dimensions for DSMM
pub fn validate_dsmm_shapes(
    dense_shape: &[usize],
    sparse_shape: [usize; 2],
) -> Result<([usize; 2], usize)> {
    use crate::error::Error;

    if dense_shape.len() != 2 {
        return Err(Error::Internal(format!(
            "Expected 2D tensor for dense matrix, got {}D",
            dense_shape.len()
        )));
    }

    let m = dense_shape[0];
    let k_a = dense_shape[1];
    let [k_b, n] = sparse_shape;

    if k_a != k_b {
        return Err(Error::ShapeMismatch {
            expected: vec![k_a],
            got: vec![k_b],
        });
    }

    Ok(([m, n], k_a))
}

/// Validate dtypes match for sparse operations
pub fn validate_dtype_match(lhs: DType, rhs: DType) -> Result<()> {
    if lhs != rhs {
        return Err(crate::error::Error::DTypeMismatch { lhs, rhs });
    }
    Ok(())
}
