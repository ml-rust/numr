//! Linear algebra algorithm contracts for backend consistency
//!
//! This module defines traits that ensure all backends implement the same
//! mathematical algorithms for linear algebra operations. This guarantees
//! numerical parity across CPU, CUDA, WebGPU, and other backends.
//!
//! # Design Principles
//!
//! 1. **Algorithm-Level Contract**: Each trait method represents a specific algorithm
//! 2. **Backend Parity**: Same algorithm must produce same results (within FP tolerance)
//! 3. **Explicit Contracts**: Missing implementations cause compile errors
//! 4. **Testability**: Easy to verify all backends implement the same algorithm
//!
//! # Why Not Vendor Libraries?
//!
//! numr must work WITHOUT cuSOLVER/MKL/LAPACK. Native implementations are required.
//!
//! # Module Structure
//!
//! - `decompositions`: Result types (LuDecomposition, QrDecomposition, etc.)
//! - `traits`: LinearAlgebraAlgorithms and MatrixFunctionsAlgorithms traits
//! - `helpers`: Validation utilities
//! - `matrix_functions_core`: Shared numerical algorithms for matrix functions

pub mod decompositions;
pub mod helpers;
pub mod matrix_functions_core;
pub mod traits;

// Re-export all public types for convenient access
pub use decompositions::*;
pub use helpers::*;
pub use traits::*;

// ============================================================================
// Matrix Norm Orders
// ============================================================================

/// Matrix norm order specification.
///
/// Different norms are appropriate for different use cases:
///
/// - **Frobenius**: General-purpose norm, similar to Euclidean distance.
///   Use for measuring overall matrix magnitude or computing loss functions.
///
/// - **Spectral**: Maximum amplification factor of the matrix.
///   Use for stability analysis, condition number estimation, or bounding
///   operator effects in neural networks.
///
/// - **Nuclear**: Sum of singular values (trace norm).
///   Use for matrix rank approximation, low-rank regularization, or
///   compressed sensing applications.
///
/// # Examples
///
/// ```ignore
/// use numr::algorithm::linalg::MatrixNormOrder;
///
/// // Frobenius norm: measures overall magnitude
/// // ||A||_F = sqrt(sum(A[i,j]²))
/// let fro_norm = client.matrix_norm(&matrix, MatrixNormOrder::Frobenius)?;
///
/// // Spectral norm: largest singular value (operator norm)
/// // ||A||_2 = sigma_max(A)
/// let spec_norm = client.matrix_norm(&matrix, MatrixNormOrder::Spectral)?;
///
/// // Nuclear norm: sum of singular values
/// // ||A||_* = sum(sigma_i)
/// let nuc_norm = client.matrix_norm(&matrix, MatrixNormOrder::Nuclear)?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNormOrder {
    /// Frobenius norm: sqrt(sum(A[i,j]²))
    ///
    /// The Frobenius norm treats the matrix as a flattened vector and computes
    /// its Euclidean length. It's always available since it only requires
    /// element-wise square, sum, and sqrt operations.
    Frobenius,

    /// Spectral norm (2-norm): maximum singular value
    ///
    /// The spectral norm equals the largest singular value of the matrix,
    /// which represents the maximum factor by which the matrix can stretch
    /// any input vector. Requires SVD computation.
    Spectral,

    /// Nuclear norm (trace norm): sum of singular values
    ///
    /// The nuclear norm equals the sum of all singular values. It's the
    /// tightest convex relaxation of matrix rank and is used in low-rank
    /// matrix recovery algorithms. Requires SVD computation.
    Nuclear,
}
