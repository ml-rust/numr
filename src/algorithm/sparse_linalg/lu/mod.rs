//! Sparse LU Factorization
//!
//! Gilbert-Peierls left-looking algorithm for sparse LU factorization with
//! partial pivoting: PA = LU
//!
//! # Algorithm
//!
//! The Gilbert-Peierls algorithm processes the matrix column by column:
//!
//! ```text
//! For each column k = 0 to n-1:
//!   1. Initialize: x = A[:, k]
//!   2. Sparse triangular solve: solve L[0:k, 0:k] * y = x[0:k]
//!   3. Partial pivoting: find p = argmax |x[i]| for i >= k, swap rows
//!   4. Store: L[k+1:n, k] = x[k+1:n] / x[k], U[0:k+1, k] = x[0:k+1]
//! ```
//!
//! # Kernel Operations
//!
//! The factorization uses these primitive kernel operations:
//!
//! - **scatter_column**: Copy sparse column into dense work vector
//! - **sparse_axpy**: work[i] -= scale * values[i] for sparse indices
//! - **find_pivot**: Find maximum absolute value (SIMD reduction)
//! - **gather_and_clear**: Extract nonzeros back to sparse storage
//!
//! # Usage
//!
//! ```ignore
//! use numr::algorithm::sparse_linalg::lu::*;
//!
//! // Simple factorization (without solvr's symbolic analysis)
//! let factors = sparse_lu_simple_cpu(&matrix, &LuOptions::default())?;
//!
//! // Solve Ax = b
//! let x = sparse_lu_solve_cpu(&factors, &b)?;
//!
//! // With full symbolic analysis from solvr
//! let symbolic = solvr::symbolic_lu(&matrix)?;
//! let factors = sparse_lu_cpu(&matrix, &symbolic, &options)?;
//! ```
//!
//! # Backend Support
//!
//! - **CPU**: Gilbert-Peierls with SIMD-accelerated kernels (AVX2/FMA)
//! - **CUDA**: Left-looking with parallel scatter/gather
//! - **WebGPU**: Compute shader implementation

pub mod cpu;
pub mod traits;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// Re-export types
pub use types::{LuFactors, LuMetrics, LuOptions, LuSymbolic, LuSymbolicSimple, LuWorkspace};

// Re-export traits
pub use traits::{SparseLuKernels, SparseLuOps, validate_lu_solve_dims, validate_symbolic_pattern};

// Re-export CPU implementations
pub use cpu::{
    sparse_lu_cpu, sparse_lu_cpu_with_metrics, sparse_lu_cpu_with_workspace,
    sparse_lu_cpu_with_workspace_and_metrics, sparse_lu_simple_cpu, sparse_lu_solve_cpu,
};

// Re-export CPU kernels
pub use cpu::{
    divide_by_pivot, find_pivot, gather_and_clear, scatter_column, sparse_axpy, swap_rows,
};

// Re-export CUDA implementations
#[cfg(feature = "cuda")]
pub use cuda::{sparse_lu_cuda, sparse_lu_simple_cuda, sparse_lu_solve_cuda};

// Re-export WebGPU implementations
#[cfg(feature = "wgpu")]
pub use wgpu::{sparse_lu_simple_wgpu, sparse_lu_solve_wgpu, sparse_lu_wgpu};
