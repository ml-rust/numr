//! Sparse QR Factorization
//!
//! Householder QR factorization for sparse matrices: A*P = Q*R
//!
//! # Algorithm
//!
//! Column-wise left-looking Householder QR:
//!
//! ```text
//! For each column k = 0 to min(m, n) - 1:
//!   1. Apply previous reflectors to column k
//!   2. Compute Householder reflector for column k below diagonal
//!   3. Store R[0:k+1, k] and Householder vector v_k, tau_k
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use numr::algorithm::sparse_linalg::qr::*;
//!
//! // Simple factorization
//! let factors = sparse_qr_simple_cpu(&matrix, &QrOptions::default())?;
//!
//! // Solve Ax = b
//! let x = sparse_qr_solve_cpu(&factors, &b)?;
//!
//! // Least-squares min ||Ax - b||
//! let x = sparse_qr_least_squares_cpu(&factors, &b)?;
//! ```

pub mod cpu;
pub mod symbolic;
pub mod traits;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// Re-export types
pub use types::{QrFactors, QrMetrics, QrOptions, QrOrdering, QrSymbolic};

// Re-export symbolic analysis
pub use symbolic::sparse_qr_symbolic;

// Re-export CPU implementations
pub use cpu::{
    sparse_qr_cpu, sparse_qr_cpu_with_metrics, sparse_qr_least_squares_cpu, sparse_qr_simple_cpu,
    sparse_qr_solve_cpu,
};

// Re-export CUDA implementations
#[cfg(feature = "cuda")]
pub use cuda::{sparse_qr_cuda, sparse_qr_simple_cuda, sparse_qr_solve_cuda};

// Re-export WebGPU implementations
#[cfg(feature = "wgpu")]
pub use wgpu::{sparse_qr_simple_wgpu, sparse_qr_solve_wgpu, sparse_qr_wgpu};
