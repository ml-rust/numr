//! CPU implementation of sparse QR factorization
//!
//! Householder QR with COLAMD column ordering.

pub(crate) mod algorithm;
pub(crate) mod helpers;
mod qr;

pub use qr::{
    sparse_qr_cpu, sparse_qr_cpu_with_metrics, sparse_qr_least_squares_cpu, sparse_qr_simple_cpu,
    sparse_qr_solve_cpu,
};
