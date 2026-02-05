//! CUDA implementation of sparse LU factorization
//!
//! Uses CUDA kernels for scatter, sparse AXPY, pivot search, and gather operations.
//! The column-by-column structure of Gilbert-Peierls limits GPU parallelism,
//! but within-column operations are parallelized.

mod lu;

pub use lu::{sparse_lu_cuda, sparse_lu_simple_cuda, sparse_lu_solve_cuda};
