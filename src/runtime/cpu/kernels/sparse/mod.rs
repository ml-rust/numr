//! Sparse matrix kernels for CPU
//!
//! Low-level SIMD-accelerated kernels for sparse matrix operations.
//! These are primitive operations used by sparse algorithms (LU, Cholesky, etc.)

mod axpy;
mod gather;
mod pivot;
mod scatter;

pub use axpy::{sparse_axpy, sparse_axpy_i32};
pub use gather::{divide_by_pivot, gather_and_clear, gather_and_clear_i32, swap_rows};
pub use pivot::{find_pivot, find_pivot_range};
pub use scatter::{scatter_column, scatter_column_i32};
