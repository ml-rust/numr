//! CPU implementation of sparse LU factorization
//!
//! Gilbert-Peierls left-looking algorithm with SIMD-accelerated kernels.

mod lu;

// Re-export kernel functions from runtime for convenience
pub use crate::runtime::cpu::kernels::sparse::{
    divide_by_pivot, find_pivot, find_pivot_range, gather_and_clear, gather_and_clear_i32,
    scatter_column, scatter_column_i32, sparse_axpy, sparse_axpy_i32, swap_rows,
};

pub use lu::{
    sparse_lu_cpu, sparse_lu_cpu_with_metrics, sparse_lu_simple_cpu, sparse_lu_solve_cpu,
};
