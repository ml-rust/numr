//! COO Sparse Matrix Element-wise CUDA Operations
//!
//! Provides Rust wrappers for GPU-native COO sparse operations using CUB radix sort.
//!
//! # Algorithm
//!
//! COO element-wise operations use a sort-merge approach:
//! 1. Compute composite keys (row * ncols + col) for each entry
//! 2. Concatenate entries from both matrices
//! 3. Sort by key using CUB DeviceRadixSort
//! 4. Merge duplicates based on operation semantics
//!
//! For add/sub (union): Keep all unique positions, combine duplicates
//! For mul/div (intersection): Only keep positions where both matrices have values

mod kernels;
mod merge;

// Re-export public high-level merge operations
pub use merge::{coo_add_merge, coo_div_merge, coo_mul_merge, coo_sub_merge};

// Re-export kernel launch functions for format conversions
pub use kernels::{
    launch_coo_gather, launch_coo_gather_i64, launch_coo_init_indices,
    launch_thrust_sort_pairs_i64_i32,
};
