//! Sparse matrix element-wise merge kernel launchers
//!
//! Two-pass algorithm for CSR element-wise operations:
//! 1. Count output size per row
//! 2. Exclusive scan to get row_ptrs
//! 3. Compute merged output

mod csc;
mod csr;
mod generic;
mod helpers;

pub use csc::*;
pub use csr::*;
pub use generic::*;
