//! Sparse element-wise merge kernel launchers for CSR and CSC operands.
//!
//! Two-pass algorithm:
//! 1. Count output entries per row (CSR) or column (CSC).
//! 2. Exclusive scan of the counts into output pointers.
//! 3. Compute the merged output.

mod compressed;
mod helpers;

pub use compressed::*;
