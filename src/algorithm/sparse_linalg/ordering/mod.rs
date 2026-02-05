//! Sparse matrix ordering algorithms.
//!
//! Column and row ordering algorithms to minimize fill-in during sparse
//! matrix factorizations. These are CPU-only graph algorithms that operate
//! on sparsity structure, not numerical values.
//!
//! # Algorithms
//!
//! - **COLAMD**: Column Approximate Minimum Degree ordering for LU/QR
//! - **AMD**: Approximate Minimum Degree ordering for symmetric matrices (Cholesky)
//!
//! # Usage
//!
//! These orderings are computed once per sparsity pattern and reused for
//! multiple numeric factorizations. A good ordering can reduce fill-in by
//! orders of magnitude.

pub mod colamd;
pub mod traits;

pub use colamd::{ColamdOptions, ColamdStats, colamd};
pub use traits::SparseOrdering;
