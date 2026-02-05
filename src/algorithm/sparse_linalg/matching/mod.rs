//! Bipartite matching algorithms for sparse matrices.
//!
//! Maximum matching algorithms to find row/column permutations that maximize
//! nonzeros on the diagonal. These are CPU-only graph algorithms that operate
//! on sparsity structure.
//!
//! # Algorithms
//!
//! - **Hopcroft-Karp**: O(E√V) maximum cardinality matching
//!
//! # Use Cases
//!
//! - **Maximum transversal**: Find row permutation to maximize diagonal nonzeros
//! - **Structural rank**: Determine if matrix has full structural rank
//! - **BTF preprocessing**: Prepare matrix for block triangular form decomposition

pub mod hopcroft_karp;
pub mod traits;

pub use hopcroft_karp::{MatchingResult, hopcroft_karp, maximum_transversal};
pub use traits::BipartiteMatching;
