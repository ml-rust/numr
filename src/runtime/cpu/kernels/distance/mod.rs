//! CPU kernels for distance computation.

pub mod acc;
pub mod metrics;
pub mod pairwise;
pub mod squareform;

pub use pairwise::{cdist_block_kernel, pdist_row_kernel};
pub use squareform::{squareform_inverse_kernel, squareform_kernel};
