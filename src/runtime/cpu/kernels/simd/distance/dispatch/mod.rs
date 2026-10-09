//! Level dispatch for the distance reductions.

pub mod cosine;
pub mod dot;
pub mod manhattan;
pub mod sqeuclidean;
#[cfg(test)]
pub mod test_support;

pub use cosine::{cosine_sums_f32, cosine_sums_f64};
pub use manhattan::{manhattan_f32, manhattan_f64};
pub use sqeuclidean::{sqeuclidean_f32, sqeuclidean_f64};
