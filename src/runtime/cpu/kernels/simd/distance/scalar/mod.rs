//! Scalar distance reductions.
//!
//! Each kernel keeps one sequential accumulator in index order. That is the
//! order of the generic loops in `kernels::distance::metrics`, so
//! `SimdLevel::Scalar` reproduces them bit for bit.

pub mod cosine;
pub mod dot;
pub mod manhattan;
pub mod sqeuclidean;

pub use cosine::{cosine_sums_f32, cosine_sums_f64};
pub use dot::{dot_f32, dot_f64};
pub use manhattan::{manhattan_f32, manhattan_f64};
pub use sqeuclidean::{sqeuclidean_f32, sqeuclidean_f64};
