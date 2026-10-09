//! Level dispatch for the distance reductions.

pub mod cosine;
pub mod dot;
pub mod manhattan;
pub mod sqeuclidean;
#[cfg(test)]
pub mod test_avx512;
#[cfg(test)]
pub mod test_support;

pub use cosine::{cosine_sums_f32, cosine_sums_f32_with, cosine_sums_f64, cosine_sums_f64_with};
pub use dot::{dot_f32_with, dot_f64_with};
pub use manhattan::{manhattan_f32, manhattan_f32_with, manhattan_f64, manhattan_f64_with};
pub use sqeuclidean::{
    sqeuclidean_f32, sqeuclidean_f32_with, sqeuclidean_f64, sqeuclidean_f64_with,
};
