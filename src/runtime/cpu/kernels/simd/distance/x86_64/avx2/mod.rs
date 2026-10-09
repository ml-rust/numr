//! AVX2+FMA distance reductions.
//!
//! Every kernel here needs AVX2 and FMA. The dispatcher calls them for
//! `SimdLevel::Avx2Fma`. `SimdLevel::Avx512` runs the kernels in `avx512`.

pub mod cosine;
pub mod dot;
pub mod manhattan;
pub mod sqeuclidean;

pub use cosine::{cosine_sums_f32, cosine_sums_f64};
pub use dot::{dot_f32, dot_f64};
pub use manhattan::{manhattan_f32, manhattan_f64};
pub use sqeuclidean::{sqeuclidean_f32, sqeuclidean_f64};
