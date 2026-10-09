//! NEON distance reductions.
//!
//! Every kernel here enables `neon` only. AArch64 always has NEON. The
//! dispatcher calls them for `SimdLevel::Neon` and `SimdLevel::NeonFp16`.
//!
//! A scalar loop takes the last `len % lanes` elements, so no load reads past
//! `len`.

pub mod cosine;
pub mod dot;
pub mod manhattan;
pub mod sqeuclidean;

pub use cosine::{cosine_sums_f32, cosine_sums_f64};
pub use dot::{dot_f32, dot_f64};
pub use manhattan::{manhattan_f32, manhattan_f64};
pub use sqeuclidean::{sqeuclidean_f32, sqeuclidean_f64};
