//! AVX-512 distance reductions.
//!
//! Every kernel here enables `avx512f` only. The dispatcher calls them only
//! when `detect_simd()` reports `Avx512`. That level guarantees avx512f,
//! avx512vl, avx512dq, avx512bw and fma.
//!
//! The tail uses a masked load. Masked-off lanes read as zero and never fault,
//! so no load reads past `len`.

pub mod cosine;
pub mod dot;
pub mod manhattan;
pub mod sqeuclidean;

pub use cosine::{cosine_sums_f32, cosine_sums_f64};
pub use dot::{dot_f32, dot_f64};
pub use manhattan::{manhattan_f32, manhattan_f64};
pub use sqeuclidean::{sqeuclidean_f32, sqeuclidean_f64};
