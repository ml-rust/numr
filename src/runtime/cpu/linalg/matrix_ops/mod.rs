//! Basic matrix operations (inverse, det, trace, diag, rank, norms)

mod det;
mod diag;
mod inverse;
mod khatri_rao;
mod kron;
mod norm;
mod rank;
mod slogdet;
mod trace;
mod triangular;

pub use det::det_impl;
pub use diag::{diag_impl, diagflat_impl};
pub use inverse::inverse_impl;
pub use khatri_rao::khatri_rao_impl;
pub use kron::kron_impl;
pub use norm::matrix_norm_impl;
pub use rank::matrix_rank_impl;
pub use slogdet::slogdet_impl;
pub use trace::trace_impl;
pub use triangular::{tril_impl, triu_impl};
