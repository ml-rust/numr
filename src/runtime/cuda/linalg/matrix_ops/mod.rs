//! Matrix operations for CUDA (inverse, det, trace, diag, diagflat, rank, norm)

mod basic;
mod products;
mod rank_norm;
mod triangular;

pub use basic::{det_impl, diag_impl, diagflat_impl, inverse_impl, trace_impl};
pub use products::{khatri_rao_impl, kron_impl};
pub use rank_norm::{matrix_norm_impl, matrix_rank_impl};
pub use triangular::{slogdet_impl, tril_impl, triu_impl};
