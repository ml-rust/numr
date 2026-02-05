//! Generic implementations of iterative solvers
//!
//! These implementations use primitive operations (BinaryOps, UnaryOps, ReduceOps)
//! and can run on any backend that implements those traits.

mod adaptive_gmres;
mod bicgstab;
mod gmres;

pub use adaptive_gmres::adaptive_gmres_impl;
pub use bicgstab::bicgstab_impl;
pub use gmres::gmres_impl;
