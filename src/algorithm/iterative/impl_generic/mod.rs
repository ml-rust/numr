//! Generic implementations of iterative solvers
//!
//! These implementations use primitive operations (BinaryOps, UnaryOps, ReduceOps)
//! and can run on any backend that implements those traits.

mod adaptive_gmres;
mod arnoldi_eig;
mod bicgstab;
mod cg;
mod cgs;
pub(crate) mod dense_eig;
mod gmres;
mod lanczos_eig;
mod minres;

pub use adaptive_gmres::adaptive_gmres_impl;
pub use arnoldi_eig::arnoldi_eig_impl;
pub use bicgstab::bicgstab_impl;
pub use cg::cg_impl;
pub use cgs::cgs_impl;
pub use gmres::gmres_impl;
pub use lanczos_eig::lanczos_eig_impl;
pub use minres::minres_impl;
