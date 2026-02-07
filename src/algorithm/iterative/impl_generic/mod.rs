//! Generic implementations of iterative solvers
//!
//! These implementations use primitive operations (BinaryOps, UnaryOps, ReduceOps)
//! and can run on any backend that implements those traits.

mod adaptive_gmres;
pub(crate) mod amg;
pub(crate) mod amg_coarsen;
mod arnoldi_eig;
mod bicgstab;
mod cg;
mod cgs;
pub(crate) mod dense_eig;
mod gmres;
mod jacobi;
mod lanczos_eig;
mod lgmres;
mod minres;
mod qmr;
mod sor;
mod svds;

pub use adaptive_gmres::adaptive_gmres_impl;
pub use amg::{amg_preconditioned_cg, amg_setup, amg_vcycle};
pub use arnoldi_eig::arnoldi_eig_impl;
pub use bicgstab::bicgstab_impl;
pub use cg::cg_impl;
pub use cgs::cgs_impl;
pub use gmres::gmres_impl;
pub use jacobi::jacobi_impl;
pub use lanczos_eig::lanczos_eig_impl;
pub use lgmres::lgmres_impl;
pub use minres::minres_impl;
pub use qmr::qmr_impl;
pub use sor::sor_impl;
pub use svds::svds_impl;
