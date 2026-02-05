//! Generic implementations of iterative solvers
//!
//! These implementations use tensor primitives and work on all backends.
//! Backend-specific implementations delegate to these generic functions.

pub mod bicgstab;
pub mod gmres;

pub use bicgstab::bicgstab_impl;
pub use gmres::gmres_impl;
