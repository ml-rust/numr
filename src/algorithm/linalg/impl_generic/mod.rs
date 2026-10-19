//! Generic implementations of composite linear algebra algorithms.
//!
//! These compose primitive ops and run on any backend that implements them.

mod lstsq;
mod svd_cutoff;

pub use lstsq::lstsq_impl;
pub use svd_cutoff::{resolve_rcond, svd_cutoff};
