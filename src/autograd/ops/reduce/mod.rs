//! Backward implementations for reduction operations

mod common;
mod extremum;
mod statistical;
mod sum_mean;

pub use extremum::*;
pub use statistical::*;
pub use sum_mean::*;
