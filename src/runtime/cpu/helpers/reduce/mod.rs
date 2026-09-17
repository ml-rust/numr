//! Reduction operation helpers for CPU tensors

mod common;
mod dispatch;
mod multi_dim;
mod precision;
mod single_dim;

pub use dispatch::reduce_impl;
pub use precision::reduce_impl_with_precision;
