//! Tensor types and operations
//!
//! This module provides the core `Tensor` type, which represents an n-dimensional
//! array stored on a compute device (CPU, GPU, etc.).

mod core;
pub(crate) mod id;
mod layout;
pub(crate) mod shape;
mod storage;
mod strides;

pub use core::Tensor;
pub(crate) use id::TensorId;
pub use layout::{Layout, Shape, Strides};
pub use storage::Storage;
