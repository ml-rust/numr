//! Tensor types and operations
//!
//! This module provides the core `Tensor` type, which represents an n-dimensional
//! array stored on a compute device (CPU, GPU, etc.).

mod layout;
mod storage;
mod id;
mod core;

pub use layout::Layout;
pub use storage::Storage;
pub use id::TensorId;
pub use core::Tensor;
