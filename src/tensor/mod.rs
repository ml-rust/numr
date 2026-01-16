//! Tensor types and operations
//!
//! This module provides the core `Tensor` type, which represents an n-dimensional
//! array stored on a compute device (CPU, GPU, etc.).

mod core;
mod id;
mod layout;
mod storage;

pub use core::Tensor;
pub use id::TensorId;
pub use layout::Layout;
pub use storage::Storage;
