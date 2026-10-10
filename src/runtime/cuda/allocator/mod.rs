//! CUDA stream-ordered allocator with a Rust-side free list.

mod core;
mod free_list;
mod ops;

pub use self::core::CudaAllocator;
pub(in crate::runtime::cuda) use self::core::alloc_error;
