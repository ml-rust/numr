//! WebGPU implementation of sparse LU factorization
//!
//! Uses WGSL compute shaders for scatter, sparse AXPY, pivot search, and gather.

mod lu;

pub use lu::{sparse_lu_simple_wgpu, sparse_lu_solve_wgpu, sparse_lu_wgpu};
