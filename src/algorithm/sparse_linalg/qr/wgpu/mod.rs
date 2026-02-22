//! WebGPU implementation of sparse Householder QR factorization

#[cfg(feature = "wgpu")]
mod factorize;
mod qr;
mod solve;

pub use qr::{sparse_qr_simple_wgpu, sparse_qr_wgpu};
pub use solve::sparse_qr_solve_wgpu;
