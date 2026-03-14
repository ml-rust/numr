//! GEMM epilogue CPU kernels
//!
//! Fused matmul + bias + activation/residual kernels.

pub mod backward;
pub mod forward;

pub use backward::matmul_bias_activation_bwd_kernel;
pub use forward::{matmul_bias_activation_kernel, matmul_bias_residual_kernel};
