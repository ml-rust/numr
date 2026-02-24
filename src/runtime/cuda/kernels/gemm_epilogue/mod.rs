//! CUDA GEMM epilogue kernels and launchers.

mod launcher;

pub use launcher::{
    launch_gemm_bias_act_batched_kernel, launch_gemm_bias_act_kernel,
    launch_gemm_bias_residual_batched_kernel, launch_gemm_bias_residual_kernel,
};
