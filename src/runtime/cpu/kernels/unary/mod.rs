//! Unary and activation operation kernels
//!
//! Provides element-wise unary operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

pub mod activations;
mod clamp;
mod complex;
pub mod fused_activations;
mod int;
mod predicates;
mod relu;
pub mod scalar;
pub mod snake;
mod unary_dispatch;

pub use activations::{elu_kernel, gelu_kernel, leaky_relu_kernel, sigmoid_kernel, silu_kernel};
pub use clamp::clamp_kernel;
pub use fused_activations::{
    gelu_mul_kernel, relu_mul_kernel, sigmoid_mul_kernel, silu_mul_kernel,
};
pub use predicates::{isinf_kernel, isnan_kernel};
pub use relu::relu_kernel;
pub use scalar::{relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64};
pub use snake::{snake_beta_bwd_kernel, snake_beta_kernel};
pub use unary_dispatch::unary_op_kernel;
