//! Operations on Var that build the computation graph
//!
//! These functions perform forward computation and create the appropriate
//! backward functions for gradient tracking.
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, var_mul, backward};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! // Create leaf variables
//! let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device)?, true);
//! let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device)?, true);
//!
//! // Build computation graph: z = x * y
//! let z = var_mul(&x, &y, &client)?;
//!
//! // Compute gradients
//! let grads = backward(&z, &client)?;
//! # Ok::<(), numr::error::Error>(())
//! ```

mod macros;
pub mod ops;

mod activation;
mod arithmetic;
mod binary_extremum;
mod cast;
mod conv1d;
mod conv2d;
mod conv_common;
mod conv_transpose1d;
mod cumulative;
mod dropout;
mod einsum;
mod elementwise_derivative;
mod elu;
mod erf;
mod flip;
mod fused_activation_mul;
mod gelu;
mod gemm_epilogue;
mod index_select;
mod indexing;
mod l2_normalize;
mod leaky_relu;
mod logsumexp;

pub mod linalg;
mod matmul;
mod matmul_wide;
mod maximum;
mod minimum;
mod normalization;
mod pad;
mod prod;
pub mod reduce;
mod rsqrt;
mod scalar;
mod scatter;
mod scatter_add;
mod snake;
mod stack_split;
mod stats;
mod swiglu;
mod unary;
mod utility;
mod where_cond;

// Re-export all public functions
pub use activation::{var_log_softmax, var_relu, var_sigmoid, var_silu, var_softmax, var_softplus};
pub use arithmetic::{var_add, var_div, var_mul, var_pow, var_sub};
pub use cast::var_cast;
pub use conv_transpose1d::var_conv_transpose1d;
pub use conv1d::var_conv1d;
pub use conv2d::var_conv2d;
pub use cumulative::{var_cumprod, var_cumsum};
pub use dropout::var_dropout;
pub use einsum::var_einsum;
pub use elu::var_elu;
pub use erf::var_erf;
pub use flip::var_flip;
pub use fused_activation_mul::{
    var_gelu_erf_mul, var_gelu_mul, var_relu_mul, var_sigmoid_mul, var_silu_mul,
};
pub use gelu::{var_gelu, var_gelu_erf};
pub use gemm_epilogue::var_matmul_bias_activation;
pub use index_select::var_index_select;
pub use indexing::{var_embedding_lookup, var_gather};
pub use l2_normalize::var_l2_normalize;
pub use leaky_relu::var_leaky_relu;
pub use linalg::{var_cholesky, var_det, var_inverse, var_solve, var_trace};
pub use logsumexp::var_logsumexp;
pub use matmul::var_matmul;
pub use matmul_wide::var_matmul_wide;
pub use maximum::var_maximum;
pub use minimum::var_minimum;
pub use normalization::{
    var_fused_add_layer_norm, var_fused_add_rms_norm, var_group_norm, var_layer_norm, var_rms_norm,
};
pub use pad::var_pad;
pub use prod::var_prod;
pub use reduce::{var_max, var_mean, var_min, var_sum};
pub use rsqrt::var_rsqrt;
pub use scalar::{var_add_scalar, var_div_scalar, var_mul_scalar, var_pow_scalar, var_sub_scalar};
pub use scatter::var_scatter;
pub use scatter_add::var_scatter_add;
pub use snake::var_snake_beta;
pub use stack_split::{var_chunk, var_split, var_stack};
pub use stats::{var_std, var_var};
pub use swiglu::var_swiglu;
pub use unary::{
    var_abs, var_cos, var_exp, var_log, var_neg, var_recip, var_sin, var_sqrt, var_square, var_tan,
    var_tanh,
};
pub use utility::var_clamp;
pub use where_cond::var_where;
