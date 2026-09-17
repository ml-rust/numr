//! Fused activation-multiplication with gradient support
//!
//! Each function computes `activation(a) * b` in a single memory pass.
//! Backward computes:
//! - d_a = grad_output * b * activation'(a)
//! - d_b = grad_output * activation(a)

mod backward;
mod derivative;
mod forward;
mod fused_kind;

pub use forward::{var_gelu_mul, var_relu_mul, var_sigmoid_mul, var_silu_mul};
