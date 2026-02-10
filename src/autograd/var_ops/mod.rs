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
//! let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
//! let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
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
mod cumulative;
mod indexing;
pub mod linalg;
mod matmul;
pub mod reduce;
mod scalar;
mod stats;
mod unary;
mod utility;

// Re-export all public functions
pub use activation::{var_relu, var_sigmoid, var_softmax};
pub use arithmetic::{var_add, var_div, var_mul, var_pow, var_sub};
pub use cumulative::{var_cumprod, var_cumsum};
pub use indexing::var_gather;
pub use linalg::{var_cholesky, var_det, var_inverse, var_solve, var_trace};
pub use matmul::var_matmul;
pub use reduce::{var_max, var_mean, var_min, var_sum};
pub use scalar::{var_add_scalar, var_div_scalar, var_mul_scalar, var_pow_scalar, var_sub_scalar};
pub use stats::{var_std, var_var};
pub use unary::{
    var_abs, var_cos, var_exp, var_log, var_neg, var_recip, var_sin, var_sqrt, var_square, var_tan,
    var_tanh,
};
pub use utility::var_clamp;
