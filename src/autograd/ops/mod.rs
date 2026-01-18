//! Backward implementations for operations
//!
//! Each operation has a corresponding backward struct that implements
//! `GradFn` to compute gradients during the backward pass.
//!
//! # Structure
//!
//! - `arithmetic`: Binary operations (add, sub, mul, div, pow)
//! - `unary`: Unary operations (neg, exp, log, sqrt, tan, etc.)
//! - `matmul`: Matrix multiplication
//! - `reduce`: Reductions (sum, mean, max, min)
//! - `activation`: Activation functions (relu, sigmoid, softmax)
//! - `scalar`: Tensor-scalar operations (add_scalar, mul_scalar, etc.)
//! - `linalg`: Linear algebra operations (trace, inverse, det, solve, cholesky)

mod activation;
mod arithmetic;
mod linalg;
mod matmul;
mod reduce;
mod scalar;
mod unary;

pub use activation::*;
pub use arithmetic::*;
pub use linalg::*;
pub use matmul::*;
pub use reduce::*;
pub use scalar::*;
pub use unary::*;
