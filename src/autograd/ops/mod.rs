//! Backward implementations for operations
//!
//! Each operation has a corresponding backward struct that implements
//! `GradFn` to compute gradients during the backward pass.
//!
//! # Structure
//!
//! - `arithmetic`: Binary operations (add, sub, mul, div, pow)
//! - `unary`: Unary operations (neg, exp, log, sqrt, etc.)
//! - `matmul`: Matrix multiplication
//! - `reduce`: Reductions (sum, mean)
//! - `activation`: Activation functions (relu, sigmoid, softmax)

mod activation;
mod arithmetic;
mod matmul;
mod reduce;
mod unary;

pub use activation::*;
pub use arithmetic::*;
pub use matmul::*;
pub use reduce::*;
pub use unary::*;
