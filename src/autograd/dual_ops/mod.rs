//! Dual tensor operations for forward-mode automatic differentiation
//!
//! Each operation implements the appropriate tangent propagation rule.

mod activation;
mod arithmetic;
mod matmul;
mod reduce;
mod scalar;
mod shape;
mod unary;

pub use activation::*;
pub use arithmetic::*;
pub use matmul::*;
pub use reduce::*;
pub use scalar::*;
pub use shape::*;
pub use unary::*;
