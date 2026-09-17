//! Conv2d autograd operation
//!
//! Wraps `ConvOps::conv2d` with gradient tracking.
//!
//! Backward computes:
//! - d_input  = transposed convolution of grad_output with weight
//! - d_weight = cross-correlation of input with grad_output
//! - d_bias   = sum(grad_output) over batch and spatial dims

mod backward;
mod forward;
mod grad_math;

pub use forward::var_conv2d;
