//! Conv1d autograd operation
//!
//! Wraps `ConvOps::conv1d` with gradient tracking.
//!
//! Backward computes:
//! - d_input  = transposed convolution of grad_output with weight
//! - d_weight = cross-correlation of input with grad_output
//! - d_bias   = sum(grad_output) over batch and spatial dims

mod backward;
mod forward;
mod grad_math;

pub use forward::var_conv1d;
pub(super) use grad_math::conv1d_weight_backward;
