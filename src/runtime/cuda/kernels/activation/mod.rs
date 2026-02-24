//! Activation CUDA kernel launchers
//!
//! Split into submodules:
//! - `elementwise` - relu, sigmoid, silu, gelu, leaky_relu, elu
//! - `softmax` - softmax forward + backward (last-dim and non-last-dim)

mod elementwise;
mod softmax;

pub use elementwise::*;
pub use softmax::*;
