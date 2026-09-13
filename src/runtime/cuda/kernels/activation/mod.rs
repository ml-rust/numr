//! Activation CUDA kernel launchers
//!
//! Split into submodules:
//! - `elementwise` - relu, sigmoid, silu, gelu, leaky_relu, elu
//! - `softmax` - softmax forward + backward (last-dim and non-last-dim)
//! - `snake` - snake_beta forward, `d_x`, and per-channel parameter gradients

mod elementwise;
mod snake;
mod softmax;

pub use elementwise::*;
pub use snake::*;
pub use softmax::*;
