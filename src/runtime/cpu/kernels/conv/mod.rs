//! CPU convolution kernels.
//!
//! Direct convolution implementations without im2col transformation.
//!
//! # Accumulator rule
//!
//! Every tap loop sums in a [`FloatAcc`](super::wide_acc::FloatAcc), never in
//! the element type: `f64` for F64 input, `f32` for every other float. A half
//! or FP8 accumulator rounds on every add and stalls once its spacing exceeds
//! twice the increment (see [`super::wide_acc`]). The public `*_kernel`
//! functions pick the accumulator and call one private `*_kernel_acc` per
//! operation, so the tap loops exist once.

mod conv1d;
mod conv2d;
mod depthwise;

pub use conv1d::conv1d_kernel;
pub use conv2d::conv2d_kernel;
pub use depthwise::depthwise_conv2d_kernel;
