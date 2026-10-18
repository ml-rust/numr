// Backend parity for the autograd ops: the forward value and every input
// gradient of `backward(sum(op(x) * w))` on CUDA and WebGPU must match CPU.

pub mod harness;

pub mod einsum;
pub mod elementwise;
pub mod indexing;
pub mod mode;
pub mod reduce;
pub mod shape;
