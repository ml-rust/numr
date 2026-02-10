//! Linear algebra algorithm trait definitions
//!
//! These traits define the contract that all backends must implement
//! to ensure numerical parity across CPU, CUDA, WebGPU, and other backends.

pub mod linear_algebra;
pub mod matrix_functions;
pub mod tensor_decompose;

pub use linear_algebra::LinearAlgebraAlgorithms;
pub use matrix_functions::MatrixFunctionsAlgorithms;
pub use tensor_decompose::TensorDecomposeAlgorithms;
