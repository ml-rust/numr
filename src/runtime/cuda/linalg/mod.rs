//! CUDA implementation of linear algebra algorithms
//!
//! This module implements the [`LinearAlgebraAlgorithms`] trait for CUDA.
//! All algorithms follow the exact specification in the trait documentation
//! to ensure backend parity with CPU/WebGPU implementations.
//!
//! Native CUDA kernels are used - NO cuSOLVER dependency.

mod advanced_decompositions;
mod banded;
mod decompositions;
mod eig_general;
mod eig_symmetric;
pub(crate) mod helpers;
mod matrix_functions;
mod matrix_ops;
mod schur;
mod solvers;
mod statistics;
mod svd;
mod tensor_decompose;

mod linear_algebra_algorithms;
mod matrix_functions_algorithms;

#[cfg(test)]
mod test_support;
