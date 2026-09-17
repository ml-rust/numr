//! CPU implementation of linear algebra algorithms
//!
//! This module implements the [`crate::algorithm::linalg::LinearAlgebraAlgorithms`] trait for CPU.
//! All algorithms follow the exact specification in the trait documentation
//! to ensure backend parity with CUDA/WebGPU implementations.

mod advanced_decompositions;
mod banded;
mod decompositions;
mod eig_general;
mod eig_symmetric;
mod linear_algebra_algorithms;
mod matrix_functions;
mod matrix_functions_algorithms;
mod matrix_ops;
mod schur;
mod solvers;
mod statistics;
mod svd;
mod tensor_decompose;

#[cfg(test)]
mod test_support;
