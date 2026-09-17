//! WebGPU implementation of linear algebra algorithms.
//!
//! This module implements the [`LinearAlgebraAlgorithms`] and [`MatrixFunctionsAlgorithms`]
//! traits for WebGPU. All algorithms use native WGSL compute shaders - NO CPU FALLBACK.
//!
//! # Supported Data Types
//!
//! **F32 only.** WGSL (WebGPU Shading Language) does not natively support 64-bit
//! floating point operations. All functions in this module will return
//! [`Error::UnsupportedDType`] for non-F32 inputs.
//!
//! For F64 linear algebra, use the CPU or CUDA backends instead.
//!
//! # Performance Note
//!
//! Operations use native WGSL compute shaders running entirely on the GPU.

mod advanced_decompositions;
mod banded;
mod helpers;

mod decompositions;
mod eig_general;
mod eig_symmetric;
mod lstsq;
mod matrix_functions;
mod matrix_ops;
mod schur;
mod solvers;
mod statistics;
mod svd;
mod tensor_decompose;
mod triangular_solve;

mod linear_algebra_algorithms;
mod matrix_functions_algorithms;

#[cfg(test)]
mod test_support;
