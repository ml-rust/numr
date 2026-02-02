//! WebGPU backend operation implementations
//!
//! Each operation type is implemented in its own module to maintain clean separation
//! and make the codebase easier to navigate.

pub mod type_conversion;
pub mod complex;
pub mod normalization;
pub mod matmul;
pub mod cumulative;
pub mod activation;
pub mod binary;
pub mod unary;
pub mod random;
pub mod linalg;
pub mod shape;
pub mod statistics;
pub mod sorting;
pub mod indexing;
pub mod reduce;
pub mod conditional;
pub mod utility;
