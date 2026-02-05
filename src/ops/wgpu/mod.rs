//! WebGPU backend operation implementations
//!
//! Each operation type is implemented in its own module to maintain clean separation
//! and make the codebase easier to navigate.

pub mod activation;
pub mod advanced_random;
pub mod binary;
pub mod complex;
pub mod conditional;
pub mod conv;
pub mod cumulative;
pub mod distance;
pub mod indexing;
pub mod linalg;
pub mod logical;
pub mod matmul;
pub mod multivariate;
pub mod normalization;
pub mod quasirandom;
pub mod random;
pub mod reduce;
pub mod shape;
pub mod sorting;
pub mod statistics;
pub mod type_conversion;
pub mod unary;
pub mod utility;
