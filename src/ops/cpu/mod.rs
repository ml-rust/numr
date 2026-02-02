//! CPU implementation of tensor operations.
//!
//! This module contains the operation trait implementations for the CPU runtime.
//! Each operation type has its own module.

pub mod type_conversion;
pub mod complex;
pub mod normalization;
pub mod matmul;
pub mod cumulative;
pub mod activation;
pub mod binary;
pub mod unary;
pub mod linalg;
pub mod statistics;
pub mod random;
pub mod distance;
pub mod reduce;
pub mod sorting;
pub mod conditional;
pub mod utility;
pub mod scalar;
pub mod compare;
pub mod logical;
pub mod indexing;
pub mod shape;
