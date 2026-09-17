//! CUDA implementation of sparse linear algebra algorithms.
//!
//! This module provides GPU-accelerated sparse linear algebra using level scheduling
//! for parallel execution of inherently sequential algorithms.
//!
//! # Level Scheduling
//!
//! ILU(0), IC(0), and sparse triangular solve have row-to-row dependencies.
//! Level scheduling analyzes the sparsity pattern to find independent rows:
//! - Rows at the same level can execute in parallel
//! - Levels execute sequentially
//!
//! This enables GPU parallelism while maintaining correctness.

mod common;
mod dispatch;
mod ic0;
mod ilu0;
mod iluk;
mod triangular_solve;

use super::{CudaClient, CudaRuntime};
