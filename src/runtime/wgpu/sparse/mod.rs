//! Sparse tensor operations for WebGPU runtime.
//!
//! This module provides GPU-accelerated sparse linear algebra operations
//! using level scheduling for parallel execution.

use super::{WgpuClient, WgpuRuntime};

mod common;
mod conversions;
mod dsmm;
mod esc_spgemm;
mod high_level_ops;
mod ic0;
mod ilu0;
mod iluk;
mod iterative;
mod merge;
mod sparse_algorithms;
mod sparse_linalg_algorithms;
mod spmv;
mod triangular_solve;
