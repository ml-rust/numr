//! WebGPU runtime operation helpers and TensorOps implementation.
//!
//! Trait implementations (CompareOps, ScalarOps, LogicalOps, etc.) live in `ops/wgpu/`.
//! This module provides shared helpers, native GPU launchers, and TensorOps.

pub(crate) mod helpers;
pub(crate) mod native;
mod tensor;
