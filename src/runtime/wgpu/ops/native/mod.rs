//! Native GPU operation implementations for WebGPU.
//!
//! This module contains native_* helper functions that launch WGSL compute shaders.
//! Each submodule focuses on a specific category of operations.

mod activation;
mod binary;
mod cast;
mod compare;
mod conditional;
mod cumulative;
mod indexing;
mod masking;
mod matmul;
mod normalization;
mod reduce;
mod unary;

// Re-export all native functions for use by ops/wgpu/ implementations
pub(super) use activation::native_parametric_activation;
pub(super) use binary::{native_binary_op, native_scalar_op};
pub(super) use cast::native_cast_op;
pub(super) use compare::native_compare_op;
pub(super) use conditional::{native_clamp, native_where_cond};
pub(super) use cumulative::{native_cumprod, native_cumsum, native_logsumexp};
pub(super) use indexing::{native_gather, native_index_put, native_index_select, native_scatter};
pub(super) use masking::{native_embedding_lookup, native_masked_fill, native_masked_select};
pub(super) use matmul::{native_matmul, native_matmul_bias};
pub(super) use normalization::{native_layer_norm, native_rms_norm};
pub(super) use reduce::{native_argreduce_op, native_reduce_op, native_softmax};
pub(super) use unary::native_unary_op;
