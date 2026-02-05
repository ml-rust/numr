//! Native GPU operation implementations for WebGPU.
//!
//! This module contains native_* helper functions that launch WGSL compute shaders.
//! Each submodule focuses on a specific category of operations.

mod helpers;

mod activation;
mod binary;
mod cast;
mod compare;
mod conditional;
mod cumulative;
mod indexing;
pub(crate) mod logical;
mod masking;
mod matmul;
mod normalization;
mod reduce;
mod unary;

// Re-export all native functions for use by ops/wgpu/ implementations
pub(crate) use activation::native_parametric_activation;
pub(crate) use binary::{native_binary_op, native_scalar_op};
pub(crate) use cast::native_cast_op;
pub(crate) use compare::native_compare_op;
pub(crate) use conditional::{native_clamp, native_where_cond};
pub(crate) use cumulative::{native_cumprod, native_cumsum, native_logsumexp};
pub(crate) use indexing::{native_gather, native_index_put, native_index_select, native_scatter};
pub(crate) use logical::{
    native_logical_and, native_logical_not, native_logical_or, native_logical_xor,
};
pub(crate) use masking::{native_embedding_lookup, native_masked_fill, native_masked_select};
pub(crate) use matmul::{native_matmul, native_matmul_bias};
pub(crate) use normalization::{native_layer_norm, native_rms_norm};
pub(crate) use reduce::{native_argreduce_op, native_reduce_op, native_softmax};
pub(crate) use unary::native_unary_op;
