//! Integration tests for index operations (embedding_lookup, gather, scatter, index_select)
//!
//! Tests verify correctness across:
//! - Different dtypes (f32, f64, i32)
//! - Various embedding dimensions
//! - Boundary conditions
//! - Edge cases (single element, out of bounds handling)

#[path = "index_ops/advanced.rs"]
mod advanced;

#[path = "index_ops/embedding.rs"]
mod embedding;

#[path = "index_ops/gather_scatter.rs"]
mod gather_scatter;

#[path = "index_ops/masked.rs"]
mod masked;
