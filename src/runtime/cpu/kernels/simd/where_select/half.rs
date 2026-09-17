//! f16/bf16 wrappers for conditional select (where) operation (block-convert-compute via f32)

#[cfg(feature = "f16")]
use super::dispatch::where_f32;

half_where!(r#where, where_f32);
