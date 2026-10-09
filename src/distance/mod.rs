//! SIMD distance kernels on plain slices.
//!
//! These functions need no `Tensor`, no `Runtime` and no client. They take
//! slices and return a value, so a vector-search engine can call them per pair
//! or per query.
//!
//! - Per pair: [`dot_f32`], [`l2_squared_f32`], [`manhattan_f32`],
//!   [`cosine_distance_f32`], and their `_f64` twins.
//! - One query against many rows: [`dot_many_f32`], [`l2_squared_many_f32`],
//!   [`manhattan_many_f32`], [`cosine_distance_many_f32`], and their `_f64`
//!   twins. `rows` is row-major with `d` components per row, and `out` gets one
//!   value per row.
//! - Signed 8-bit: [`dot_i8`], [`dot_i8_scaled`], [`dot_i8_many`].
//!
//! Each free function detects the SIMD level of this CPU through a cached
//! read. [`Kernels`] holds one level, so a caller can detect once and reuse
//! it, or pin a lower level such as [`SimdLevel::Scalar`].
//!
//! # Numerics
//!
//! - The SIMD kernels sum in several lanes and fuse multiply-adds. The
//!   summation order differs from a sequential scalar loop and between SIMD
//!   levels, so results can differ in the last bits.
//! - NaN and infinity in the inputs propagate to the result.
//! - Cosine distance is `1 - dot / sqrt(|a|^2 * |b|^2)`. A zero vector gives 0.
//! - For unit-length vectors, cosine distance is `1 - dot`. Use
//!   [`dot_many_f32`] and subtract from 1 to skip the two norms.
//! - The i8 dot product is exact. A total outside `i32` range saturates to
//!   `i32::MIN` or `i32::MAX`.
//!
//! # Panics
//!
//! Every function panics when the two lengths differ, or when a `*_many_*`
//! shape does not match. An empty pair (`d == 0`) is valid and gives 0.
//!
//! # Example
//!
//! ```
//! use numr::distance;
//!
//! let a = [1.0f32, 2.0, 3.0];
//! let b = [4.0f32, 5.0, 6.0];
//! assert_eq!(distance::dot_f32(&a, &b), 32.0);
//! assert_eq!(distance::l2_squared_f32(&a, &b), 27.0);
//! assert!(distance::cosine_distance_f32(&a, &a).abs() < 1e-6);
//!
//! // One query against two rows of 3 components.
//! let query = [1.0f32, 0.0, 0.0];
//! let rows = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let mut out = [0.0f32; 2];
//! distance::dot_many_f32(&query, &rows, 3, &mut out);
//! assert_eq!(out, [1.0, 0.0]);
//!
//! // Detect once, reuse for many calls.
//! let k = distance::Kernels::detect();
//! assert_eq!(k.manhattan_f32(&a, &b), 9.0);
//! ```

mod cosine;
mod dot;
mod int8;
mod kernels;
mod l2_squared;
mod manhattan;
mod shape;

pub use crate::runtime::cpu::kernels::simd::SimdLevel;
pub use cosine::{
    cosine_distance_f32, cosine_distance_f64, cosine_distance_many_f32, cosine_distance_many_f64,
};
pub use dot::{dot_f32, dot_f64, dot_many_f32, dot_many_f64};
pub use int8::{dot_i8, dot_i8_many, dot_i8_scaled};
pub use kernels::Kernels;
pub use l2_squared::{l2_squared_f32, l2_squared_f64, l2_squared_many_f32, l2_squared_many_f64};
pub use manhattan::{manhattan_f32, manhattan_f64, manhattan_many_f32, manhattan_many_f64};
