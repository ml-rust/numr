//! Generic implementations of composite operations.
//!
//! This module contains backend-agnostic algorithm implementations that are shared
//! across all backends (CPU, CUDA, WebGPU). Each backend delegates to these
//! implementations to ensure numerical parity and reduce code duplication.
//!
//! # Architecture
//!
//! Composite operations (operations built from primitives) are defined here as generic
//! functions with trait bounds. The backends provide thin wrappers that call these
//! implementations.
//!
//! ```text
//! impl_generic/multivariate.rs
//!     └── multivariate_normal_impl<R, C>()
//!             │
//!             ├── cpu/multivariate.rs delegates here
//!             ├── cuda/multivariate.rs delegates here
//!             └── wgpu/multivariate.rs delegates here
//! ```

pub mod einsum;
pub mod linalg;
pub mod multivariate;
pub mod random;
pub mod shape;
pub mod utility;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub use linalg::{slogdet_impl, tril_impl, triu_impl};
pub use multivariate::{
    DTypeSupport, MultinomialSamplingOps, dirichlet_impl, multinomial_samples_impl,
    multivariate_normal_impl, wishart_impl,
};
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub use random::randperm_impl;
pub use shape::{repeat_interleave_impl, unfold_impl};
pub use utility::meshgrid_impl;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub use utility::one_hot_impl;
