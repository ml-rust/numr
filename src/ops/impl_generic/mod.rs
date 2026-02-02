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

pub mod multivariate;

pub use multivariate::{
    DTypeSupport, MultinomialSamplingOps, dirichlet_impl, multinomial_samples_impl,
    multivariate_normal_impl, wishart_impl,
};
