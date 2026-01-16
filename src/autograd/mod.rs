//! Automatic differentiation (autograd)
//!
//! This module provides reverse-mode automatic differentiation for
//! computing gradients of tensor computations.
//!
//! # Overview
//!
//! The autograd system consists of:
//! - [`Var`]: A tensor that tracks gradients
//! - [`GradFn`]: Trait for computing gradients in backward pass
//! - [`GradStore`]: Storage for accumulated gradients
//! - [`backward`]: Function to compute gradients via reverse-mode AD
//!
//! # Example
//!
//! ```ignore
//! use numr::prelude::*;
//! use numr::autograd::{Var, backward};
//!
//! let device = CpuDevice::new();
//! let client = CpuRuntime::default_client(&device);
//!
//! // Create leaf variables
//! let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
//! let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
//!
//! // Forward: z = x * y
//! let z = var_mul(&x, &y, &client)?;
//!
//! // Backward
//! let grads = backward(&z, &client)?;
//!
//! // dx = y = 3.0, dy = x = 2.0
//! let grad_x = grads.get(x.id()).unwrap();
//! let grad_y = grads.get(y.id()).unwrap();
//! ```

mod backward;
mod grad_fn;
mod grad_store;
mod var;
mod var_ops;

pub mod ops;

pub use backward::backward;
pub use grad_fn::GradFn;
pub use grad_store::GradStore;
pub use var::Var;
pub use var_ops::*;
