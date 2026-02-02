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
//! - [`GradStore`]: Storage for accumulated gradients (first-order)
//! - [`VarGradStore`]: Storage for gradient Vars (second-order)
//! - [`backward`]: Function to compute gradients via reverse-mode AD
//! - [`backward_with_graph`]: Backward with graph retention for Hessians
//!
//! # First-Order Example
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
//!
//! # Second-Order Example (Hessian-Vector Product)
//!
//! ```ignore
//! use numr::prelude::*;
//! use numr::autograd::{Var, backward, backward_with_graph, var_mul, var_sum};
//!
//! let device = CpuDevice::new();
//! let client = CpuRuntime::default_client(&device);
//!
//! // f(x) = x²
//! let x = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
//! let y = var_mul(&x, &x, &client)?;
//!
//! // First backward with graph retention
//! let grads = backward_with_graph(&y, &client)?;
//! let grad_x = grads.get_var(x.id()).unwrap();  // dy/dx = 2x = 6
//!
//! // Compute Hessian-vector product: H @ v where v = [1.0]
//! let v = Var::new(Tensor::from_slice(&[1.0f32], &[1], &device), false);
//! let grad_v = var_mul(grad_x, &v, &client)?;
//! let scalar = var_sum(&grad_v, &[], false, &client)?;
//!
//! // Second backward gives d²y/dx² * v = 2 * 1 = 2
//! let second_grads = backward(&scalar, &client)?;
//! ```

mod backward;
mod grad_fn;
mod grad_store;
mod var;
mod var_grad_store;
mod var_ops;

pub mod ops;

pub use backward::{backward, backward_with_graph};
pub use grad_fn::GradFn;
pub use grad_store::GradStore;
pub use var::Var;
pub use var_grad_store::VarGradStore;
pub use var_ops::*;
