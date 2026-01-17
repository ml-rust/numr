//! Operations on Var that build the computation graph
//!
//! These functions perform forward computation and create the appropriate
//! backward functions for gradient tracking.
//!
//! # Example
//!
//! ```ignore
//! use numr::prelude::*;
//! use numr::autograd::{Var, var_ops::*, backward};
//!
//! let device = CpuDevice::new();
//! let client = CpuRuntime::default_client(&device);
//!
//! // Create leaf variables
//! let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
//! let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
//!
//! // Build computation graph: z = x * y
//! let z = var_mul(&x, &y, &client)?;
//!
//! // Compute gradients
//! let grads = backward(&z, &client)?;
//! ```

use super::Var;
use super::ops::*;
use crate::error::Result;
use crate::ops::{CompareOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

// ============================================================================
// Macro-based code generation for var_* functions
// ============================================================================
//
// These macros eliminate boilerplate by generating var_* functions with
// consistent patterns. Each macro handles a specific backward constructor
// signature pattern.

/// Binary operation with shape-based backward (add, sub)
macro_rules! impl_var_binary_op_shapes {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor(), b.tensor())?;

            if a.requires_grad() || b.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(
                    a.id(),
                    b.id(),
                    a.shape(),
                    b.shape(),
                    a.grad_fn().cloned(),
                    b.grad_fn().cloned(),
                );
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Binary operation with tensor-based backward (mul, div, matmul)
macro_rules! impl_var_binary_op_tensors {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor(), b.tensor())?;

            if a.requires_grad() || b.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(
                    a.id(),
                    b.id(),
                    a.tensor().clone(),
                    b.tensor().clone(),
                    a.grad_fn().cloned(),
                    b.grad_fn().cloned(),
                );
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with id-only backward (neg)
macro_rules! impl_var_unary_op_id {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with output-based backward (exp, sqrt, tanh, recip, sigmoid)
macro_rules! impl_var_unary_op_output {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), output.clone());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with input tensor backward (log, sin, cos, square, abs, tan, relu)
macro_rules! impl_var_unary_op_input {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), a.tensor().clone());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Scalar operation with id-only backward (add_scalar, sub_scalar)
macro_rules! impl_var_scalar_op_id {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + ScalarOps<R>,
            R::Client: ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor(), scalar)?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Scalar operation with scalar value backward (mul_scalar, div_scalar)
macro_rules! impl_var_scalar_op_scalar {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + ScalarOps<R>,
            R::Client: ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor(), scalar)?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), scalar);
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

// ============================================================================
// Binary Operations
// ============================================================================

impl_var_binary_op_shapes!(
    /// Add two variables: z = a + b
    ///
    /// Creates AddBackward for gradient computation.
    var_add, add, AddBackward
);

impl_var_binary_op_shapes!(
    /// Subtract two variables: z = a - b
    ///
    /// Creates SubBackward for gradient computation.
    var_sub, sub, SubBackward
);

impl_var_binary_op_tensors!(
    /// Multiply two variables: z = a * b
    ///
    /// Creates MulBackward for gradient computation.
    var_mul, mul, MulBackward
);

impl_var_binary_op_tensors!(
    /// Divide two variables: z = a / b
    ///
    /// Creates DivBackward for gradient computation.
    var_div, div, DivBackward
);

/// Power: z = a^b
///
/// Creates PowBackward for gradient computation.
/// Note: Has special backward constructor requiring output tensor.
pub fn var_pow<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.pow(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = PowBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            output.clone(),
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

impl_var_binary_op_tensors!(
    /// Matrix multiplication: z = a @ b
    ///
    /// Creates MatmulBackward for gradient computation.
    var_matmul, matmul, MatmulBackward
);

// ============================================================================
// Unary Operations
// ============================================================================

impl_var_unary_op_id!(
    /// Negation: z = -a
    var_neg, neg, NegBackward
);

impl_var_unary_op_output!(
    /// Exponential: z = exp(a)
    var_exp, exp, ExpBackward
);

impl_var_unary_op_input!(
    /// Natural logarithm: z = log(a)
    var_log, log, LogBackward
);

/// Square root: z = sqrt(a)
///
/// Note: Requires additional ScalarOps bound.
pub fn var_sqrt<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.sqrt(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SqrtBackward::<R>::new(a.id(), output.clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

impl_var_unary_op_input!(
    /// Sine: z = sin(a)
    var_sin, sin, SinBackward
);

impl_var_unary_op_input!(
    /// Cosine: z = cos(a)
    var_cos, cos, CosBackward
);

impl_var_unary_op_output!(
    /// Hyperbolic tangent: z = tanh(a)
    var_tanh, tanh, TanhBackward
);

/// Square: z = a²
///
/// Note: Requires additional ScalarOps bound.
pub fn var_square<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.square(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SquareBackward::<R>::new(a.id(), a.tensor().clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

impl_var_unary_op_output!(
    /// Reciprocal: z = 1/a
    var_recip, recip, RecipBackward
);

impl_var_unary_op_input!(
    /// Absolute value: z = |a|
    var_abs, abs, AbsBackward
);

impl_var_unary_op_input!(
    /// Tangent: z = tan(a)
    var_tan, tan, TanBackward
);

// ============================================================================
// Scalar Operations
// ============================================================================

impl_var_scalar_op_id!(
    /// Add scalar: z = a + scalar
    var_add_scalar, add_scalar, AddScalarBackward
);

impl_var_scalar_op_id!(
    /// Subtract scalar: z = a - scalar
    var_sub_scalar, sub_scalar, SubScalarBackward
);

impl_var_scalar_op_scalar!(
    /// Multiply by scalar: z = a * scalar
    var_mul_scalar, mul_scalar, MulScalarBackward
);

impl_var_scalar_op_scalar!(
    /// Divide by scalar: z = a / scalar
    var_div_scalar, div_scalar, DivScalarBackward
);

/// Power by scalar: z = a^scalar
///
/// Note: Requires both ScalarOps and TensorOps bounds.
pub fn var_pow_scalar<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.pow_scalar(a.tensor(), scalar)?;

    if a.requires_grad() {
        let grad_fn = PowScalarBackward::<R>::new(a.id(), a.tensor().clone(), scalar);
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum along dimensions: z = sum(a, dims)
pub fn var_sum<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let output = client.sum(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = SumBackward::<R>::new(a.id(), a.shape(), dims, keepdim);
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Mean along dimensions: z = mean(a, dims)
pub fn var_mean<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: ScalarOps<R>,
{
    let output = client.mean(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = MeanBackward::<R>::new(a.id(), a.shape(), dims, keepdim);
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Max along dimensions: z = max(a, dims)
pub fn var_max<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.max(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = MaxBackward::<R>::new(a.id(), a.tensor().clone(), dims, keepdim);
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Min along dimensions: z = min(a, dims)
pub fn var_min<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.min(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = MinBackward::<R>::new(a.id(), a.tensor().clone(), dims, keepdim);
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Activation Operations
// ============================================================================

/// ReLU: z = max(0, a)
pub fn var_relu<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + CompareOps<R>,
{
    let output = client.relu(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = ReluBackward::<R>::new(a.id(), a.tensor().clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

impl_var_unary_op_output!(
    /// Sigmoid: z = 1 / (1 + exp(-a))
    var_sigmoid, sigmoid, SigmoidBackward
);

/// Softmax along dimension: z_i = exp(a_i) / sum(exp(a))
pub fn var_softmax<R, C>(a: &Var<R>, dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.softmax(a.tensor(), dim)?;

    if a.requires_grad() {
        let grad_fn = SoftmaxBackward::<R>::new(a.id(), output.clone(), dim);
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_mul_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let z = var_mul(&x, &y, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![6.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 3.0).abs() < 1e-6);
        assert!((grad_y[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_add_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let z = var_add(&x, &y, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 1.0).abs() < 1e-6);
        assert!((grad_y[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_chain_rule() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // z = (x * y) + x = x*(y+1)
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let xy = var_mul(&x, &y, &client).unwrap();
        let z = var_add(&xy, &x, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![8.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 4.0).abs() < 1e-6); // dz/dx = y + 1 = 4
        assert!((grad_y[0] - 2.0).abs() < 1e-6); // dz/dy = x = 2
    }

    #[test]
    fn test_var_exp_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device),
            true,
        );

        let z = var_exp(&x, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_matmul_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2, 1], &device),
            true,
        );

        let c = var_matmul(&a, &b, &client).unwrap();

        let c_data: Vec<f32> = c.tensor().to_vec();
        assert_eq!(c_data, vec![11.0]);

        let grads = backward(&c, &client).unwrap();

        let grad_a: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
        assert_eq!(grad_a, vec![3.0, 4.0]);

        let grad_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();
        assert_eq!(grad_b, vec![1.0, 2.0]);
    }

    #[test]
    fn test_var_tan_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device),
            true,
        );

        let z = var_tan(&x, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_add_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = var_add_scalar(&x, 5.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![7.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_mul_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = var_mul_scalar(&x, 3.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![6.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_pow_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let z = var_pow_scalar(&x, 2.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![9.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 6.0).abs() < 1e-6);
    }
}
