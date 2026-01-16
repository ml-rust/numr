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
// Binary Operations
// ============================================================================

/// Add two variables: z = a + b
///
/// Creates AddBackward for gradient computation.
pub fn var_add<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.add(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = AddBackward::<R>::new(
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

/// Subtract two variables: z = a - b
///
/// Creates SubBackward for gradient computation.
pub fn var_sub<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.sub(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = SubBackward::<R>::new(
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

/// Multiply two variables: z = a * b
///
/// Creates MulBackward for gradient computation.
pub fn var_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.mul(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = MulBackward::<R>::new(
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

/// Divide two variables: z = a / b
///
/// Creates DivBackward for gradient computation.
pub fn var_div<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.div(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = DivBackward::<R>::new(
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

/// Power: z = a^b
///
/// Creates PowBackward for gradient computation.
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

/// Matrix multiplication: z = a @ b
///
/// Creates MatmulBackward for gradient computation.
pub fn var_matmul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.matmul(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = MatmulBackward::<R>::new(
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

// ============================================================================
// Unary Operations
// ============================================================================

/// Negation: z = -a
pub fn var_neg<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.neg(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = NegBackward::<R>::new(a.id());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Exponential: z = exp(a)
pub fn var_exp<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.exp(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = ExpBackward::<R>::new(a.id(), output.clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Natural logarithm: z = log(a)
pub fn var_log<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.log(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = LogBackward::<R>::new(a.id(), a.tensor().clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Square root: z = sqrt(a)
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

/// Sine: z = sin(a)
pub fn var_sin<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.sin(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SinBackward::<R>::new(a.id(), a.tensor().clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Cosine: z = cos(a)
pub fn var_cos<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.cos(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = CosBackward::<R>::new(a.id(), a.tensor().clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Hyperbolic tangent: z = tanh(a)
pub fn var_tanh<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.tanh(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = TanhBackward::<R>::new(a.id(), output.clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Square: z = a²
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

/// Reciprocal: z = 1/a
pub fn var_recip<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.recip(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = RecipBackward::<R>::new(a.id(), output.clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Absolute value: z = |a|
pub fn var_abs<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.abs(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = AbsBackward::<R>::new(a.id(), a.tensor().clone());
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

/// Sigmoid: z = 1 / (1 + exp(-a))
pub fn var_sigmoid<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.sigmoid(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SigmoidBackward::<R>::new(a.id(), output.clone());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

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

        // z = x * y where x = 2, y = 3
        // z = 6
        // dz/dx = y = 3
        // dz/dy = x = 2
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let z = var_mul(&x, &y, &client).unwrap();

        // Verify forward
        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![6.0]);

        // Backward
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

        // z = x + y
        // dz/dx = 1, dz/dy = 1
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

        // z = (x * y) + x
        // z = x*y + x = x*(y+1)
        // dz/dx = y + 1 = 4
        // dz/dy = x = 2
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

        // z = 2*3 + 2 = 8
        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![8.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        // dz/dx = d(xy + x)/dx = y + 1 = 4
        assert!((grad_x[0] - 4.0).abs() < 1e-6);
        // dz/dy = d(xy + x)/dy = x = 2
        assert!((grad_y[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_exp_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // z = exp(x) where x = 0
        // dz/dx = exp(x) = 1
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

        // A = [[1, 2]] (1x2), B = [[3], [4]] (2x1)
        // C = A @ B = [[11]] (1x1)
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2, 1], &device),
            true,
        );

        let c = var_matmul(&a, &b, &client).unwrap();

        // Verify forward
        let c_data: Vec<f32> = c.tensor().to_vec();
        assert_eq!(c_data, vec![11.0]);

        let grads = backward(&c, &client).unwrap();

        // dC/dA = dC/dC @ B^T = [[1]] @ [[3, 4]] = [[3, 4]]
        let grad_a: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
        assert_eq!(grad_a, vec![3.0, 4.0]);

        // dC/dB = A^T @ dC/dC = [[1], [2]] @ [[1]] = [[1], [2]]
        let grad_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();
        assert_eq!(grad_b, vec![1.0, 2.0]);
    }
}
