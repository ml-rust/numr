//! Leaky ReLU with gradient support: `y = x` for `x > 0`, else `slope * x`.
//!
//! The derivative is `1` for `x > 0` and `slope` otherwise, so it is `slope` at `x == 0`.

use crate::autograd::Var;
use crate::autograd::var_ops::elementwise_derivative::unary_with_derivative;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ActivationOps, CompareOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Leaky ReLU with negative slope `slope`.
pub fn var_leaky_relu<R, C>(a: &Var<R>, slope: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let output = client.leaky_relu(a.tensor(), slope)?;
    Ok(unary_with_derivative(
        a,
        output,
        "LeakyReluBackward",
        slope,
        derivative::<R>,
    ))
}

/// `slope + (1 - slope) * (x > 0)`.
fn derivative<R: Runtime<DType = DType>>(
    client: &R::Client,
    x: &Tensor<R>,
    _y: &Tensor<R>,
    slope: f64,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let zeros = Tensor::<R>::zeros(x.shape(), x.dtype(), x.device())?;
    let positive = client.gt(x, &zeros)?;
    let scaled = client.mul_scalar(&positive, 1.0 - slope)?;
    client.add_scalar(&scaled, slope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const X: [f64; 6] = [-2.3, -0.7, 0.2, 0.9, 1.6, -3.1];

    #[test]
    fn fd_leaky_relu() {
        check_grad_f64(
            "var_leaky_relu",
            |inputs, c| var_leaky_relu(&inputs[0], 0.1, c),
            &[f64_tensor(&X, &[2, 3])],
        );
    }

    #[test]
    fn slope_applies_to_negative_inputs_only() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[-1.0, 2.0], &[2]), true);
        let y = var_leaky_relu(&x, 0.25, &client).unwrap();
        let loss = crate::autograd::var_sum(&y, &[], false, &client).unwrap();
        let grads = crate::autograd::backward(&loss, &client).unwrap();
        assert_eq!(grads.get(x.id()).unwrap().to_vec::<f64>(), vec![0.25, 1.0]);
        assert_eq!(y.tensor().to_vec::<f64>(), vec![-0.25, 2.0]);
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[6]), false);
        let y = var_leaky_relu(&x, 0.1, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }
}
