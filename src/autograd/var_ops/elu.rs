//! ELU with gradient support: `y = x` for `x > 0`, else `alpha * (exp(x) - 1)`.
//!
//! The derivative is `1` for `x > 0` and `y + alpha` otherwise.

use crate::autograd::Var;
use crate::autograd::var_ops::elementwise_derivative::unary_with_derivative;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ActivationOps, CompareOps, ConditionalOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// ELU with saturation value `alpha`.
pub fn var_elu<R, C>(a: &Var<R>, alpha: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let output = client.elu(a.tensor(), alpha)?;
    Ok(unary_with_derivative(
        a,
        output,
        "EluBackward",
        alpha,
        derivative::<R>,
    ))
}

fn derivative<R: Runtime<DType = DType>>(
    client: &R::Client,
    x: &Tensor<R>,
    y: &Tensor<R>,
    alpha: f64,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + ConditionalOps<R>,
{
    let zeros = Tensor::<R>::zeros(x.shape(), x.dtype(), x.device())?;
    let ones = Tensor::<R>::ones(x.shape(), x.dtype(), x.device())?;
    let positive = client.gt(x, &zeros)?;
    let negative_side = client.add_scalar(y, alpha)?;
    client.where_cond(&positive, &ones, &negative_side)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const X: [f64; 6] = [-2.3, -0.7, 0.2, 0.9, 1.6, -3.1];

    #[test]
    fn fd_elu() {
        check_grad_f64(
            "var_elu",
            |inputs, c| var_elu(&inputs[0], 1.7, c),
            &[f64_tensor(&X, &[2, 3])],
        );
    }

    #[test]
    fn positive_side_has_unit_gradient() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[0.5, 3.0], &[2]), true);
        let y = var_elu(&x, 2.0, &client).unwrap();
        let loss = crate::autograd::var_sum(&y, &[], false, &client).unwrap();
        let grads = crate::autograd::backward(&loss, &client).unwrap();
        assert_eq!(grads.get(x.id()).unwrap().to_vec::<f64>(), vec![1.0, 1.0]);
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[6]), false);
        let y = var_elu(&x, 1.0, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }
}
