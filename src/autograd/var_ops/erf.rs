//! Error function with gradient support: `d erf(x) / dx = 2 / sqrt(pi) * exp(-x^2)`.

use crate::autograd::Var;
use crate::autograd::var_ops::elementwise_derivative::unary_with_derivative;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, ScalarOps, SpecialFunctions, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Element-wise error function.
pub fn var_erf<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + SpecialFunctions<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R>,
{
    let output = client.erf(a.tensor())?;
    Ok(unary_with_derivative(
        a,
        output,
        "ErfBackward",
        0.0,
        derivative::<R>,
    ))
}

fn derivative<R: Runtime<DType = DType>>(
    client: &R::Client,
    x: &Tensor<R>,
    _y: &Tensor<R>,
    _param: f64,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    let gauss = client.exp(&client.neg(&client.mul(x, x)?)?)?;
    client.mul_scalar(&gauss, std::f64::consts::FRAC_2_SQRT_PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn fd_erf() {
        check_grad_f64(
            "var_erf",
            |inputs, c| var_erf(&inputs[0], c),
            &[f64_tensor(&[-2.1, -0.6, 0.0, 0.4, 1.3, 2.7], &[2, 3])],
        );
    }

    #[test]
    fn gradient_at_zero_is_two_over_sqrt_pi() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[0.0], &[1]), true);
        let y = var_erf(&x, &client).unwrap();
        let loss = crate::autograd::var_sum(&y, &[], false, &client).unwrap();
        let grads = crate::autograd::backward(&loss, &client).unwrap();
        let g = grads.get(x.id()).unwrap().to_vec::<f64>()[0];
        assert!((g - std::f64::consts::FRAC_2_SQRT_PI).abs() < 1e-12);
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[0.3, -0.8], &[2]), false);
        let y = var_erf(&x, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }
}
