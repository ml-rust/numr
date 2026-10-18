//! Reciprocal square root with gradient support: `d rsqrt(x) / dx = -0.5 * y^3`.

use crate::autograd::Var;
use crate::autograd::var_ops::elementwise_derivative::unary_with_derivative;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Element-wise `1 / sqrt(x)`.
pub fn var_rsqrt<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + UnaryOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.rsqrt(a.tensor())?;
    Ok(unary_with_derivative(
        a,
        output,
        "RsqrtBackward",
        0.0,
        derivative::<R>,
    ))
}

fn derivative<R: Runtime<DType = DType>>(
    client: &R::Client,
    _x: &Tensor<R>,
    y: &Tensor<R>,
    _param: f64,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R>,
{
    let cube = client.mul(&client.mul(y, y)?, y)?;
    client.mul_scalar(&cube, -0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn fd_rsqrt() {
        check_grad_f64(
            "var_rsqrt",
            |inputs, c| var_rsqrt(&inputs[0], c),
            &[f64_tensor(&[0.3, 0.8, 1.5, 2.4, 4.0, 9.0], &[2, 3])],
        );
    }

    #[test]
    fn gradient_at_four_is_minus_one_sixteenth() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[4.0], &[1]), true);
        let y = var_rsqrt(&x, &client).unwrap();
        let loss = crate::autograd::var_sum(&y, &[], false, &client).unwrap();
        let grads = crate::autograd::backward(&loss, &client).unwrap();
        let g = grads.get(x.id()).unwrap().to_vec::<f64>()[0];
        assert!((g + 0.0625).abs() < 1e-12);
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[0.5, 2.0], &[2]), false);
        let y = var_rsqrt(&x, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }
}
