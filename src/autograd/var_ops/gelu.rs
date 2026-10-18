//! GELU activations with gradient support: tanh form and exact erf form.
//!
//! The derivative comes from the fused `gelu_mul_bwd` and `gelu_erf_mul_bwd`
//! kernels called with `grad = 1` and `b = 1`, so every backend that has the
//! fused kernels also has these backwards.

use crate::autograd::Var;
use crate::autograd::var_ops::elementwise_derivative::unary_with_derivative;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ActivationOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// GELU, tanh approximation: `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn var_gelu<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ActivationOps<R>,
{
    let output = client.gelu(a.tensor())?;
    Ok(unary_with_derivative(
        a,
        output,
        "GeluBackward",
        0.0,
        gelu_derivative::<R>,
    ))
}

/// GELU, exact form: `y = 0.5 * x * (1 + erf(x / sqrt(2)))`.
pub fn var_gelu_erf<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ActivationOps<R>,
{
    let output = client.gelu_erf(a.tensor())?;
    Ok(unary_with_derivative(
        a,
        output,
        "GeluErfBackward",
        0.0,
        gelu_erf_derivative::<R>,
    ))
}

fn gelu_derivative<R: Runtime<DType = DType>>(
    client: &R::Client,
    x: &Tensor<R>,
    _y: &Tensor<R>,
    _param: f64,
) -> Result<Tensor<R>>
where
    R::Client: ActivationOps<R>,
{
    let x = x.contiguous()?;
    let ones = Tensor::<R>::ones(x.shape(), x.dtype(), x.device())?;
    Ok(client.gelu_mul_bwd(&ones, &x, &ones)?.0)
}

fn gelu_erf_derivative<R: Runtime<DType = DType>>(
    client: &R::Client,
    x: &Tensor<R>,
    _y: &Tensor<R>,
    _param: f64,
) -> Result<Tensor<R>>
where
    R::Client: ActivationOps<R>,
{
    let x = x.contiguous()?;
    let ones = Tensor::<R>::ones(x.shape(), x.dtype(), x.device())?;
    Ok(client.gelu_erf_mul_bwd(&ones, &x, &ones)?.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const X: [f64; 6] = [-2.3, -0.7, 0.2, 0.9, 1.6, 3.1];

    #[test]
    fn fd_gelu() {
        check_grad_f64(
            "var_gelu",
            |inputs, c| var_gelu(&inputs[0], c),
            &[f64_tensor(&X, &[2, 3])],
        );
    }

    #[test]
    fn fd_gelu_erf() {
        check_grad_f64(
            "var_gelu_erf",
            |inputs, c| var_gelu_erf(&inputs[0], c),
            &[f64_tensor(&X, &[2, 3])],
        );
    }

    #[test]
    fn forward_matches_client_op() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[6]), true);
        let want = client.gelu(x.tensor()).unwrap().to_vec::<f64>();
        let got = var_gelu(&x, &client).unwrap();
        assert_eq!(got.tensor().to_vec::<f64>(), want);
        let want = client.gelu_erf(x.tensor()).unwrap().to_vec::<f64>();
        let got = var_gelu_erf(&x, &client).unwrap();
        assert_eq!(got.tensor().to_vec::<f64>(), want);
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[6]), false);
        for y in [
            var_gelu(&x, &client).unwrap(),
            var_gelu_erf(&x, &client).unwrap(),
        ] {
            assert!(!y.requires_grad());
            assert!(y.grad_fn().is_none());
        }
    }
}
