//! PReLU with gradient support: `y = x` for `x > 0`, else `weight * x`.
//!
//! Built from `var_where` and `var_mul`, so the gradient for `weight` is summed
//! back to its own shape over the broadcast dimensions. The gradient for `x` is
//! `1` for `x > 0` and `weight` otherwise, so it is `weight` at `x == 0`.

use crate::autograd::Var;
use crate::autograd::var_ops::{var_mul, var_where};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::activation_common::validate_prelu;
use crate::ops::{CompareOps, ConditionalOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// PReLU with a learned slope `weight`.
///
/// `weight` must have the dtype of `a` and must broadcast to `a` without
/// changing its shape.
pub fn var_prelu<R, C>(a: &Var<R>, weight: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + CompareOps<R> + ConditionalOps<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    validate_prelu(
        a.tensor().dtype(),
        a.shape(),
        weight.tensor().dtype(),
        weight.shape(),
    )?;

    // A rank-0 zero broadcasts to any shape, so the mask needs no full-size zeros.
    let zero = Tensor::<R>::zeros(&[], a.tensor().dtype(), a.tensor().device())?;
    let positive = client.gt(a.tensor(), &zero)?;
    let scaled = var_mul(a, weight, client)?;
    var_where(&positive, a, &scaled, client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::autograd::{backward, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const X: [f64; 8] = [-2.3, -0.7, 0.2, 0.9, 1.6, -3.1, 0.4, -1.2];

    #[test]
    fn fd_prelu_scalar_weight() {
        check_grad_f64(
            "var_prelu scalar weight",
            |inputs, c| var_prelu(&inputs[0], &inputs[1], c),
            &[f64_tensor(&X, &[2, 4]), f64_tensor(&[0.3], &[1])],
        );
    }

    #[test]
    fn fd_prelu_per_channel_weight() {
        // [N=1, C=2, W=4], weight [C, 1].
        check_grad_f64(
            "var_prelu per channel",
            |inputs, c| var_prelu(&inputs[0], &inputs[1], c),
            &[
                f64_tensor(&X, &[1, 2, 4]),
                f64_tensor(&[0.3, 0.15], &[2, 1]),
            ],
        );
    }

    #[test]
    fn gradients_follow_the_positive_mask_and_reduce_to_weight_shape() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[-1.0, 2.0, 0.0, -4.0], &[2, 2]), true);
        let w = Var::new(f64_tensor(&[0.5, 0.25], &[2]), true);
        let y = var_prelu(&x, &w, &client).unwrap();
        assert_eq!(y.tensor().to_vec::<f64>(), vec![-0.5, 2.0, 0.0, -1.0]);
        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        // dx is 1 where x > 0, else the weight (including x == 0).
        assert_eq!(
            grads.get(x.id()).unwrap().to_vec::<f64>(),
            vec![0.5, 1.0, 0.5, 0.25]
        );
        // dw sums x over the non-positive entries of each column.
        let dw = grads.get(w.id()).unwrap();
        assert_eq!(dw.shape(), &[2]);
        assert_eq!(dw.to_vec::<f64>(), vec![-1.0, -4.0]);
    }

    #[test]
    fn rejects_non_broadcasting_weight() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[2, 4]), true);
        let w = Var::new(f64_tensor(&[0.1, 0.2, 0.3], &[3]), true);
        assert!(var_prelu(&x, &w, &client).is_err());
    }

    #[test]
    fn no_grad_inputs_give_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[8]), false);
        let w = Var::new(f64_tensor(&[0.1], &[1]), false);
        let y = var_prelu(&x, &w, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }
}
