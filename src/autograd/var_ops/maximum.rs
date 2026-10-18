//! Element-wise `maximum` with gradient support.
//!
//! The gradient goes to the larger input. A tie splits it 0.5 and 0.5. The
//! shared backward lives in `binary_extremum`.

use crate::autograd::Var;
use crate::autograd::var_ops::binary_extremum::{Extremum, var_extremum};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, CompareOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};

/// Element-wise maximum of `a` and `b` with broadcasting.
pub fn var_maximum<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    var_extremum(a, b, Extremum::Maximum, client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::autograd::{backward, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn fd_same_shape() {
        check_grad_f64(
            "var_maximum same shape",
            |inputs, c| var_maximum(&inputs[0], &inputs[1], c),
            &[
                f64_tensor(&[0.5, -1.0, 2.0, 0.3, -0.7, 1.1], &[2, 3]),
                f64_tensor(&[1.5, 0.25, -2.0, 0.8, 0.9, -1.4], &[2, 3]),
            ],
        );
    }

    #[test]
    fn fd_broadcast() {
        check_grad_f64(
            "var_maximum broadcast",
            |inputs, c| var_maximum(&inputs[0], &inputs[1], c),
            &[
                f64_tensor(&[0.5, -1.0, 2.0, 0.3, -0.7, 1.1], &[2, 3]),
                f64_tensor(&[0.45, -0.2, 1.4], &[3]),
            ],
        );
        check_grad_f64(
            "var_maximum broadcast both",
            |inputs, c| var_maximum(&inputs[0], &inputs[1], c),
            &[
                f64_tensor(&[0.5, -1.0, 2.0], &[3, 1]),
                f64_tensor(&[0.45, -0.2, 1.4, 0.9], &[1, 4]),
            ],
        );
    }

    #[test]
    fn ties_split_the_gradient_evenly() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0, 3.0], &[3]), true);
        let b = Var::new(f64_tensor(&[1.0, 5.0, 0.0], &[3]), true);
        let z = var_maximum(&a, &b, &client).unwrap();
        let loss = var_sum(&z, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        let ga = grads.get(a.id()).unwrap().to_vec::<f64>();
        let gb = grads.get(b.id()).unwrap().to_vec::<f64>();
        assert_eq!(ga[0], 0.5);
        assert_eq!(gb[0], 0.5);
        assert_eq!(ga[1] + gb[1], 1.0);
        assert_eq!(ga[2] + gb[2], 1.0);
        assert!(ga[1] != gb[1] && ga[2] != gb[2]);
    }

    #[test]
    fn no_grad_inputs_give_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0], &[2]), false);
        let b = Var::new(f64_tensor(&[2.0, 1.0], &[2]), false);
        let z = var_maximum(&a, &b, &client).unwrap();
        assert!(!z.requires_grad());
        assert!(z.grad_fn().is_none());
    }
}
