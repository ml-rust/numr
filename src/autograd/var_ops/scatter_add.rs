//! Scatter-add: sum `src` into `target` at positions given by an index tensor

use super::ops::*;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{IndexingOps, ScatterReduceOp};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Scatter-add along a dimension.
///
/// z = scatter_reduce(target, dim, index, src, Sum, include_self=true)
///
/// For `dim == 1` on 2-D inputs, `z[i][index[i][j]] += src[i][j]`, starting
/// from `z = target`. A repeated index sums every `src` element that names it.
///
/// `index` holds integer positions and is not differentiable. `index` and `src`
/// must have the same shape, and the backend returns `ShapeMismatch` otherwise.
///
/// Gradients flow to both `target` and `src`, and the backward is itself
/// differentiable.
pub fn var_scatter_add<R, C>(
    target: &Var<R>,
    dim: usize,
    index: &Tensor<R>,
    src: &Var<R>,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R>,
    R::Client: IndexingOps<R>,
{
    let output = client.scatter_reduce(
        target.tensor(),
        dim,
        index,
        src.tensor(),
        ScatterReduceOp::Sum,
        true,
    )?;

    if target.requires_grad() || src.requires_grad() {
        let grad_fn = ScatterAddBackward::<R>::new(
            target.id(),
            src.id(),
            dim,
            index.clone(),
            target.grad_fn().cloned(),
            src.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::autograd::{backward, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn index(values: &[i64], shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(values, shape, &CpuDevice::new()).unwrap()
    }

    #[test]
    fn forward_sums_repeated_indices() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let target = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]), false);
        let src = Var::new(f64_tensor(&[10.0, 20.0, 30.0], &[3]), false);
        let idx = index(&[0, 0, 2], &[3]);

        let z = var_scatter_add(&target, 0, &idx, &src, &client).unwrap();
        assert!(!z.requires_grad());
        assert_eq!(z.tensor().to_vec::<f64>(), vec![31.0, 2.0, 33.0, 4.0]);
    }

    #[test]
    fn src_grad_gathers_and_target_grad_passes_through() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let target = Var::new(f64_tensor(&[0.0; 4], &[4]), true);
        let src = Var::new(f64_tensor(&[1.0, 1.0, 1.0], &[3]), true);
        let idx = index(&[0, 0, 2], &[3]);
        let weight = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]), false);

        let z = var_scatter_add(&target, 0, &idx, &src, &client).unwrap();
        let weighted = crate::autograd::var_mul(&z, &weight, &client).unwrap();
        let loss = var_sum(&weighted, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let g_target: Vec<f64> = grads.get(target.id()).unwrap().to_vec();
        let g_src: Vec<f64> = grads.get(src.id()).unwrap().to_vec();
        assert_eq!(g_target, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(g_src, vec![1.0, 1.0, 3.0]);
    }

    #[test]
    fn fd_dim0_duplicate_indices() {
        let idx = index(&[0, 2, 1, 0, 2, 2, 0, 1], &[2, 4]);
        check_grad_f64(
            "var_scatter_add dim 0",
            move |inputs, c| var_scatter_add(&inputs[0], 0, &idx, &inputs[1], c),
            &[
                f64_tensor(
                    &[
                        0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.0, -0.6, 0.8,
                    ],
                    &[3, 4],
                ),
                f64_tensor(&[1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -2.5], &[2, 4]),
            ],
        );
    }

    #[test]
    fn fd_dim1_duplicate_indices() {
        let idx = index(&[1, 1, 0, 2, 2, 2], &[2, 3]);
        check_grad_f64(
            "var_scatter_add dim 1",
            move |inputs, c| var_scatter_add(&inputs[0], 1, &idx, &inputs[1], c),
            &[
                f64_tensor(&[0.5, -0.25, 1.75, -1.0, 2.5, 0.125], &[2, 3]),
                f64_tensor(&[0.9, -1.1, 0.4, 1.3, -0.7, 0.2], &[2, 3]),
            ],
        );
    }

    #[test]
    fn fd_double_backward_through_scatter_add() {
        // g(src) = d/dsrc sum(c * scatter_add(t, src)^3), which records a graph.
        // The check differentiates g, so it covers `ScatterAddBackward::backward_var`.
        let idx = index(&[0, 0, 2, 1], &[4]);
        check_grad_f64(
            "scatter_add second order",
            move |inputs, c| {
                let target = Var::new(f64_tensor(&[0.5, -1.0, 0.25], &[3]), false);
                let z = var_scatter_add(&target, 0, &idx, &inputs[0], c)?;
                let sq = crate::autograd::var_mul(&z, &z, c)?;
                let cube = crate::autograd::var_mul(&sq, &z, c)?;
                let loss = var_sum(&cube, &[], false, c)?;
                let grads = crate::autograd::backward_with_graph(&loss, c)?;
                crate::autograd::test_fd::grad_var(&grads, inputs[0].id())
            },
            &[f64_tensor(&[0.3, -0.8, 1.2, 0.6], &[4])],
        );
    }
}
