//! Product reduction with gradient support: `y = prod(a, dims)`.
//!
//! The gradient to element `i` is `g` times the product of every other element
//! in its reduced group. The backward builds that product as an exclusive prefix
//! product times an exclusive suffix product. It never divides by `a`, so zeros
//! in the input give exact gradients.
//!
//! The first-order backward uses `cumprod`. The differentiable backward uses a
//! log-step prefix scan built from `var_mul`, `var_narrow` and `var_cat`,
//! because `var_cumprod` divides by its input and has no second-order graph.

use crate::autograd::var_ops::flip::{var_contiguous, var_flip};
use crate::autograd::var_ops::var_mul;
use crate::autograd::{GradFn, Var, var_cat, var_narrow, var_permute, var_reshape};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Product of `a` over `dims`.
///
/// An empty `dims` reduces every dimension. `keepdim` keeps each reduced
/// dimension with size 1. `dims` must be in range and must not repeat.
pub fn var_prod<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let dims = resolve_dims(dims, a.ndim())?;
    let output = client.prod(a.tensor(), &dims, keepdim)?;

    if a.requires_grad() {
        let mut kept_shape = a.shape().to_vec();
        for &d in &dims {
            kept_shape[d] = 1;
        }
        let grad_fn = ProdBackward::<R> {
            input_ids: [a.id()],
            saved: [a.tensor().clone()],
            plan: Plan::new(a.shape(), &dims),
            kept_shape,
            input_grad_fn: a.grad_fn().cloned(),
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Sorted, distinct dimensions. An empty list means every dimension.
fn resolve_dims(dims: &[usize], ndim: usize) -> Result<Vec<usize>> {
    if dims.is_empty() {
        return Ok((0..ndim).collect());
    }
    let mut sorted = dims.to_vec();
    sorted.sort_unstable();
    if let Some(&bad) = sorted.iter().find(|&&d| d >= ndim) {
        return Err(Error::InvalidDimension {
            dim: bad as isize,
            ndim,
        });
    }
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::InvalidArgument {
            arg: "dims",
            reason: format!("var_prod names a dimension more than once in {dims:?}"),
        });
    }
    Ok(sorted)
}

/// How to view the input as `[outer, k]` with the reduced dimensions last.
struct Plan {
    /// Kept dimensions, then reduced dimensions.
    perm: Vec<usize>,
    /// The inverse of `perm`.
    inverse: Vec<usize>,
    /// The input's shape reordered by `perm`.
    permuted_shape: Vec<usize>,
    /// Product of the kept sizes.
    outer: usize,
    /// Product of the reduced sizes.
    k: usize,
}

impl Plan {
    fn new(shape: &[usize], dims: &[usize]) -> Self {
        let mut perm: Vec<usize> = (0..shape.len()).filter(|d| !dims.contains(d)).collect();
        let outer = perm.iter().map(|&d| shape[d]).product();
        perm.extend_from_slice(dims);
        let k = dims.iter().map(|&d| shape[d]).product();
        let mut inverse = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inverse[p] = i;
        }
        let permuted_shape = perm.iter().map(|&d| shape[d]).collect();
        Self {
            perm,
            inverse,
            permuted_shape,
            outer,
            k,
        }
    }
}

/// Backward for `var_prod`: input `[a]`.
pub struct ProdBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    saved: [Tensor<R>; 1],
    plan: Plan,
    /// The input's shape with each reduced dimension set to 1.
    kept_shape: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime<DType = DType>> ProdBackward<R>
where
    R::Client: TensorOps<R>,
{
    /// Product of all other elements of each group, in the input's shape.
    fn others(&self, client: &R::Client) -> Result<Tensor<R>> {
        let [x] = &self.saved;
        let Plan {
            perm,
            inverse,
            permuted_shape,
            outer,
            k,
        } = &self.plan;
        let x2d = x.permute(perm)?.contiguous()?.reshape(&[*outer, *k])?;
        let forward = exclusive_cumprod(client, &x2d)?;
        let flipped = x2d.flip(1)?.contiguous()?;
        let reverse = exclusive_cumprod(client, &flipped)?.flip(1)?.contiguous()?;
        let product = client.mul(&forward, &reverse)?;
        Ok(product.reshape(permuted_shape)?.permute(inverse)?)
    }

    /// The same product as `others`, recorded on the graph for second-order use.
    fn others_var(&self, x: &Var<R>, client: &R::Client) -> Result<Var<R>> {
        let Plan {
            perm,
            inverse,
            permuted_shape,
            outer,
            k,
        } = &self.plan;
        let x2d = var_reshape(&var_contiguous(&var_permute(x, perm)?)?, &[*outer, *k])?;
        let forward = exclusive_prefix_var(&x2d, client)?;
        let flipped = var_flip(&x2d, &[1])?;
        let reverse = var_flip(&exclusive_prefix_var(&flipped, client)?, &[1])?;
        let product = var_mul(&forward, &reverse, client)?;
        var_permute(&var_reshape(&product, permuted_shape)?, inverse)
    }

    fn empty_grad(&self) -> Result<Tensor<R>> {
        let [x] = &self.saved;
        Tensor::<R>::zeros(x.shape(), x.dtype(), x.device())
    }
}

/// `out[:, i] = prod(x[:, ..i])` with `out[:, 0] = 1`, for a contiguous `[outer, k]`.
fn exclusive_cumprod<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    let [outer, k] = [x.shape()[0], x.shape()[1]];
    let ones = Tensor::<R>::ones(&[outer, 1], x.dtype(), x.device())?;
    if k == 1 {
        return Ok(ones);
    }
    let inclusive = client.cumprod(x, 1)?;
    client.cat(&[&ones, &inclusive.narrow(1, 0, k - 1)?], 1)
}

/// The differentiable form of `exclusive_cumprod`.
///
/// A log-step scan multiplies in the element `s` places back, for `s = 1, 2, 4, ..`.
/// Each step reads only products, so a zero input gives exact values and gradients.
fn exclusive_prefix_var<R>(x: &Var<R>, client: &R::Client) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>,
{
    let [outer, k] = [x.shape()[0], x.shape()[1]];
    let (dtype, device) = (x.tensor().dtype(), x.tensor().device());
    let ones = Var::new(Tensor::<R>::ones(&[outer, 1], dtype, device)?, false);
    if k == 1 {
        return Ok(ones);
    }
    let len = k - 1;
    let mut prefix = var_narrow(x, 1, 0, len)?;
    let mut step = 1;
    while step < len {
        let head = var_narrow(&prefix, 1, 0, step)?;
        let tail = var_narrow(&prefix, 1, step, len - step)?;
        let behind = var_narrow(&prefix, 1, 0, len - step)?;
        let scanned = var_mul(&tail, &behind, client)?;
        prefix = var_cat(&[&head, &scanned], 1, client)?;
        step *= 2;
    }
    var_cat(&[&ones, &prefix], 1, client)
}

impl<R: Runtime<DType = DType>> GradFn<R> for ProdBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        if self.saved[0].numel() == 0 {
            return Ok(vec![Some(self.empty_grad()?)]);
        }
        let client = R::default_client(grad_output.device());
        let g_kept = grad_output.contiguous()?.reshape(&self.kept_shape)?;
        let grad = client.mul(&g_kept, &self.others(&client)?)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        if self.saved[0].numel() == 0 {
            return Ok(vec![Some(Var::new(self.empty_grad()?, false))]);
        }
        let client = R::default_client(grad_output.tensor().device());
        let x = Var::with_id_and_grad_fn(
            self.saved[0].clone(),
            self.input_ids[0],
            self.input_grad_fn.clone(),
        );
        let g_kept = var_reshape(&var_contiguous(grad_output)?, &self.kept_shape)?;
        let grad = var_mul(&g_kept, &self.others_var(&x, &client)?, &client)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved
    }

    fn name(&self) -> &'static str {
        "ProdBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor, grad_var};
    use crate::autograd::{backward, backward_with_graph, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const X: [f64; 12] = [
        0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.5, -0.6, 0.8,
    ];

    fn grad_of_prod(values: &[f64], shape: &[usize], dims: &[usize]) -> Vec<f64> {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(values, shape), true);
        let y = var_prod(&x, dims, true, &client).unwrap();
        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        grads.get(x.id()).unwrap().to_vec::<f64>()
    }

    #[test]
    fn forward_matches_reduce_prod() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), false);
        let y = var_prod(&x, &[1], false, &client).unwrap();
        assert_eq!(y.tensor().to_vec::<f64>(), vec![6.0, 120.0]);
        let all = var_prod(&x, &[], true, &client).unwrap();
        assert_eq!(all.shape(), &[1, 1]);
        assert_eq!(all.tensor().to_vec::<f64>(), vec![720.0]);
    }

    #[test]
    fn fd_single_dim_keepdim_both() {
        for (dims, keepdim) in [([1usize], false), ([1], true), ([0], false), ([0], true)] {
            check_grad_f64(
                "var_prod single dim",
                move |inputs, c| var_prod(&inputs[0], &dims, keepdim, c),
                &[f64_tensor(&X, &[3, 4])],
            );
        }
    }

    #[test]
    fn fd_multi_dim() {
        for keepdim in [false, true] {
            check_grad_f64(
                "var_prod multi dim",
                move |inputs, c| var_prod(&inputs[0], &[0, 2], keepdim, c),
                &[f64_tensor(&X, &[2, 3, 2])],
            );
        }
        check_grad_f64(
            "var_prod all dims",
            |inputs, c| var_prod(&inputs[0], &[], true, c),
            &[f64_tensor(&X, &[3, 4])],
        );
    }

    #[test]
    fn zero_safe_with_one_zero() {
        assert_eq!(
            grad_of_prod(&[2.0, 0.0, 3.0], &[3], &[0]),
            vec![0.0, 6.0, 0.0]
        );
        assert_eq!(
            grad_of_prod(&[0.0, 2.0, 3.0], &[3], &[0]),
            vec![6.0, 0.0, 0.0]
        );
    }

    #[test]
    fn zero_safe_with_two_zeros() {
        assert_eq!(
            grad_of_prod(&[0.0, 0.0, 3.0], &[3], &[0]),
            vec![0.0, 0.0, 0.0]
        );
        let rows = grad_of_prod(&[0.0, 2.0, 0.0, 4.0, 0.0, 1.0], &[2, 3], &[1]);
        assert_eq!(rows, vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn fd_with_zeros_in_input() {
        check_grad_f64(
            "var_prod zeros dim 1",
            |inputs, c| var_prod(&inputs[0], &[1], false, c),
            &[f64_tensor(
                &[1.5, 0.0, 2.0, 0.0, 0.0, 0.7, -0.5, 1.2, 0.9],
                &[3, 3],
            )],
        );
        check_grad_f64(
            "var_prod zeros multi dim",
            |inputs, c| var_prod(&inputs[0], &[0, 2], true, c),
            &[f64_tensor(
                &[0.0, 1.0, 2.0, -1.0, 0.5, 0.0, 1.5, 0.0, 3.0, 0.4, -0.7, 2.0],
                &[2, 3, 2],
            )],
        );
    }

    #[test]
    fn size_one_reduced_dim_passes_gradient_through() {
        assert_eq!(grad_of_prod(&[3.0, 4.0], &[2, 1], &[1]), vec![1.0, 1.0]);
    }

    #[test]
    fn differentiable_others_match_cumprod_others_with_zeros() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let values = [0.0, 2.0, 3.0, 0.0, 4.0, 1.5, 0.0, 0.0, 2.5, 1.0, -2.0, 0.5];
        let x = Var::new(f64_tensor(&values, &[2, 3, 2]), true);
        let y = var_prod(&x, &[0, 2], false, &client).unwrap();
        let grad_fn = y.grad_fn().unwrap();
        let first = grad_fn
            .backward_all(&f64_tensor(&[1.0, 1.0, 1.0], &[3]))
            .unwrap();
        let second = grad_fn
            .backward_var(&Var::new(f64_tensor(&[1.0, 1.0, 1.0], &[3]), false))
            .unwrap();
        let a = first[0].as_ref().unwrap().to_vec::<f64>();
        let b = second[0].as_ref().unwrap().tensor().to_vec::<f64>();
        assert_eq!(a, b);
    }

    #[test]
    fn fd_second_order() {
        for dims in [vec![1usize], vec![0]] {
            check_grad_f64(
                "var_prod second order",
                move |inputs, c| {
                    let z = var_prod(&inputs[0], &dims, false, c)?;
                    let sq = crate::autograd::var_mul(&z, &z, c)?;
                    let loss = var_sum(&sq, &[], false, c)?;
                    let grads = backward_with_graph(&loss, c)?;
                    grad_var(&grads, inputs[0].id())
                },
                &[f64_tensor(&[0.5, -0.9, 1.3, 0.7, 1.1, -0.6], &[2, 3])],
            );
        }
    }

    #[test]
    fn bad_dims_are_errors() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[3, 4]), true);
        assert!(var_prod(&x, &[2], false, &client).is_err());
        assert!(var_prod(&x, &[1, 1], false, &client).is_err());
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&X, &[3, 4]), false);
        let y = var_prod(&x, &[1], false, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }
}
