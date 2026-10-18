//! L2 normalization with gradient support: `y = x / max(||x||_2, eps)`.
//!
//! The norm is taken along one dimension and kept as a size-1 dimension. This
//! matches the ggml `l2_norm` op. Forward composes existing tensor ops.
//!
//! Backward, with `n = max(||x||, eps)` and `s = <y, dy>` along the dimension:
//! - where `||x|| > eps`: `dx = (dy - y * s) / n`
//! - where `||x|| <= eps`: `dx = dy / eps`
//!
//! Both cases are one formula with `s` masked to zero in the second case. The
//! divisor `n` is never below `eps`, so a zero row gives a finite gradient.

use crate::autograd::var_ops::reduce::var_sum;
use crate::autograd::var_ops::{var_mul, var_sub};
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TensorOps, UtilityOps,
};
use crate::runtime::common::helpers::normalize_dim;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// L2 normalization along `dim`: `y = x / max(||x||_2, eps)`.
///
/// `dim` can be negative. `eps` must be finite and greater than zero. `x` must
/// have a float dtype. A row with norm at or below `eps` is scaled by `1 / eps`.
pub fn var_l2_normalize<R, C>(x: &Var<R>, dim: isize, eps: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidArgument {
            arg: "eps",
            reason: format!("l2_normalize needs a finite eps above zero, got {eps}"),
        });
    }
    let dtype = x.tensor().dtype();
    if !dtype.is_float() {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("l2_normalize needs a float dtype, got {dtype:?}"),
        });
    }
    let dim_idx = normalize_dim(dim, x.shape().len())?;

    let squares = client.mul(x.tensor(), x.tensor())?;
    let norm = client.sqrt(&client.sum(&squares, &[dim_idx], true)?)?;
    let eps_t = client.fill(norm.shape(), eps, dtype)?;
    let n = client.maximum(&norm, &eps_t)?;
    let output = client.div(x.tensor(), &n)?;

    if x.requires_grad() {
        let mask = client.gt(&norm, &eps_t)?;
        let grad_fn = L2NormalizeBackward::<R> {
            input_ids: [x.id()],
            saved: [output.clone(), client.recip(&n)?, mask],
            dim: dim_idx,
            input_grad_fn: x.grad_fn().cloned(),
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for `var_l2_normalize`: input `[x]`.
pub struct L2NormalizeBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    /// `[y, 1 / n, mask]`. The mask holds 1 where `||x|| > eps`, else 0.
    saved: [Tensor<R>; 3],
    dim: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime<DType = DType>> GradFn<R> for L2NormalizeBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let [y, inv_n, mask] = &self.saved;

        // s = <y, dy> along the dimension, kept as a size-1 dimension.
        let dot = client.mul(y, grad_output)?;
        let s = client.sum(&dot, &[self.dim], true)?;
        let zeros = client.fill(s.shape(), 0.0, s.dtype())?;
        let s_masked = client.where_cond(mask, &s, &zeros)?;

        let projected = client.mul(y, &s_masked)?;
        let dx = client.mul(&client.sub(grad_output, &projected)?, inv_n)?;
        Ok(vec![Some(dx)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let [y, inv_n, mask] = &self.saved;
        let y_var = Var::new(y.clone(), false);

        // The saved tensors are constants. The dependence on `dy` stays on the graph.
        let dot = var_mul(grad_output, &y_var, &client)?;
        let s = var_sum(&dot, &[self.dim], true, &client)?;
        let s_masked = var_mul(&s, &Var::new(mask.clone(), false), &client)?;
        let projected = var_mul(&y_var, &s_masked, &client)?;
        let diff = var_sub(grad_output, &projected, &client)?;
        let dx = var_mul(&diff, &Var::new(inv_n.clone(), false), &client)?;
        Ok(vec![Some(dx)])
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
        "L2NormalizeBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const EPS: f64 = 1e-3;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        (CpuRuntime::default_client(&device), device)
    }

    fn tensor(device: &CpuDevice, data: &[f64], shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
    }

    /// Fixed non-uniform upstream gradient, so `dy` differs per element.
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 0.5 + 0.37 * i as f64 - 0.11 * (i * i) as f64)
            .collect()
    }

    /// `sum(w * l2_normalize(x))` without autograd.
    fn loss(
        client: &CpuClient,
        device: &CpuDevice,
        data: &[f64],
        shape: &[usize],
        dim: isize,
    ) -> f64 {
        let x = Var::new(tensor(device, data, shape), false);
        let y = var_l2_normalize(&x, dim, EPS, client).unwrap();
        let w = weights(data.len());
        y.tensor()
            .to_vec::<f64>()
            .iter()
            .zip(w.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Analytic gradient of `sum(w * l2_normalize(x))`.
    fn analytic(
        client: &CpuClient,
        device: &CpuDevice,
        data: &[f64],
        shape: &[usize],
        dim: isize,
    ) -> Vec<f64> {
        let x = Var::new(tensor(device, data, shape), true);
        let y = var_l2_normalize(&x, dim, EPS, client).unwrap();
        let w = Var::new(tensor(device, &weights(data.len()), shape), false);
        let prod = var_mul(&y, &w, client).unwrap();
        let all: Vec<usize> = (0..shape.len()).collect();
        let total = var_sum(&prod, &all, false, client).unwrap();
        let grads = backward(&total, client).unwrap();
        grads.get(x.id()).expect("gradient present").to_vec()
    }

    fn numeric(
        client: &CpuClient,
        device: &CpuDevice,
        data: &[f64],
        shape: &[usize],
        dim: isize,
    ) -> Vec<f64> {
        let h = 1e-6;
        (0..data.len())
            .map(|i| {
                let mut plus = data.to_vec();
                let mut minus = data.to_vec();
                plus[i] += h;
                minus[i] -= h;
                (loss(client, device, &plus, shape, dim) - loss(client, device, &minus, shape, dim))
                    / (2.0 * h)
            })
            .collect()
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(g.is_finite(), "element {i} is not finite: {g}");
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "element {i}: {g} vs {w}"
            );
        }
    }

    const ROWS: [f64; 12] = [
        0.3, -1.2, 2.5, 0.7, 4.0, -0.7, 1.1, 0.9, -2.0, 0.5, 1.5, -0.25,
    ];

    #[test]
    fn gradient_matches_finite_differences_on_last_dim() {
        let (client, device) = setup();
        let got = analytic(&client, &device, &ROWS, &[3, 4], -1);
        let want = numeric(&client, &device, &ROWS, &[3, 4], -1);
        assert_close(&got, &want, 1e-6);
    }

    #[test]
    fn gradient_matches_finite_differences_on_other_dims() {
        let (client, device) = setup();
        for dim in [0isize, 1] {
            let got = analytic(&client, &device, &ROWS, &[2, 3, 2], dim);
            let want = numeric(&client, &device, &ROWS, &[2, 3, 2], dim);
            assert_close(&got, &want, 1e-6);
        }
        let got = analytic(&client, &device, &ROWS, &[3, 4], 0);
        let want = numeric(&client, &device, &ROWS, &[3, 4], 0);
        assert_close(&got, &want, 1e-6);
    }

    #[test]
    fn zero_row_gradient_is_finite_and_equals_dy_over_eps() {
        let (client, device) = setup();
        let mut data = ROWS.to_vec();
        data[4..8].fill(0.0);
        let got = analytic(&client, &device, &data, &[3, 4], -1);
        let w = weights(12);
        let want: Vec<f64> = (4..8).map(|i| w[i] / EPS).collect();
        assert_close(&got[4..8], &want, 1e-12);
        let fd = numeric(&client, &device, &data, &[3, 4], -1);
        assert_close(&got[0..4], &fd[0..4], 1e-6);
        assert_close(&got[8..12], &fd[8..12], 1e-6);
    }

    #[test]
    fn sub_eps_row_gradient_equals_dy_over_eps() {
        let (client, device) = setup();
        let mut data = ROWS.to_vec();
        data[0..4].copy_from_slice(&[1e-4, -2e-4, 3e-4, 1e-4]);
        let got = analytic(&client, &device, &data, &[3, 4], -1);
        let w = weights(12);
        let want: Vec<f64> = (0..4).map(|i| w[i] / EPS).collect();
        assert_close(&got[0..4], &want, 1e-12);
        let fd = numeric(&client, &device, &data, &[3, 4], -1);
        assert_close(&got, &fd, 1e-5);
    }

    #[test]
    fn forward_matches_manual_formula() {
        let (client, device) = setup();
        let mut data = ROWS.to_vec();
        data[8..12].fill(0.0);
        let x = Var::new(tensor(&device, &data, &[3, 4]), false);
        let y: Vec<f64> = var_l2_normalize(&x, -1, EPS, &client)
            .unwrap()
            .tensor()
            .to_vec();
        for r in 0..3 {
            let row = &data[r * 4..r * 4 + 4];
            let n = row.iter().map(|v| v * v).sum::<f64>().sqrt().max(EPS);
            for c in 0..4 {
                assert!((y[r * 4 + c] - row[c] / n).abs() < 1e-14);
            }
        }
        assert!(y[8..12].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn shapes_and_grad_flags() {
        let (client, device) = setup();
        let x = Var::new(tensor(&device, &ROWS, &[2, 3, 2]), true);
        let y = var_l2_normalize(&x, 1, EPS, &client).unwrap();
        assert_eq!(y.tensor().shape(), &[2, 3, 2]);
        assert!(y.requires_grad());
        let g = analytic(&client, &device, &ROWS, &[2, 3, 2], 1);
        assert_eq!(g.len(), 12);
        let frozen = Var::new(tensor(&device, &ROWS, &[3, 4]), false);
        let y = var_l2_normalize(&frozen, 0, EPS, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let (client, device) = setup();
        let x = Var::new(tensor(&device, &ROWS, &[3, 4]), true);
        assert!(var_l2_normalize(&x, 2, EPS, &client).is_err());
        assert!(var_l2_normalize(&x, -3, EPS, &client).is_err());
        assert!(var_l2_normalize(&x, 1, 0.0, &client).is_err());
        assert!(var_l2_normalize(&x, 1, f64::NAN, &client).is_err());
        let ints = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[2, 2], &device).unwrap(),
            false,
        );
        assert!(var_l2_normalize(&ints, 1, EPS, &client).is_err());
    }
}
