//! Einstein summation with gradient support.
//!
//! The gradient to operand `k` is an einsum of the output gradient and every
//! other operand, written to operand `k`'s labels. A label that only operand `k`
//! holds was summed away in the forward. Its gradient is the same for every
//! index of that label, so the result broadcasts back along it.
//!
//! This first cut rejects a repeated label within one operand (traces and
//! diagonals) and the ellipsis. Both return `InvalidArgument`.

use crate::autograd::var_ops::flip::var_contiguous;
use crate::autograd::{GradFn, Var, var_broadcast_to, var_reshape};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{EinsumOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Evaluate the einsum `equation` over `operands`.
///
/// Labels are lowercase ASCII letters. The equation can be explicit
/// (`"ij,jk->ik"`) or implicit (`"ij,jk"`), where the output holds the labels
/// that occur once, in sorted order. Whitespace is ignored.
///
/// A label must not repeat within one operand, and `...` is not supported.
pub fn var_einsum<R, C>(equation: &str, operands: &[&Var<R>], client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + EinsumOps<R>,
    R::Client: EinsumOps<R> + TensorOps<R>,
{
    let spec = Spec::parse(equation, operands.len())?;
    let tensors: Vec<&Tensor<R>> = operands.iter().map(|v| v.tensor()).collect();
    let output = client.einsum(&spec.equation(), &tensors)?;

    if operands.iter().any(|v| v.requires_grad()) {
        let grad_fn = EinsumBackward::<R> {
            input_ids: operands.iter().map(|v| v.graph_id()).collect(),
            saved: tensors.into_iter().cloned().collect(),
            spec,
            input_grad_fns: operands.iter().map(|v| v.grad_fn().cloned()).collect(),
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// A parsed equation: one label list per operand and the output labels.
struct Spec {
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
}

impl Spec {
    fn parse(equation: &str, operand_count: usize) -> Result<Self> {
        let text: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
        if text.contains('.') {
            return Err(invalid("ellipsis is not supported by var_einsum"));
        }
        let (lhs, rhs) = match text.split_once("->") {
            Some((lhs, rhs)) => (lhs, Some(rhs)),
            None => (text.as_str(), None),
        };

        let inputs: Vec<Vec<char>> = lhs.split(',').map(|part| part.chars().collect()).collect();
        if inputs.len() != operand_count {
            return Err(invalid(&format!(
                "equation names {} operands but {operand_count} were given",
                inputs.len()
            )));
        }

        let mut counts: BTreeMap<char, usize> = BTreeMap::new();
        for (i, labels) in inputs.iter().enumerate() {
            check_labels(labels)?;
            for (pos, &c) in labels.iter().enumerate() {
                if labels[..pos].contains(&c) {
                    return Err(invalid(&format!(
                        "label '{c}' repeats within operand {i}, which var_einsum does not support"
                    )));
                }
                *counts.entry(c).or_insert(0) += 1;
            }
        }

        let output: Vec<char> = match rhs {
            Some(rhs) => {
                let labels: Vec<char> = rhs.chars().collect();
                check_labels(&labels)?;
                for (pos, &c) in labels.iter().enumerate() {
                    if labels[..pos].contains(&c) {
                        return Err(invalid(&format!("output label '{c}' repeats")));
                    }
                    if !counts.contains_key(&c) {
                        return Err(invalid(&format!("output label '{c}' is in no operand")));
                    }
                }
                labels
            }
            None => counts
                .iter()
                .filter(|&(_, &n)| n == 1)
                .map(|(&c, _)| c)
                .collect(),
        };
        Ok(Self { inputs, output })
    }

    /// The explicit form of the equation.
    fn equation(&self) -> String {
        explicit(&self.inputs, &self.output)
    }
}

fn explicit(inputs: &[Vec<char>], output: &[char]) -> String {
    let lhs: Vec<String> = inputs.iter().map(|l| l.iter().collect()).collect();
    format!("{}->{}", lhs.join(","), output.iter().collect::<String>())
}

fn check_labels(labels: &[char]) -> Result<()> {
    match labels.iter().find(|c| !c.is_ascii_lowercase()) {
        Some(c) => Err(invalid(&format!(
            "label '{c}' is not a lowercase ASCII letter"
        ))),
        None => Ok(()),
    }
}

fn invalid(reason: &str) -> Error {
    Error::InvalidArgument {
        arg: "equation",
        reason: reason.to_string(),
    }
}

/// Backward for `var_einsum`: one input per operand.
pub struct EinsumBackward<R: Runtime> {
    input_ids: Vec<TensorId>,
    saved: Vec<Tensor<R>>,
    spec: Spec,
    input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
}

impl<R: Runtime<DType = DType>> EinsumBackward<R>
where
    R::Client: EinsumOps<R> + TensorOps<R>,
{
    /// The gradient to operand `k` for output gradient `g`, as a Var on the graph.
    fn operand_grad(&self, k: usize, g: &Var<R>, operands: &[Var<R>]) -> Result<Var<R>> {
        let client = R::default_client(g.tensor().device());
        let labels = &self.spec.inputs[k];

        let mut subscripts = vec![self.spec.output.clone()];
        let mut vars = vec![g];
        for (j, v) in operands.iter().enumerate().filter(|&(j, _)| j != k) {
            subscripts.push(self.spec.inputs[j].clone());
            vars.push(v);
        }

        // Labels in no other operand and not in the output were summed in the forward.
        let reachable = |c: &char| subscripts.iter().any(|s| s.contains(c));
        let target: Vec<char> = labels.iter().copied().filter(reachable).collect();
        let reduced = var_einsum(&explicit(&subscripts, &target), &vars, &client)?;
        if target.len() == labels.len() {
            return Ok(reduced);
        }

        let shape = operands[k].shape();
        let singleton: Vec<usize> = labels
            .iter()
            .zip(shape)
            .map(|(c, &n)| if target.contains(c) { n } else { 1 })
            .collect();
        let reduced = var_reshape(&var_contiguous(&reduced)?, &singleton)?;
        var_broadcast_to(&reduced, shape)
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for EinsumBackward<R>
where
    R::Client: EinsumOps<R> + TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        let operands: Vec<Var<R>> = self
            .saved
            .iter()
            .map(|t| Var::new(t.clone(), false))
            .collect();
        let g = Var::new(grad_output.clone(), false);
        (0..self.saved.len())
            .map(|k| -> Result<Option<Tensor<R>>> {
                if needed.get(k).copied().unwrap_or(true) {
                    let grad = self.operand_grad(k, &g, &operands)?;
                    Ok(Some(grad.tensor().contiguous()?))
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let operands: Vec<Var<R>> = (0..self.saved.len())
            .map(|j| {
                Var::with_id_and_grad_fn(
                    self.saved[j].clone(),
                    self.input_ids[j],
                    self.input_grad_fns[j].clone(),
                )
            })
            .collect();
        (0..self.saved.len())
            .map(|k| -> Result<Option<Var<R>>> {
                Ok(Some(self.operand_grad(k, grad_output, &operands)?))
            })
            .collect()
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.clone()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved
    }

    fn name(&self) -> &'static str {
        "EinsumBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor, grad_var};
    use crate::autograd::{backward, backward_with_graph, var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const P: [f64; 12] = [
        0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.5, -0.6, 0.8,
    ];
    const Q: [f64; 12] = [
        1.4, 0.2, -0.3, 0.6, -1.7, 2.5, 0.35, -0.9, 1.25, -0.45, 0.75, 1.05,
    ];

    fn fd(name: &str, equation: &'static str, shapes: &[&[usize]]) {
        let inputs: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let n: usize = s.iter().product();
                let src = if i % 2 == 0 { &P } else { &Q };
                f64_tensor(&src[..n], s)
            })
            .collect();
        check_grad_f64(
            name,
            move |vars, c| {
                let refs: Vec<&Var<CpuRuntime>> = vars.iter().collect();
                var_einsum(equation, &refs, c)
            },
            &inputs,
        );
    }

    #[test]
    fn fd_matmul() {
        fd("einsum matmul", "ij,jk->ik", &[&[2, 3], &[3, 4]]);
    }

    #[test]
    fn fd_batched_matmul() {
        fd("einsum batched", "bij,bjk->bik", &[&[2, 2, 3], &[2, 3, 2]]);
    }

    #[test]
    fn fd_outer_product() {
        fd("einsum outer", "i,j->ij", &[&[3], &[4]]);
    }

    #[test]
    fn fd_single_operand_reduction() {
        fd("einsum reduce", "ij->i", &[&[3, 4]]);
        fd("einsum transpose", "ij->ji", &[&[3, 4]]);
    }

    #[test]
    fn fd_three_operands() {
        fd("einsum chain", "ij,jk,kl->il", &[&[2, 3], &[3, 2], &[2, 3]]);
    }

    #[test]
    fn fd_labels_summed_in_one_operand() {
        fd("einsum k only in b", "ij,jk->i", &[&[2, 3], &[3, 2]]);
        fd("einsum disjoint sums", "ij,kl->i", &[&[2, 3], &[2, 2]]);
        fd("einsum shared sum", "ij,ij->i", &[&[3, 4], &[3, 4]]);
    }

    #[test]
    fn fd_implicit_output() {
        fd("einsum implicit", "ij,jk", &[&[2, 3], &[3, 4]]);
    }

    #[test]
    fn fd_second_order() {
        check_grad_f64(
            "einsum second order",
            |inputs, c| {
                let w = Var::new(f64_tensor(&Q[..6], &[3, 2]), false);
                let z = var_einsum("ij,jk->ik", &[&inputs[0], &w], c)?;
                let sq = var_mul(&z, &z, c)?;
                let loss = var_sum(&sq, &[], false, c)?;
                let grads = backward_with_graph(&loss, c)?;
                grad_var(&grads, inputs[0].id())
            },
            &[f64_tensor(&P[..6], &[2, 3])],
        );
    }

    #[test]
    fn grad_free_operand_gets_no_gradient_but_the_other_does() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let b = Var::new(f64_tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]), true);
        let z = var_einsum("ij,jk->ik", &[&a, &b], &client).unwrap();
        assert!(z.requires_grad());
        let loss = var_sum(&z, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(
            grads.get(b.id()).unwrap().to_vec::<f64>(),
            vec![4.0, 4.0, 6.0, 6.0]
        );
    }

    #[test]
    fn unsupported_equations_are_invalid_argument() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let m = Var::new(f64_tensor(&P[..4], &[2, 2]), true);
        for equation in [
            "ii->i",
            "ii->",
            "...i->...i",
            "ij,jk->ik",
            "ij->k",
            "iJ->i",
            "ij->ii",
        ] {
            let result = var_einsum(equation, &[&m], &client);
            assert!(
                matches!(result, Err(Error::InvalidArgument { .. })),
                "{equation} must be rejected"
            );
        }
    }

    #[test]
    fn no_grad_inputs_give_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&P[..4], &[2, 2]), false);
        let b = Var::new(f64_tensor(&Q[..4], &[2, 2]), false);
        let z = var_einsum("ij,jk->ik", &[&a, &b], &client).unwrap();
        assert!(!z.requires_grad());
        assert!(z.grad_fn().is_none());
    }
}
