// Backend parity for `var_einsum`.
//
// The gradient to each operand is itself an einsum over the output gradient
// and the other operands. A label only one operand holds is summed away in the
// forward, and its gradient broadcasts back along that label. Contractions
// reorder sums per backend, so these cases use `REDUCE_TOL`.

use numr::autograd::Var;
use numr::autograd::var_ops::var_einsum;
use numr::dtype::DType;
use numr::error::Result;
use numr::runtime::Runtime;

use super::harness::{Ctx, GraphOp, Leaf, OnWgpu, ParityClient, REDUCE_TOL, check_parity, leaf};

/// One einsum equation over every leaf, in order.
struct Einsum(&'static str);

impl GraphOp for Einsum {
    fn build<R>(&self, leaves: &[Var<R>], ctx: &Ctx<'_, R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>,
    {
        let operands: Vec<&Var<R>> = leaves.iter().collect();
        var_einsum(self.0, &operands, ctx.client)
    }
}

/// Deterministic values in `[-1.5, 1.5]` for a leaf of `shape`.
fn ramp(seed: usize, shape: &[usize]) -> Leaf {
    let numel: usize = shape.iter().product();
    let data: Vec<f64> = (0..numel)
        .map(|i| (((i * 7 + seed * 3) % 13) as f64 - 6.0) * 0.25)
        .collect();
    leaf(&data, shape)
}

fn check(equation: &'static str, shapes: &[&[usize]]) {
    let leaves: Vec<Leaf> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| ramp(i + 1, s))
        .collect();
    check_parity(
        &format!("var_einsum {equation}"),
        &Einsum(equation),
        &leaves,
        REDUCE_TOL,
        OnWgpu::Run,
    );
}

#[test]
fn test_var_einsum_matmul_parity() {
    check("ij,jk->ik", &[&[3, 4], &[4, 5]]);
}

#[test]
fn test_var_einsum_batched_parity() {
    check("bij,bjk->bik", &[&[2, 3, 4], &[2, 4, 2]]);
}

/// Three operands, so each gradient contracts the other two.
#[test]
fn test_var_einsum_three_operand_parity() {
    check("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]]);
}

/// `j` is held by the first operand only, and `k` by the second only.
#[test]
fn test_var_einsum_private_labels_parity() {
    check("ij,k->ik", &[&[3, 4], &[2]]);
}

/// Implicit output: the labels that occur once, in sorted order (`ik`).
#[test]
fn test_var_einsum_implicit_parity() {
    check("ij,jk", &[&[3, 4], &[4, 2]]);
}

/// Attention-style scores: `d` contracts, `h` and `k` stay.
#[test]
fn test_var_einsum_scores_parity() {
    check("bhd,bkd->bhk", &[&[2, 3, 4], &[2, 5, 4]]);
}

#[test]
fn test_var_einsum_outer_and_reduce_parity() {
    check("i,j->ij", &[&[3], &[4]]);
    check("ij->j", &[&[3, 4]]);
}
