//! Stack, split and chunk with gradient support.
//!
//! All three compose existing differentiable ops and add no `GradFn`.
//! `var_stack` is a reshape of each input followed by `var_cat`. `var_split` and
//! `var_chunk` are lists of `var_narrow` views. Every piece receives its own
//! gradient, and a piece the loss never reads contributes zero.

use crate::autograd::var_ops::flip::var_contiguous;
use crate::autograd::{Var, var_cat, var_narrow, var_reshape};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ShapeOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};

/// Join `vars` along a new dimension inserted at `dim`.
///
/// Every input must have the same shape. `dim` accepts negative indices and
/// counts against the rank of the output, which is one more than the inputs'.
pub fn var_stack<R, C>(vars: &[&Var<R>], dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ShapeOps<R>,
{
    let first = vars.first().ok_or_else(|| Error::InvalidArgument {
        arg: "vars",
        reason: "var_stack needs at least one input".to_string(),
    })?;
    let rank = first.ndim() + 1;
    let d = resolve_dim(dim, rank)?;

    for (i, v) in vars.iter().enumerate() {
        if v.shape() != first.shape() {
            return Err(Error::InvalidArgument {
                arg: "vars",
                reason: format!(
                    "var_stack input {i} has shape {:?}, expected {:?}",
                    v.shape(),
                    first.shape()
                ),
            });
        }
    }

    let mut shape = first.shape().to_vec();
    shape.insert(d, 1);
    let expanded = vars
        .iter()
        .map(|v| var_reshape(&var_contiguous(v)?, &shape))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Var<R>> = expanded.iter().collect();
    var_cat(&refs, d as isize, client)
}

/// Split `a` along `dim` into pieces of the given `sizes`.
///
/// `sizes` must sum to the size of `dim`. Each piece is a differentiable view.
pub fn var_split<R>(a: &Var<R>, sizes: &[usize], dim: isize) -> Result<Vec<Var<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R> + TensorOps<R> + ShapeOps<R>,
{
    let d = resolve_dim(dim, a.ndim())?;
    let total: usize = sizes.iter().sum();
    if total != a.shape()[d] {
        return Err(Error::InvalidArgument {
            arg: "sizes",
            reason: format!(
                "var_split sizes {sizes:?} sum to {total}, but dimension {d} has size {}",
                a.shape()[d]
            ),
        });
    }

    let mut start = 0;
    let mut pieces = Vec::with_capacity(sizes.len());
    for &len in sizes {
        pieces.push(var_narrow(a, d as isize, start, len)?);
        start += len;
    }
    Ok(pieces)
}

/// Split `a` along `dim` into at most `chunks` pieces of equal size.
///
/// The piece size is `ceil(size / chunks)`. The last piece is smaller when the
/// size does not divide evenly, and fewer than `chunks` pieces come back when
/// the pieces run out first. `chunks` must be positive and `dim` must not be empty.
pub fn var_chunk<R>(a: &Var<R>, chunks: usize, dim: isize) -> Result<Vec<Var<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R> + TensorOps<R> + ShapeOps<R>,
{
    if chunks == 0 {
        return Err(Error::InvalidArgument {
            arg: "chunks",
            reason: "var_chunk needs at least one chunk".to_string(),
        });
    }
    let d = resolve_dim(dim, a.ndim())?;
    let size = a.shape()[d];
    if size == 0 {
        return Err(Error::InvalidArgument {
            arg: "dim",
            reason: format!("var_chunk cannot split empty dimension {d}"),
        });
    }

    let step = size.div_ceil(chunks);
    let mut sizes = vec![step; size / step];
    if !size.is_multiple_of(step) {
        sizes.push(size % step);
    }
    var_split(a, &sizes, d as isize)
}

fn resolve_dim(dim: isize, rank: usize) -> Result<usize> {
    let idx = if dim < 0 { dim + rank as isize } else { dim };
    if idx < 0 || idx >= rank as isize {
        return Err(Error::InvalidDimension { dim, ndim: rank });
    }
    Ok(idx as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::autograd::{backward, var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const A: [f64; 6] = [0.3, -1.2, 0.7, 2.1, -0.4, 1.5];
    const B: [f64; 6] = [0.9, -2.2, 1.1, 0.5, -0.6, 0.8];
    const C: [f64; 6] = [1.4, 0.2, -0.3, 0.6, -1.7, 2.5];

    #[test]
    fn forward_stack_inserts_the_new_dim() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0], &[2]), false);
        let b = Var::new(f64_tensor(&[3.0, 4.0], &[2]), false);
        let first = var_stack(&[&a, &b], 0, &client).unwrap();
        assert_eq!(first.shape(), &[2, 2]);
        assert_eq!(first.tensor().to_vec::<f64>(), vec![1.0, 2.0, 3.0, 4.0]);
        let last = var_stack(&[&a, &b], -1, &client).unwrap();
        assert_eq!(last.tensor().to_vec::<f64>(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn fd_stack_every_dim() {
        for dim in [0isize, 1, 2, -1] {
            check_grad_f64(
                "var_stack",
                move |inputs, c| var_stack(&[&inputs[0], &inputs[1], &inputs[2]], dim, c),
                &[
                    f64_tensor(&A, &[2, 3]),
                    f64_tensor(&B, &[2, 3]),
                    f64_tensor(&C, &[2, 3]),
                ],
            );
        }
    }

    #[test]
    fn stack_gradient_reaches_every_input() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 1.0], &[2]), true);
        let b = Var::new(f64_tensor(&[1.0, 1.0], &[2]), true);
        let c = Var::new(f64_tensor(&[1.0, 1.0], &[2]), true);
        let w = Var::new(
            f64_tensor(&[1.0, 2.0, 10.0, 20.0, 100.0, 200.0], &[3, 2]),
            false,
        );
        let z = var_stack(&[&a, &b, &c], 0, &client).unwrap();
        let loss = var_sum(&var_mul(&z, &w, &client).unwrap(), &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(grads.get(a.id()).unwrap().to_vec::<f64>(), vec![1.0, 2.0]);
        assert_eq!(grads.get(b.id()).unwrap().to_vec::<f64>(), vec![10.0, 20.0]);
        assert_eq!(
            grads.get(c.id()).unwrap().to_vec::<f64>(),
            vec![100.0, 200.0]
        );
    }

    #[test]
    fn stack_accepts_a_strided_input() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let m = f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let view = Var::new(m.permute(&[1, 0]).unwrap(), false);
        let z = var_stack(&[&view, &view], 0, &client).unwrap();
        assert_eq!(
            z.tensor().to_vec::<f64>(),
            vec![1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn stack_rejects_bad_input() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0], &[2]), true);
        let b = Var::new(f64_tensor(&[1.0, 2.0, 3.0], &[3]), true);
        assert!(var_stack::<CpuRuntime, _>(&[], 0, &client).is_err());
        assert!(var_stack(&[&a, &b], 0, &client).is_err());
        assert!(var_stack(&[&a, &a], 2, &client).is_err());
        assert!(var_stack(&[&a, &a], -3, &client).is_err());
    }

    #[test]
    fn split_pieces_match_and_every_piece_gets_gradient() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]), true);
        let pieces = var_split(&x, &[1, 3, 2], 0).unwrap();
        assert_eq!(pieces[1].tensor().to_vec::<f64>(), vec![2.0, 3.0, 4.0]);

        // Scale piece k by 10^k so a misrouted gradient shows up in its slot.
        let mut loss: Option<Var<CpuRuntime>> = None;
        for (k, piece) in pieces.iter().enumerate() {
            let w = Var::new(
                f64_tensor(&vec![10f64.powi(k as i32); piece.numel()], piece.shape()),
                false,
            );
            let term = var_sum(&var_mul(piece, &w, &client).unwrap(), &[], false, &client).unwrap();
            loss = Some(match loss {
                Some(acc) => crate::autograd::var_add(&acc, &term, &client).unwrap(),
                None => term,
            });
        }
        let grads = backward(&loss.unwrap(), &client).unwrap();
        assert_eq!(
            grads.get(x.id()).unwrap().to_vec::<f64>(),
            vec![1.0, 10.0, 10.0, 10.0, 100.0, 100.0]
        );
    }

    #[test]
    fn fd_split_reordered_pieces() {
        // Rejoining the pieces out of order makes every piece reach the output.
        check_grad_f64(
            "var_split",
            |inputs, c| {
                let p = var_split(&inputs[0], &[1, 2, 1], 1)?;
                var_cat(&[&p[2], &p[0], &p[1]], 1, c)
            },
            &[f64_tensor(
                &[0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2],
                &[2, 4],
            )],
        );
    }

    #[test]
    fn fd_chunk_uneven() {
        check_grad_f64(
            "var_chunk",
            |inputs, c| {
                let p = var_chunk(&inputs[0], 3, 0)?;
                var_cat(&[&p[1], &p[2], &p[0]], 0, c)
            },
            &[f64_tensor(
                &[0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.3, 0.4],
                &[5, 2],
            )],
        );
    }

    #[test]
    fn chunk_sizes_follow_ceil_rule() {
        let x = Var::new(f64_tensor(&[0.0; 5], &[5]), false);
        let sizes = |chunks| -> Vec<usize> {
            var_chunk(&x, chunks, 0)
                .unwrap()
                .iter()
                .map(|p| p.numel())
                .collect()
        };
        assert_eq!(sizes(2), vec![3, 2]);
        assert_eq!(sizes(3), vec![2, 2, 1]);
        assert_eq!(sizes(4), vec![2, 2, 1]);
        assert_eq!(sizes(9), vec![1, 1, 1, 1, 1]);
        assert_eq!(sizes(1), vec![5]);
    }

    #[test]
    fn split_and_chunk_reject_bad_input() {
        let x = Var::new(f64_tensor(&[0.0; 6], &[2, 3]), true);
        assert!(var_split(&x, &[1, 1], 1).is_err());
        assert!(var_split(&x, &[3], 2).is_err());
        assert!(var_chunk(&x, 0, 0).is_err());
        assert!(var_chunk(&x, 2, 5).is_err());
    }

    #[test]
    fn no_grad_inputs_give_detached_vars() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]), false);
        let stacked = var_stack(&[&x, &x], 0, &client).unwrap();
        assert!(!stacked.requires_grad());
        assert!(stacked.grad_fn().is_none());
        for piece in var_chunk(&x, 2, 0).unwrap() {
            assert!(!piece.requires_grad());
            assert!(piece.grad_fn().is_none());
        }
    }
}
