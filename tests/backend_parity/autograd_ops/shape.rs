// Backend parity for the shape autograd ops: stack, split, chunk, flip and pad.
//
// Their backward passes narrow, concatenate, flip and copy gradient views on
// the device. A wrong offset or stride shows up as a misplaced gradient value.

use numr::autograd::var_ops::{var_chunk, var_flip, var_mul_scalar, var_pad, var_split, var_stack};
use numr::autograd::{Var, var_cat};
use numr::error::Result;
use numr::ops::{ScalarOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};

use super::harness::{OnWgpu, TOL, check_parity, graph_op, leaf};

/// Twelve distinct values for `[2, 6]`, `[3, 4]` or `[2, 3, 2]` leaves.
const V12: [f64; 12] = [
    0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.25, -0.6, 0.8,
];

/// Scale piece `i` by `i + 1` and join the pieces back along `dim`.
///
/// Each piece then reaches the loss with its own factor, so a gradient routed
/// to the wrong piece changes the value.
fn rejoin<R, C>(pieces: &[Var<R>], dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R>,
    R::Client: ScalarOps<R>,
{
    let scaled = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| var_mul_scalar(p, (i + 1) as f64, client))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Var<R>> = scaled.iter().collect();
    var_cat(&refs, dim, client)
}

graph_op!(StackDim1, |v, ctx| {
    var_stack(&[&v[0], &v[1], &v[2]], 1, ctx.client)
});
graph_op!(StackLast, |v, ctx| {
    var_stack(&[&v[0], &v[1], &v[2]], -1, ctx.client)
});

#[test]
fn test_var_stack_parity() {
    let leaves = [
        leaf(&V12[..6], &[2, 3]),
        leaf(&V12[6..], &[2, 3]),
        leaf(&[1.0, -1.0, 0.5, -0.5, 2.0, -2.0], &[2, 3]),
    ];
    check_parity("var_stack dim 1", &StackDim1, &leaves, TOL, OnWgpu::Run);
    check_parity("var_stack dim -1", &StackLast, &leaves, TOL, OnWgpu::Run);
}

graph_op!(
    /// Uneven sizes `[1, 3, 2]` along dim 1 of `[2, 6]`.
    SplitUneven,
    |v, ctx| {
        let pieces = var_split(&v[0], &[1, 3, 2], 1)?;
        rejoin(&pieces, 1, ctx.client)
    }
);

graph_op!(
    /// Sizes `[2, 1]` along dim 0 of `[3, 4]`, a split across rows.
    SplitRows,
    |v, ctx| {
        let pieces = var_split(&v[0], &[2, 1], 0)?;
        rejoin(&pieces, 0, ctx.client)
    }
);

#[test]
fn test_var_split_parity() {
    check_parity(
        "var_split dim 1",
        &SplitUneven,
        &[leaf(&V12, &[2, 6])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_split dim 0",
        &SplitRows,
        &[leaf(&V12, &[3, 4])],
        TOL,
        OnWgpu::Run,
    );
}

graph_op!(
    /// Three chunks of dim 0 of `[5, 2]`: sizes 2, 2 and 1.
    ChunkUneven,
    |v, ctx| {
        let pieces = var_chunk(&v[0], 3, 0)?;
        rejoin(&pieces, 0, ctx.client)
    }
);

graph_op!(
    /// Two chunks of the last dim of `[2, 3, 2]`, each a strided view.
    ChunkLast,
    |v, ctx| {
        let pieces = var_chunk(&v[0], 2, -1)?;
        rejoin(&pieces, -1, ctx.client)
    }
);

#[test]
fn test_var_chunk_parity() {
    check_parity(
        "var_chunk uneven dim 0",
        &ChunkUneven,
        &[leaf(&V12[..10], &[5, 2])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_chunk dim -1",
        &ChunkLast,
        &[leaf(&V12, &[2, 3, 2])],
        TOL,
        OnWgpu::Run,
    );
}

graph_op!(FlipBoth, |v, _ctx| var_flip(&v[0], &[0, -1]));
graph_op!(FlipMiddle, |v, _ctx| var_flip(&v[0], &[1]));

#[test]
fn test_var_flip_parity() {
    check_parity(
        "var_flip dims 0, -1",
        &FlipBoth,
        &[leaf(&V12, &[3, 4])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_flip middle dim",
        &FlipMiddle,
        &[leaf(&V12, &[2, 3, 2])],
        TOL,
        OnWgpu::Run,
    );
}

graph_op!(
    /// `[2, 3]` padded to `[3, 6]`: last dim `(1, 2)`, dim 0 `(0, 1)`, value 0.5.
    PadTwoDims,
    |v, ctx| var_pad(&v[0], &[1, 2, 0, 1], 0.5, ctx.client)
);

graph_op!(
    /// `[2, 3, 2]` with only the last dim padded, `(2, 0)`, value 0.
    PadLastOnly,
    |v, ctx| var_pad(&v[0], &[2, 0], 0.0, ctx.client)
);

#[test]
fn test_var_pad_parity() {
    check_parity(
        "var_pad two dims",
        &PadTwoDims,
        &[leaf(&V12[..6], &[2, 3])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_pad last dim",
        &PadLastOnly,
        &[leaf(&V12, &[2, 3, 2])],
        TOL,
        OnWgpu::Run,
    );
}
