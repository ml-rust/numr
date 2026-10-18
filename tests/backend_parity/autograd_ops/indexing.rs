// Backend parity for the indexing autograd ops: gather, index_select, scatter
// and scatter_add.
//
// Their backward passes run `scatter_reduce(Sum)` and `gather` on the device
// with the forward's index tensor. Repeated indices force several gradient
// values onto one slot, which is where an atomic or ordering error shows.
// CPU and CUDA index in I64. WebGPU indexes in I32, its widest integer.

use numr::autograd::var_ops::{var_gather, var_index_select, var_scatter, var_scatter_add};
use numr::autograd::{Var, backward, backward_with_graph, var_mul};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;

use super::harness::{
    Ctx, OnWgpu, ParityClient, Probe, Probed, TOL, check_parity, check_probe, graph_op, leaf,
    weighted_loss,
};

/// `[3, 4]` input with distinct values.
const X34: [f64; 12] = [
    0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.0, -0.6, 0.8,
];

/// `[3, 3]` positions along dim 1 of `X34`. Rows 0 and 2 repeat a column.
const GATHER_DIM1: [i64; 9] = [0, 0, 2, 1, 1, 1, 3, 0, 3];

/// `[2, 4]` positions along dim 0 of `X34`. Columns 0 and 1 repeat a row.
const GATHER_DIM0: [i64; 8] = [2, 0, 2, 2, 0, 0, 1, 2];

graph_op!(GatherDim1, |v, ctx| {
    let index = ctx.index(&GATHER_DIM1, &[3, 3])?;
    var_gather(&v[0], 1, &index, ctx.client)
});

graph_op!(GatherDim0, |v, ctx| {
    let index = ctx.index(&GATHER_DIM0, &[2, 4])?;
    var_gather(&v[0], 0, &index, ctx.client)
});

#[test]
fn test_var_gather_duplicate_index_parity() {
    check_parity(
        "var_gather dim 1",
        &GatherDim1,
        &[leaf(&X34, &[3, 4])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_gather dim 0",
        &GatherDim0,
        &[leaf(&X34, &[3, 4])],
        TOL,
        OnWgpu::Run,
    );
}

/// `loss = sum(gather(x)^2 * w)`, then `loss2 = sum(grad_x * w)`.
///
/// `grad_x` is a `scatter_add` of `2 * gather(x) * w`, built as a graph. Its
/// own gradient runs a gather of the scatter's upstream gradient, so the
/// second pass covers the backward of the backward on the device.
struct GatherDoubleBackward;

impl Probe for GatherDoubleBackward {
    fn run<R>(&self, ctx: &Ctx<'_, R>) -> Result<Probed<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>,
    {
        let x = Var::new(ctx.f32(&X34, &[3, 4])?, true);
        let index = ctx.index(&GATHER_DIM1, &[3, 3])?;
        let g = var_gather(&x, 1, &index, ctx.client)?;
        let sq = var_mul(&g, &g, ctx.client)?;
        let loss = weighted_loss(&sq, ctx)?;
        let first = backward_with_graph(&loss, ctx.client)?;
        let grad_x = first.get_var(x.id()).ok_or(Error::MissingGradient)?.clone();
        let loss2 = weighted_loss(&grad_x, ctx)?;
        let second = backward(&loss2, ctx.client)?;
        let grad2_x = second.get(x.id()).ok_or(Error::MissingGradient)?;
        Ok(vec![
            ("forward".to_string(), sq.tensor().contiguous()?),
            ("grad".to_string(), grad_x.tensor().contiguous()?),
            ("grad of grad".to_string(), grad2_x.contiguous()?),
        ])
    }
}

#[test]
fn test_var_gather_double_backward_parity() {
    check_probe(
        "var_gather double backward",
        &GatherDoubleBackward,
        TOL,
        OnWgpu::Run,
    );
}

graph_op!(
    /// Rows 0 and 2 twice each, row 1 once, in the backend's widest index dtype.
    IndexSelectDim0,
    |v, ctx| {
        let index = ctx.index(&[0, 2, 0, 1], &[4])?;
        var_index_select(&v[0], 0, &index, ctx.client)
    }
);

graph_op!(
    /// Column 3 three times, in I32. The op widens it to I64 for the backward.
    IndexSelectDim1I32,
    |v, ctx| {
        let index = ctx.index_i32(&[3, 1, 3, 0, 3], &[5])?;
        var_index_select(&v[0], 1, &index, ctx.client)
    }
);

/// `var_index_select` widens its index to I64 for the backward. WebGPU has no
/// I64 type, so that widening can come back as a clean unsupported `Err` there.
/// Expected-unsupported on WebGPU: `var_index_select`.
#[test]
fn test_var_index_select_duplicate_index_parity() {
    check_parity(
        "var_index_select dim 0",
        &IndexSelectDim0,
        &[leaf(&X34, &[3, 4])],
        TOL,
        OnWgpu::AllowUnsupported,
    );
    check_parity(
        "var_index_select dim 1 I32 index",
        &IndexSelectDim1I32,
        &[leaf(&X34, &[3, 4])],
        TOL,
        OnWgpu::AllowUnsupported,
    );
}

graph_op!(
    /// `[3, 4]` target, `[2, 4]` source along dim 0. Each column names two
    /// distinct rows, because a repeated slot has no defined winner.
    ScatterDim0,
    |v, ctx| {
        let index = ctx.index(&[0, 2, 1, 2, 1, 0, 2, 0], &[2, 4])?;
        var_scatter(&v[0], 0, &index, &v[1], ctx.client)
    }
);

graph_op!(
    /// `[2, 5]` target, `[2, 3]` source along dim 1, unique per row.
    ScatterDim1,
    |v, ctx| {
        let index = ctx.index(&[4, 0, 2, 1, 3, 0], &[2, 3])?;
        var_scatter(&v[0], 1, &index, &v[1], ctx.client)
    }
);

#[test]
fn test_var_scatter_parity() {
    check_parity(
        "var_scatter dim 0",
        &ScatterDim0,
        &[
            leaf(&X34, &[3, 4]),
            leaf(&[1.5, -0.5, 2.5, 0.25, -1.75, 0.6, 3.0, -2.0], &[2, 4]),
        ],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_scatter dim 1",
        &ScatterDim1,
        &[
            leaf(&X34[..10], &[2, 5]),
            leaf(&[1.5, -0.5, 2.5, 0.25, -1.75, 0.6], &[2, 3]),
        ],
        TOL,
        OnWgpu::Run,
    );
}

graph_op!(
    /// `[2, 4]` target, `[2, 5]` source along dim 1. Row 0 hits column 0 three
    /// times, row 1 hits column 1 three times.
    ScatterAddDim1,
    |v, ctx| {
        let index = ctx.index(&[0, 3, 0, 0, 2, 1, 1, 3, 1, 0], &[2, 5])?;
        var_scatter_add(&v[0], 1, &index, &v[1], ctx.client)
    }
);

graph_op!(
    /// `[3, 2]` target, `[4, 2]` source along dim 0. Slots `(0, 0)`, `(2, 0)`
    /// and `(2, 1)` each collect two values.
    ScatterAddDim0,
    |v, ctx| {
        let index = ctx.index(&[0, 2, 2, 2, 0, 1, 2, 0], &[4, 2])?;
        var_scatter_add(&v[0], 0, &index, &v[1], ctx.client)
    }
);

/// The forward sums duplicate slots on the device in no fixed order. `TOL`
/// absorbs that reordering. The gradients are a gather and a copy, so they
/// carry no ordering error.
#[test]
fn test_var_scatter_add_duplicate_index_parity() {
    check_parity(
        "var_scatter_add dim 1",
        &ScatterAddDim1,
        &[
            leaf(&[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8], &[2, 4]),
            leaf(
                &[1.1, -0.37, 2.3, 0.71, -1.9, 0.13, 3.7, -2.9, 0.47, 1.3],
                &[2, 5],
            ),
        ],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_scatter_add dim 0",
        &ScatterAddDim0,
        &[
            leaf(&[0.5, -1.5, 2.5, 0.1, -0.3, 0.9], &[3, 2]),
            leaf(&[1.1, -0.37, 2.3, 0.71, -1.9, 0.13, 3.7, -2.9], &[4, 2]),
        ],
        TOL,
        OnWgpu::Run,
    );
}
