// Backend parity for the reducing autograd ops: logsumexp, prod and
// l2_normalize.
//
// Their backward passes broadcast a kept-dim reduction back over the input.
// `var_prod` also builds exclusive prefix and suffix products with `cumprod`,
// `narrow`, `cat` and `flip` on the device, so zeros in the input reach every
// one of those kernels. Reductions reorder sums per backend, so these cases
// use `REDUCE_TOL`.

use numr::autograd::var_ops::{var_l2_normalize, var_logsumexp, var_prod};

use super::harness::{OnWgpu, REDUCE_TOL, check_parity, graph_op, leaf};

/// `[2, 3, 4]` with one row near 50, so a naive `log(sum(exp))` overflows F32.
const LSE: [f64; 24] = [
    0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.25, -0.6, 0.8, //
    50.0, 52.0, 49.5, 51.25, -3.0, 0.0, 3.0, -0.5, 1.75, -1.25, 0.6, 2.4,
];

graph_op!(LseLastKeep, |v, ctx| var_logsumexp(
    &v[0],
    &[2],
    true,
    ctx.client
));
graph_op!(LseOuterDrop, |v, ctx| {
    var_logsumexp(&v[0], &[0, 2], false, ctx.client)
});

#[test]
fn test_var_logsumexp_parity() {
    check_parity(
        "var_logsumexp dim 2 keepdim",
        &LseLastKeep,
        &[leaf(&LSE, &[2, 3, 4])],
        REDUCE_TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_logsumexp dims 0, 2",
        &LseOuterDrop,
        &[leaf(&LSE, &[2, 3, 4])],
        REDUCE_TOL,
        OnWgpu::Run,
    );
}

graph_op!(ProdRows, |v, ctx| var_prod(&v[0], &[1], false, ctx.client));
graph_op!(ProdOuter, |v, ctx| var_prod(
    &v[0],
    &[0, 2],
    true,
    ctx.client
));
graph_op!(ProdAll, |v, ctx| var_prod(&v[0], &[], true, ctx.client));

/// `[4, 4]` rows: no zero, one zero, two zeros, a zero at the row end.
///
/// The gradient of a one-zero row is nonzero only at the zero. A two-zero row
/// has an all-zero gradient. A division-based backward gets both wrong.
const PROD_ROWS: [f64; 16] = [
    1.5, -0.5, 2.0, 0.8, //
    1.25, 0.0, -1.5, 2.0, //
    0.0, 3.0, 0.0, -0.75, //
    0.9, -1.1, 1.3, 0.0,
];

/// `[2, 3, 2]` with a zero in the reduced slice for output row 1.
const PROD_3D: [f64; 12] = [
    1.5, -0.5, 1.1, 0.0, -0.8, 1.2, //
    0.7, 1.3, -0.9, 1.4, 0.6, -1.25,
];

#[test]
fn test_var_prod_parity() {
    check_parity(
        "var_prod dim 1 with zeros",
        &ProdRows,
        &[leaf(&PROD_ROWS, &[4, 4])],
        REDUCE_TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_prod dims 0, 2 with zero",
        &ProdOuter,
        &[leaf(&PROD_3D, &[2, 3, 2])],
        REDUCE_TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_prod all dims",
        &ProdAll,
        &[leaf(&PROD_3D, &[2, 3, 2])],
        REDUCE_TOL,
        OnWgpu::Run,
    );
}

graph_op!(L2Last, |v, ctx| var_l2_normalize(
    &v[0], -1, 1e-6, ctx.client
));
graph_op!(L2First, |v, ctx| var_l2_normalize(
    &v[0], 0, 1e-6, ctx.client
));

/// `[3, 4]` where row 1 is all zero, so its norm falls below `eps` and the
/// backward takes the masked branch on that row only.
const L2_ROWS: [f64; 12] = [
    0.3, -1.2, 0.7, 2.1, //
    0.0, 0.0, 0.0, 0.0, //
    1.1, 0.25, -0.6, 0.8,
];

#[test]
fn test_var_l2_normalize_parity() {
    check_parity(
        "var_l2_normalize dim -1 zero row",
        &L2Last,
        &[leaf(&L2_ROWS, &[3, 4])],
        REDUCE_TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_l2_normalize dim 0",
        &L2First,
        &[leaf(&L2_ROWS, &[3, 4])],
        REDUCE_TOL,
        OnWgpu::Run,
    );
}
