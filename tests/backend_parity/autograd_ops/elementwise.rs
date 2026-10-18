// Backend parity for the elementwise autograd ops: activations, erf, rsqrt,
// where, maximum and minimum.
//
// Each backward composes compare, scalar and where kernels on the device, so a
// divergence in any of them shows up in the gradient, not the forward.

use numr::autograd::var_ops::{
    var_elu, var_erf, var_gelu, var_gelu_erf, var_leaky_relu, var_maximum, var_minimum, var_rsqrt,
    var_where,
};

use super::harness::{OnWgpu, TOL, check_parity, graph_op, leaf};

/// Negative, zero, small and large magnitudes in one `[2, 5]` tensor.
const X: [f64; 10] = [-3.0, -1.5, -0.5, 0.0, 0.25, 0.7, 1.2, 2.0, 3.0, -2.2];

graph_op!(Gelu, |v, ctx| var_gelu(&v[0], ctx.client));
graph_op!(GeluErf, |v, ctx| var_gelu_erf(&v[0], ctx.client));
graph_op!(LeakyRelu, |v, ctx| var_leaky_relu(&v[0], 0.1, ctx.client));
graph_op!(Elu, |v, ctx| var_elu(&v[0], 1.0, ctx.client));
graph_op!(EluHalf, |v, ctx| var_elu(&v[0], 0.5, ctx.client));
graph_op!(Erf, |v, ctx| var_erf(&v[0], ctx.client));
graph_op!(Rsqrt, |v, ctx| var_rsqrt(&v[0], ctx.client));

#[test]
fn test_var_gelu_parity() {
    check_parity("var_gelu", &Gelu, &[leaf(&X, &[2, 5])], TOL, OnWgpu::Run);
}

#[test]
fn test_var_gelu_erf_parity() {
    check_parity(
        "var_gelu_erf",
        &GeluErf,
        &[leaf(&X, &[2, 5])],
        TOL,
        OnWgpu::Run,
    );
}

/// `X` holds an exact zero, so every backend must pick the same side at the kink.
#[test]
fn test_var_leaky_relu_parity() {
    check_parity(
        "var_leaky_relu",
        &LeakyRelu,
        &[leaf(&X, &[2, 5])],
        TOL,
        OnWgpu::Run,
    );
}

#[test]
fn test_var_elu_parity() {
    check_parity(
        "var_elu alpha 1",
        &Elu,
        &[leaf(&X, &[2, 5])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_elu alpha 0.5",
        &EluHalf,
        &[leaf(&X, &[2, 5])],
        TOL,
        OnWgpu::Run,
    );
}

#[test]
fn test_var_erf_parity() {
    check_parity("var_erf", &Erf, &[leaf(&X, &[2, 5])], TOL, OnWgpu::Run);
}

/// Strictly positive input. The gradient `-0.5 * x^-1.5` grows fast near zero,
/// so the smallest value stays at 0.05.
#[test]
fn test_var_rsqrt_parity() {
    let x = [0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 9.0, 16.0, 0.8];
    check_parity("var_rsqrt", &Rsqrt, &[leaf(&x, &[2, 5])], TOL, OnWgpu::Run);
}

graph_op!(WhereSameShape, |v, ctx| {
    let cond = ctx.mask(&[true, false, true, false, false, true], &[2, 3])?;
    var_where(&cond, &v[0], &v[1], ctx.client)
});

graph_op!(
    /// `x` is `[2, 1]` and `y` is `[3]`, so both gradients reduce over broadcast axes.
    WhereBroadcast,
    |v, ctx| {
        let cond = ctx.mask(&[true, false, true, false, false, true], &[2, 3])?;
        var_where(&cond, &v[0], &v[1], ctx.client)
    }
);

graph_op!(
    /// The condition is `[3]` and broadcasts across the rows of `[2, 3]` operands.
    WhereBroadcastCond,
    |v, ctx| {
        let cond = ctx.mask(&[false, true, true], &[3])?;
        var_where(&cond, &v[0], &v[1], ctx.client)
    }
);

#[test]
fn test_var_where_parity() {
    let x = [0.5, -1.0, 2.0, 1.5, -0.25, 3.0];
    let y = [-2.0, 0.75, 1.25, -0.5, 4.0, -1.5];
    check_parity(
        "var_where same shape",
        &WhereSameShape,
        &[leaf(&x, &[2, 3]), leaf(&y, &[2, 3])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_where broadcast operands",
        &WhereBroadcast,
        &[leaf(&[0.5, -1.0], &[2, 1]), leaf(&[0.75, 1.25, -2.0], &[3])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_where broadcast cond",
        &WhereBroadcastCond,
        &[leaf(&x, &[2, 3]), leaf(&y, &[2, 3])],
        TOL,
        OnWgpu::Run,
    );
}

graph_op!(Maximum, |v, ctx| var_maximum(&v[0], &v[1], ctx.client));
graph_op!(Minimum, |v, ctx| var_minimum(&v[0], &v[1], ctx.client));

/// Positions 1, 3 and 5 tie, so each side takes half the gradient there.
const A: [f64; 6] = [1.0, 2.0, -0.5, 3.0, 0.25, -1.0];
const B: [f64; 6] = [0.5, 2.0, 1.5, 3.0, -0.75, -1.0];

/// `[3]` against `[2, 3]`, with ties at `(0, 1)` and `(1, 0)`.
const ROW: [f64; 3] = [0.25, 2.0, -0.5];
const GRID: [f64; 6] = [1.0, 2.0, -1.0, 0.25, 3.0, -0.75];

#[test]
fn test_var_maximum_parity() {
    check_parity(
        "var_maximum ties",
        &Maximum,
        &[leaf(&A, &[2, 3]), leaf(&B, &[2, 3])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_maximum broadcast",
        &Maximum,
        &[leaf(&GRID, &[2, 3]), leaf(&ROW, &[3])],
        TOL,
        OnWgpu::Run,
    );
}

#[test]
fn test_var_minimum_parity() {
    check_parity(
        "var_minimum ties",
        &Minimum,
        &[leaf(&A, &[2, 3]), leaf(&B, &[2, 3])],
        TOL,
        OnWgpu::Run,
    );
    check_parity(
        "var_minimum broadcast",
        &Minimum,
        &[leaf(&ROW, &[3]), leaf(&GRID, &[2, 3])],
        TOL,
        OnWgpu::Run,
    );
}
