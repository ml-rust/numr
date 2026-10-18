// Backend parity for grad mode and activation checkpointing.
//
// `no_grad` is thread-local state, not a kernel, but the graph it leaves
// behind runs its backward on the device. `checkpoint_with_client` recomputes
// its segment during backward, on the caller's client. Both must give the
// gradients CPU gives, and the checkpointed run must equal the plain run on
// the same backend.

use numr::autograd::var_ops::{var_gelu, var_logsumexp, var_prod};
use numr::autograd::{
    Var, backward, checkpoint_with_client, enable_grad, is_grad_enabled, no_grad, var_add, var_exp,
    var_matmul, var_mul,
};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;

use super::harness::{
    Ctx, OnWgpu, ParityClient, Probe, Probed, REDUCE_TOL, TOL, assert_close, check_probe,
    weighted_loss,
};

const X: [f64; 6] = [0.5, -1.0, 1.5, -0.25, 2.0, 0.75];

/// `out = exp(x) * frozen`, where `frozen = gelu(x) * x` runs inside `no_grad`.
///
/// `frozen` is a constant to the graph, so `grad_x = exp(x) * frozen * w`.
/// A tracked `frozen` would add `exp(x) * d(frozen)/dx * w`.
/// `enable_grad` inside `no_grad` records again, so `tracked` keeps its graph.
struct NoGradBranch;

impl Probe for NoGradBranch {
    fn run<R>(&self, ctx: &Ctx<'_, R>) -> Result<Probed<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>,
    {
        let x = Var::new(ctx.f32(&X, &[2, 3])?, true);
        let (frozen, tracked) = no_grad(|| -> Result<(Var<R>, Var<R>)> {
            assert!(!is_grad_enabled(), "no_grad left recording on");
            let frozen = var_mul(&var_gelu(&x, ctx.client)?, &x, ctx.client)?;
            let tracked = enable_grad(|| var_exp(&x, ctx.client))?;
            Ok((frozen, tracked))
        })?;
        assert!(is_grad_enabled(), "no_grad did not restore recording");
        assert!(
            !frozen.requires_grad() && frozen.grad_fn().is_none(),
            "an op inside no_grad recorded a graph node"
        );
        assert!(
            tracked.requires_grad() && tracked.grad_fn().is_some(),
            "an op inside enable_grad did not record a graph node"
        );

        let out = var_mul(&tracked, &frozen, ctx.client)?;
        let grads = backward(&weighted_loss(&out, ctx)?, ctx.client)?;
        let grad_x = grads.get(x.id()).ok_or(Error::MissingGradient)?;
        Ok(vec![
            ("frozen".to_string(), frozen.tensor().contiguous()?),
            ("forward".to_string(), out.tensor().contiguous()?),
            ("grad".to_string(), grad_x.contiguous()?),
        ])
    }
}

#[test]
fn test_no_grad_untracked_parity() {
    check_probe("no_grad branch", &NoGradBranch, TOL, OnWgpu::Run);
}

/// A loss built entirely inside `no_grad` has no graph. `backward` must
/// return an `Err` on every backend, never panic.
struct NoGradLoss;

impl Probe for NoGradLoss {
    fn run<R>(&self, ctx: &Ctx<'_, R>) -> Result<Probed<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>,
    {
        let x = Var::new(ctx.f32(&X, &[2, 3])?, true);
        let loss = no_grad(|| weighted_loss(&var_gelu(&x, ctx.client)?, ctx))?;
        assert!(
            backward(&loss, ctx.client).is_err(),
            "backward on an untracked loss returned Ok"
        );
        Ok(vec![("loss".to_string(), loss.tensor().contiguous()?)])
    }
}

#[test]
fn test_no_grad_loss_backward_errors() {
    check_probe("no_grad loss", &NoGradLoss, TOL, OnWgpu::Run);
}

const CK_X: [f64; 12] = [
    0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.25, -0.6, 0.8,
];
const CK_W: [f64; 20] = [
    0.5, -0.25, 0.75, 0.1, -0.6, 0.2, 0.4, -0.3, 0.9, -0.15, //
    -0.45, 0.35, 0.05, -0.7, 0.55, 0.65, -0.1, 0.3, -0.85, 0.15,
];
const CK_B: [f64; 4] = [0.2, -0.4, 0.6, 1.1];

/// `logsumexp(gelu(x @ w), dim 1) * x + prod(x, dim 1) * b`.
///
/// `x` is `[3, 4]` and `w` is `[4, 5]`. The segment covers matmul, an
/// activation, a reduction with broadcast and a product, so the recompute
/// exercises several backward kernels.
fn segment<R>(inputs: &[Var<R>], client: &R::Client) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ParityClient<R>,
{
    let (x, w, b) = (&inputs[0], &inputs[1], &inputs[2]);
    let h = var_gelu(&var_matmul(x, w, client)?, client)?;
    let lse = var_logsumexp(&h, &[1], true, client)?;
    let p = var_prod(x, &[1], true, client)?;
    var_add(&var_mul(&lse, x, client)?, &var_mul(&p, b, client)?, client)
}

/// The same segment run plain and checkpointed on one backend.
///
/// `b` reaches the checkpoint by capture, not through `inputs`, so the capture
/// path must route its gradient too. The probe asserts both runs agree on this
/// backend, then returns the checkpointed run for CPU parity.
struct Checkpointed;

impl Probe for Checkpointed {
    fn run<R>(&self, ctx: &Ctx<'_, R>) -> Result<Probed<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>,
    {
        let leaves = || -> Result<[Var<R>; 3]> {
            Ok([
                Var::new(ctx.f32(&CK_X, &[3, 4])?, true),
                Var::new(ctx.f32(&CK_W, &[4, 5])?, true),
                Var::new(ctx.f32(&CK_B, &[1, 4])?, true),
            ])
        };

        let plain = leaves()?;
        let plain_out = segment(&plain, ctx.client)?;
        let plain_grads = backward(&weighted_loss(&plain_out, ctx)?, ctx.client)?;

        let ck = leaves()?;
        // `alias` keeps the id. `clone` mints a new one, which would hide the
        // capture from the checkpoint and cut the segment off from its inputs.
        let captured_b = ck[2].alias();
        let ck_out = checkpoint_with_client(
            move |inp: &[Var<R>], c: &R::Client| {
                segment(&[inp[0].alias(), inp[1].alias(), captured_b.alias()], c)
            },
            &[&ck[0], &ck[1]],
            ctx.client,
        )?;
        let ck_grads = backward(&weighted_loss(&ck_out, ctx)?, ctx.client)?;

        let mut probed = vec![("forward".to_string(), ck_out.tensor().contiguous()?)];
        let names = ["grad x", "grad w", "grad b (captured)"];
        for ((name, p), c) in names.iter().zip(&plain).zip(&ck) {
            let want = plain_grads
                .get(p.id())
                .ok_or_else(|| Error::Backend(format!("plain run has no gradient for {name}")))?;
            let got = ck_grads.get(c.id()).ok_or_else(|| {
                Error::Backend(format!("checkpointed run has no gradient for {name}"))
            })?;
            let want = want.contiguous()?;
            let got = got.contiguous()?;
            // The plain and the checkpointed run on one backend execute the
            // same kernels, so only `TOL` rounding can separate them.
            let label = format!("{:?} checkpoint vs plain {name}", ctx.backend);
            assert_close(&got, &want, TOL, &label);
            probed.push((name.to_string(), got));
        }
        Ok(probed)
    }
}

#[test]
fn test_checkpoint_matches_plain_parity() {
    check_probe(
        "checkpoint_with_client",
        &Checkpointed,
        REDUCE_TOL,
        OnWgpu::Run,
    );
}
