//! Forward entry points for fused activation-mul: `activation(a) * b`.

use super::backward::FusedActivationMulBackward;
use super::fused_kind::FusedKind;
use crate::algorithm::special::SpecialFunctions;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::impl_generic::activation::standard_normal_cdf;
use crate::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, UnaryOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Fused SiLU-Mul: output = silu(a) * b
pub fn var_silu_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Silu, None)
}

/// Fused GELU-Mul: output = gelu(a) * b
pub fn var_gelu_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Gelu, None)
}

/// Fused exact GELU-Mul: output = gelu_erf(a) * b
///
/// Unlike the other fused-mul variants, the backward pass needs the standard
/// normal CDF `Phi(a) = 0.5 * (1 + erf(a / sqrt(2)))` (derivative =
/// `Phi(x) + x * phi(x)`). Computing `Phi` needs `erf`, which the shared
/// [`FusedActivationMulBackward`] `GradFn` impl must NOT require — requiring it
/// forces every caller of `var_silu_mul`/`var_gelu_mul`/etc. (which never touch
/// `erf`) to also satisfy `SpecialFunctions<R>`. So `Phi` is computed once
/// here, at forward time, where the extra bound is local to this function,
/// and saved on the `GradFn` as a plain constant tensor for backward to reuse
/// — the same way `activation_a` is already saved and reused.
pub fn var_gelu_erf_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>
        + SpecialFunctions<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    // Only pay for erf when a gradient will actually be needed — mirrors the
    // requires_grad gate `var_fused_activation_mul` uses for `activation_a`.
    let saved_cdf = if a.requires_grad() || b.requires_grad() {
        Some(standard_normal_cdf(client, a.tensor())?)
    } else {
        None
    };
    var_fused_activation_mul(a, b, client, FusedKind::GeluErf, saved_cdf)
}

/// Fused ReLU-Mul: output = relu(a) * b
pub fn var_relu_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Relu, None)
}

/// Fused Sigmoid-Mul: output = sigmoid(a) * b
pub fn var_sigmoid_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Sigmoid, None)
}

/// Shared implementation for all fused activation-mul variants
///
/// `saved_cdf` is an already-computed constant tensor a specific kind's
/// backward needs beyond `activation_a` — currently only `FusedKind::GeluErf`
/// (the standard normal CDF `Phi(a)`). Every other kind passes `None`. Kept
/// as a plain `Option<Tensor<R>>` parameter, not a trait bound, so kinds that
/// don't need it never require `SpecialFunctions<R>`.
fn var_fused_activation_mul<R, C>(
    a: &Var<R>,
    b: &Var<R>,
    client: &C,
    kind: FusedKind,
    saved_cdf: Option<Tensor<R>>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    // Forward: use fused kernel
    let output = match kind {
        FusedKind::Silu => client.silu_mul(a.tensor(), b.tensor())?,
        FusedKind::Gelu => client.gelu_mul(a.tensor(), b.tensor())?,
        FusedKind::GeluErf => client.gelu_erf_mul(a.tensor(), b.tensor())?,
        FusedKind::Relu => client.relu_mul(a.tensor(), b.tensor())?,
        FusedKind::Sigmoid => client.sigmoid_mul(a.tensor(), b.tensor())?,
    };

    if a.requires_grad() || b.requires_grad() {
        // Compute activation(a) for backward (needed for d_b)
        let activation_a = match kind {
            FusedKind::Silu => client.silu(a.tensor())?,
            FusedKind::Gelu => client.gelu(a.tensor())?,
            FusedKind::GeluErf => client.gelu_erf(a.tensor())?,
            FusedKind::Relu => client.relu(a.tensor())?,
            FusedKind::Sigmoid => client.sigmoid(a.tensor())?,
        };

        let grad_fn = FusedActivationMulBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            activation_a,
            saved_cdf,
            kind,
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
