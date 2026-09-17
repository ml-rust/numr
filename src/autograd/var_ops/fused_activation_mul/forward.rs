//! Forward entry points for fused activation-mul: `activation(a) * b`.

use super::backward::FusedActivationMulBackward;
use super::fused_kind::FusedKind;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, UnaryOps,
};
use crate::runtime::{Runtime, RuntimeClient};
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
    var_fused_activation_mul(a, b, client, FusedKind::Silu)
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
    var_fused_activation_mul(a, b, client, FusedKind::Gelu)
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
    var_fused_activation_mul(a, b, client, FusedKind::Relu)
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
    var_fused_activation_mul(a, b, client, FusedKind::Sigmoid)
}

/// Shared implementation for all fused activation-mul variants
fn var_fused_activation_mul<R, C>(
    a: &Var<R>,
    b: &Var<R>,
    client: &C,
    kind: FusedKind,
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
        FusedKind::Relu => client.relu_mul(a.tensor(), b.tensor())?,
        FusedKind::Sigmoid => client.sigmoid_mul(a.tensor(), b.tensor())?,
    };

    if a.requires_grad() || b.requires_grad() {
        // Compute activation(a) for backward (needed for d_b)
        let activation_a = match kind {
            FusedKind::Silu => client.silu(a.tensor())?,
            FusedKind::Gelu => client.gelu(a.tensor())?,
            FusedKind::Relu => client.relu(a.tensor())?,
            FusedKind::Sigmoid => client.sigmoid(a.tensor())?,
        };

        let grad_fn = FusedActivationMulBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            activation_a,
            kind,
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
