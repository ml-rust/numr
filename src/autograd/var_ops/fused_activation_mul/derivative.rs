//! Elementwise activation derivatives used by the fused activation-mul backward pass.

use super::fused_kind::FusedKind;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, UnaryOps,
};
use crate::runtime::Runtime;

/// Compute activation'(x) for the backward pass
pub(super) fn compute_activation_derivative<R, C>(
    client: &C,
    a: &crate::tensor::Tensor<R>,
    activation_a: &crate::tensor::Tensor<R>,
    kind: FusedKind,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    match kind {
        FusedKind::Silu => {
            // silu'(x) = sigmoid(x) * (1 + x - silu(x))
            let sigmoid_a = client.sigmoid(a)?;
            let one_plus_a = client.add_scalar(a, 1.0)?;
            let one_plus_a_minus_silu = client.sub(&one_plus_a, activation_a)?;
            client.mul(&sigmoid_a, &one_plus_a_minus_silu)
        }
        FusedKind::Gelu => {
            // gelu'(x) = 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner)*sqrt(2/π)*(1+3*0.044715*x²)
            // where inner = sqrt(2/π) * (x + 0.044715*x³)
            //
            // Simpler: d/dx gelu(x) = gelu(x)/x + x * pdf(x)
            // But that has x=0 issues. Use the direct form:
            //
            // Let's use: gelu(x) = 0.5*x*(1+tanh(inner))
            // gelu'(x) = 0.5*(1+tanh(inner)) + 0.5*x*(1-tanh²(inner))*inner'
            // inner' = sqrt(2/π)*(1 + 3*0.044715*x²)
            let x_sq = client.mul(a, a)?;
            let x_cu = client.mul(&x_sq, a)?;
            let coef_x_cu = client.mul_scalar(&x_cu, 0.044715)?;
            let inner_arg = client.add(a, &coef_x_cu)?;
            let sqrt_2_pi = 0.7978845608028654;
            let inner = client.mul_scalar(&inner_arg, sqrt_2_pi)?;

            // tanh(inner)
            let tanh_inner = {
                // Use exp to compute tanh: tanh(x) = (exp(2x)-1)/(exp(2x)+1)
                let two_inner = client.mul_scalar(&inner, 2.0)?;
                let exp_2 = client.exp(&two_inner)?;
                let num = client.add_scalar(&exp_2, -1.0)?;
                let den = client.add_scalar(&exp_2, 1.0)?;
                client.div(&num, &den)?
            };

            // 0.5*(1+tanh(inner))
            let one_plus_tanh = client.add_scalar(&tanh_inner, 1.0)?;
            let term1 = client.mul_scalar(&one_plus_tanh, 0.5)?;

            // sech²(inner) = 1 - tanh²(inner)
            let tanh_sq = client.mul(&tanh_inner, &tanh_inner)?;
            let sech_sq = client.add_scalar(&tanh_sq, -1.0)?;
            let sech_sq = client.neg(&sech_sq)?;

            // inner' = sqrt(2/π) * (1 + 3*0.044715*x²)
            let three_coef_x_sq = client.mul_scalar(&x_sq, 3.0 * 0.044715)?;
            let inner_deriv_unscaled = client.add_scalar(&three_coef_x_sq, 1.0)?;
            let inner_deriv = client.mul_scalar(&inner_deriv_unscaled, sqrt_2_pi)?;

            // term2 = 0.5 * x * sech²(inner) * inner'
            let x_sech_sq = client.mul(a, &sech_sq)?;
            let x_sech_sq_inner_d = client.mul(&x_sech_sq, &inner_deriv)?;
            let term2 = client.mul_scalar(&x_sech_sq_inner_d, 0.5)?;

            client.add(&term1, &term2)
        }
        FusedKind::Relu => {
            // relu'(x) = 1 if x > 0, else 0
            let zeros = crate::tensor::Tensor::<R>::zeros(a.shape(), a.dtype(), a.device())?;
            let ones = crate::tensor::Tensor::<R>::ones(a.shape(), a.dtype(), a.device())?;
            let mask = client.gt(a, &zeros)?;
            client.where_cond(&mask, &ones, &zeros)
        }
        FusedKind::Sigmoid => {
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            let sigmoid_a = client.sigmoid(a)?;
            let one_minus_sig = client.add_scalar(&sigmoid_a, -1.0)?;
            let one_minus_sig = client.neg(&one_minus_sig)?;
            client.mul(&sigmoid_a, &one_minus_sig)
        }
    }
}
