//! Backward kernel for GEMM epilogue operations.
//!
//! Computes gradients for `activation(A @ B + bias)`.

use crate::dtype::Element;
use crate::ops::GemmActivation;

/// Backward pass for fused matmul + bias + activation.
///
/// Given `output = activation(A @ B + bias)`, computes:
/// - `d_a = (grad * activation'(pre_act)) @ B^T`
/// - `d_b = A^T @ (grad * activation'(pre_act))`
/// - `d_bias = sum(grad * activation'(pre_act), dim=0)`
///
/// where `pre_act = A @ B + bias`.
///
/// # Safety
/// - All pointers must be valid for the specified dimensions
/// - Output pointers must not alias with input pointers
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_activation_bwd_kernel<T: Element>(
    grad: *const T,
    a: *const T,
    b: *const T,
    bias: *const T,
    d_a: *mut T,
    d_b: *mut T,
    d_bias: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ld_grad: usize,
    activation: GemmActivation,
) {
    // Step 1: Compute pre-activation values: pre_act = A @ B + bias
    // and then compute grad_pre = grad * activation'(pre_act)
    let total = m * n;
    let mut grad_pre = vec![T::zero(); total];

    // Compute A @ B + bias into grad_pre
    for i in 0..m {
        for j in 0..n {
            grad_pre[i * n + j] = *bias.add(j);
        }
    }
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                grad_pre[i * n + j] = grad_pre[i * n + j] + a_val * *b.add(kk * ldb + j);
            }
        }
    }

    // Multiply by activation derivative
    for i in 0..total {
        let g = *grad.add((i / n) * ld_grad + (i % n));
        let pre = grad_pre[i].to_f64();
        let deriv = activation_derivative(pre, activation);
        grad_pre[i] = g * T::from_f64(deriv);
    }

    // Step 2: d_a = grad_pre @ B^T  (shape [M, K])
    // Zero d_a first
    for i in 0..m * k {
        *d_a.add(i) = T::zero();
    }
    for i in 0..m {
        for j in 0..n {
            let gp = grad_pre[i * n + j];
            for kk in 0..k {
                let d_a_ptr = d_a.add(i * k + kk);
                // B^T[j, kk] = B[kk, j] but we index B as B[kk * ldb + j]
                *d_a_ptr = *d_a_ptr + gp * *b.add(kk * ldb + j);
            }
        }
    }

    // Step 3: d_b = A^T @ grad_pre  (shape [K, N])
    // Zero d_b first
    for i in 0..k * n {
        *d_b.add(i) = T::zero();
    }
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let d_b_ptr = d_b.add(kk * n + j);
                *d_b_ptr = *d_b_ptr + a_val * grad_pre[i * n + j];
            }
        }
    }

    // Step 4: d_bias = sum(grad_pre, dim=0)  (shape [N])
    for j in 0..n {
        *d_bias.add(j) = T::zero();
    }
    for i in 0..m {
        for j in 0..n {
            let d_bias_ptr = d_bias.add(j);
            *d_bias_ptr = *d_bias_ptr + grad_pre[i * n + j];
        }
    }
}

/// Compute activation derivative at the pre-activation value.
fn activation_derivative(pre_act: f64, activation: GemmActivation) -> f64 {
    match activation {
        GemmActivation::None => 1.0,
        GemmActivation::ReLU => {
            if pre_act > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        GemmActivation::GELU => {
            let sqrt_2_over_pi: f64 = 0.7978845608028654;
            let coef: f64 = 0.044715;
            let x = pre_act;
            let inner = sqrt_2_over_pi * (x + coef * x * x * x);
            let tanh_val = inner.tanh();
            let sech2 = 1.0 - tanh_val * tanh_val;
            let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coef * x * x);
            0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner
        }
        GemmActivation::SiLU => {
            let sig = 1.0 / (1.0 + (-pre_act).exp());
            sig + pre_act * sig * (1.0 - sig)
        }
        GemmActivation::Sigmoid => {
            let sig = 1.0 / (1.0 + (-pre_act).exp());
            sig * (1.0 - sig)
        }
        GemmActivation::Tanh => {
            let t = pre_act.tanh();
            1.0 - t * t
        }
    }
}
