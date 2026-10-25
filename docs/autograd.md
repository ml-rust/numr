# Autograd

`numr::autograd` provides reverse-mode and forward-mode automatic differentiation over numr tensors.

## Modes

- **Reverse-mode**: `Var<R>` wraps a tracked tensor. `backward()` computes gradients into a `GradStore`.
- **Forward-mode**: `jvp()` and `jacobian_forward()` use dual numbers.
- **Second-order**: `hvp()` computes Hessian-vector products. `backward_with_graph()` computes higher-order gradients.
- **Activation checkpointing**: `checkpoint()` and `checkpoint_with_client()` trade compute for
  memory over a segment with one output. `checkpoint_many_with_client()` does the same for a
  segment with several outputs.
- **Grad mode**: `no_grad()` and `enable_grad()` set the mode for the current thread.
  `NoGradGuard` and `EnableGradGuard` hold it for a scope. `is_grad_enabled()` reads it. The mode
  is enabled by default. While it is disabled, an op returns an untracked `Var` with no `grad_fn`.
- **Backward hooks**: The `BackwardHook` trait receives gradient notifications, for example to trigger
  a distributed allreduce. See [Distributed](distributed.md).

## Frozen Operands

A `Var` created with `requires_grad == false` is frozen. `Var::graph_id()` returns
`TensorId::UNTRACKED` for it, and `TensorId::is_tracked()` returns `false` for that id. Backward
gives a frozen operand no gradient entry, so `grads.get(frozen.id())` is `None`. The other inputs
of the same op still receive their gradients.

## Differentiable Operations

These functions carry gradients. `numr::autograd` exports each one.

| Group | Functions |
| ----- | --------- |
| Arithmetic | `var_add`, `var_sub`, `var_mul`, `var_div`, `var_pow`, `var_add_scalar`, `var_sub_scalar`, `var_mul_scalar`, `var_div_scalar`, `var_pow_scalar`, `var_maximum`, `var_minimum` |
| Unary | `var_neg`, `var_abs`, `var_exp`, `var_log`, `var_sqrt`, `var_rsqrt`, `var_square`, `var_recip`, `var_sin`, `var_cos`, `var_tan`, `var_tanh`, `var_erf`, `var_clamp`, `var_cast` |
| Activations | `var_relu`, `var_leaky_relu`, `var_elu`, `var_sigmoid`, `var_silu`, `var_gelu`, `var_gelu_erf`, `var_softplus`, `var_snake_beta`, `var_softmax`, `var_log_softmax`, `var_dropout` |
| Fused activation-mul | `var_relu_mul`, `var_sigmoid_mul`, `var_silu_mul`, `var_gelu_mul`, `var_gelu_erf_mul`, `var_swiglu` |
| Reductions | `var_sum`, `var_mean`, `var_max`, `var_min`, `var_prod`, `var_std`, `var_var`, `var_logsumexp`, `var_cumsum`, `var_cumprod` |
| Normalization | `var_rms_norm`, `var_layer_norm`, `var_group_norm`, `var_l2_normalize`, `var_fused_add_rms_norm`, `var_fused_add_layer_norm` |
| Matmul and convolution | `var_matmul`, `var_matmul_wide`, `var_matmul_bias_activation`, `var_einsum`, `var_conv1d`, `var_conv2d`, `var_conv_transpose1d` |
| Linear algebra | `var_solve`, `var_inverse`, `var_det`, `var_cholesky`, `var_trace` |
| Indexing | `var_gather`, `var_scatter`, `var_scatter_add`, `var_index_select`, `var_embedding_lookup`, `var_where` |
| Shape | `var_reshape`, `var_transpose`, `var_permute`, `var_broadcast_to`, `var_narrow`, `var_cat`, `var_stack`, `var_split`, `var_chunk`, `var_flip`, `var_pad` |

## Example

This example comes from the `numr::autograd` module documentation, which runs as a doctest.

```rust
use numr::autograd::{Var, backward, var_mul};
use numr::prelude::*;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

fn main() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Create leaf variables
    let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device)?, true);
    let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device)?, true);

    // Forward: z = x * y
    let z = var_mul(&x, &y, &client)?;

    // Backward
    let grads = backward(&z, &client)?;

    // dx = y = 3.0, dy = x = 2.0
    let grad_x = grads.get(x.id()).unwrap();
    let grad_y = grads.get(y.id()).unwrap();
    Ok(())
}
```

## Related Pages

- [Operations](operations.md): the non-differentiable tensor operations.
