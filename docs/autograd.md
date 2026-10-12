# Autograd

`numr::autograd` provides reverse-mode and forward-mode automatic differentiation over numr tensors.

## Modes

- **Reverse-mode**: `Var<R>` wraps a tracked tensor. `backward()` computes gradients into a `GradStore`.
- **Forward-mode**: `jvp()` and `jacobian_forward()` use dual numbers.
- **Second-order**: `hvp()` computes Hessian-vector products. `backward_with_graph()` computes higher-order gradients.
- **Activation checkpointing**: `checkpoint()` trades compute for memory.
- **Backward hooks**: The `BackwardHook` trait receives gradient notifications, for example to trigger
  a distributed allreduce. See [Distributed](distributed.md).

## Differentiable Operations

These functions carry gradients: `var_matmul`, `var_conv1d`, `var_conv2d`, `var_softmax`,
`var_rms_norm`, `var_layer_norm`, `var_silu`, `var_softplus`, `var_swiglu`, `var_dropout`,
`var_fused_add_rms_norm`, `var_fused_add_layer_norm`, `var_cast`, `var_narrow`, and `var_cat`.

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
