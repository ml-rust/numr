//! Indexing operations (gather)

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::IndexingOps;
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Gather along a dimension: z = gather(a, dim, index)
pub fn var_gather<R, C>(
    a: &Var<R>,
    dim: usize,
    index: &crate::tensor::Tensor<R>,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + IndexingOps<R>,
    R::Client: IndexingOps<R>,
{
    let output = client.gather(a.tensor(), dim, index)?;

    if a.requires_grad() {
        let grad_fn =
            GatherBackward::<R>::new(a.id(), a.shape(), dim, index.clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_gather_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Input: 2x3 matrix
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device),
            true,
        );

        // Gather along dim=1 with indices [[0, 2], [1, 0]]
        let index = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1, 0], &[2, 2], &device);
        let z = var_gather(&x, 1, &index, &client).unwrap();

        // z = [[1, 3], [5, 4]]
        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![1.0, 3.0, 5.0, 4.0]);

        let loss = crate::autograd::var_ops::var_sum(&z, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        // Grad scatters 1s back: x[0,0] += 1, x[0,2] += 1, x[1,1] += 1, x[1,0] += 1
        // So grad = [[1, 0, 1], [1, 1, 0]]
        assert_eq!(grad_x, vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    }
}
