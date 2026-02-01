//! CPU implementation of normalization operations.

use crate::error::{Error, Result};
use crate::ops::NormalizationOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels,
};
use crate::tensor::Tensor;

/// NormalizationOps implementation for CPU runtime.
impl NormalizationOps<CpuRuntime> for CpuClient {
    fn rms_norm(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: weight.dtype(),
            });
        }

        // Weight must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1); // Handle 1D case

        let input_contig = ensure_contiguous(input);
        let weight_contig = ensure_contiguous(weight);
        let out = Tensor::<CpuRuntime>::empty(input_shape, dtype, &self.device);

        let input_ptr = input_contig.storage().ptr();
        let weight_ptr = weight_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::rms_norm_kernel::<T>(
                    input_ptr as *const T,
                    weight_ptr as *const T,
                    out_ptr as *mut T,
                    batch_size,
                    hidden_size,
                    eps,
                );
            }
        }, "rms_norm");

        Ok(out)
    }

    fn layer_norm(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype || bias.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if weight.dtype() != dtype {
                    weight.dtype()
                } else {
                    bias.dtype()
                },
            });
        }

        // Weight and bias must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }
        if bias.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: bias.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1); // Handle 1D case

        let input_contig = ensure_contiguous(input);
        let weight_contig = ensure_contiguous(weight);
        let bias_contig = ensure_contiguous(bias);
        let out = Tensor::<CpuRuntime>::empty(input_shape, dtype, &self.device);

        let input_ptr = input_contig.storage().ptr();
        let weight_ptr = weight_contig.storage().ptr();
        let bias_ptr = bias_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::layer_norm_kernel::<T>(
                    input_ptr as *const T,
                    weight_ptr as *const T,
                    bias_ptr as *const T,
                    out_ptr as *mut T,
                    batch_size,
                    hidden_size,
                    eps,
                );
            }
        }, "layer_norm");

        Ok(out)
    }
}
