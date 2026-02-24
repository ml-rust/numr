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

        let input_ptr = input_contig.ptr();
        let weight_ptr = weight_contig.ptr();
        let out_ptr = out.ptr();

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

        let input_ptr = input_contig.ptr();
        let weight_ptr = weight_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let out_ptr = out.ptr();

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

    fn group_norm(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        num_groups: usize,
        eps: f32,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

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

        let shape = input.shape();
        if shape.len() < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "group_norm requires at least 2D input [batch, channels, ...]".into(),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        if !channels.is_multiple_of(num_groups) {
            return Err(Error::InvalidArgument {
                arg: "num_groups",
                reason: format!("channels {channels} not divisible by num_groups {num_groups}"),
            });
        }
        let channels_per_group = channels / num_groups;
        let spatial: usize = shape[2..].iter().product::<usize>().max(1);

        if weight.shape() != [channels] || bias.shape() != [channels] {
            return Err(Error::ShapeMismatch {
                expected: vec![channels],
                got: if weight.shape() != [channels] {
                    weight.shape().to_vec()
                } else {
                    bias.shape().to_vec()
                },
            });
        }

        let input_contig = ensure_contiguous(input);
        let weight_contig = ensure_contiguous(weight);
        let bias_contig = ensure_contiguous(bias);
        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device);

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::group_norm_kernel::<T>(
                    input_contig.ptr() as *const T,
                    weight_contig.ptr() as *const T,
                    bias_contig.ptr() as *const T,
                    out.ptr() as *mut T,
                    batch,
                    channels,
                    spatial,
                    num_groups,
                    channels_per_group,
                    eps,
                );
            }
        }, "group_norm");

        Ok(out)
    }
}
