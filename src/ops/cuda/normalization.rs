//! Normalization operations for CUDA runtime
use crate::error::{Error, Result};
use crate::ops::NormalizationOps;
use crate::runtime::cuda::kernels::{launch_group_norm, launch_layer_norm, launch_rms_norm};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl NormalizationOps<CudaRuntime> for CudaClient {
    fn rms_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device);

        unsafe {
            launch_rms_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.ptr(),
                weight_contig.ptr(),
                out.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    fn layer_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device);

        unsafe {
            launch_layer_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.ptr(),
                weight_contig.ptr(),
                bias_contig.ptr(),
                out.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    fn group_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        num_groups: usize,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
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
                arg: "input".into(),
                reason: "group_norm requires at least 2D input [batch, channels, ...]".into(),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        if channels % num_groups != 0 {
            return Err(Error::InvalidArgument {
                arg: "num_groups".into(),
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
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        unsafe {
            launch_group_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.ptr(),
                weight_contig.ptr(),
                bias_contig.ptr(),
                out.ptr(),
                batch,
                channels,
                spatial,
                num_groups,
                channels_per_group,
                eps,
            )?;
        }

        Ok(out)
    }
}
