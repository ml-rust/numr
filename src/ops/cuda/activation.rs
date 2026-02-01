//! Activation functions for CUDA runtime
use crate::error::{Error, Result};
use crate::ops::{ActivationOps, normalize_softmax_dim};
use crate::runtime::cuda::kernels::{
    launch_elu, launch_gelu, launch_leaky_relu, launch_relu, launch_sigmoid, launch_silu,
    launch_softmax, launch_softmax_dim,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl ActivationOps<CudaRuntime> for CudaClient {
    fn relu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_relu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn sigmoid(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_sigmoid(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn silu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_silu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn gelu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_gelu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn leaky_relu(
        &self,
        a: &Tensor<CudaRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_leaky_relu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
                negative_slope as f32,
            )?;
        }

        Ok(out)
    }

    fn elu(&self, a: &Tensor<CudaRuntime>, alpha: f64) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_elu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
                alpha as f32,
            )?;
        }

        Ok(out)
    }

    fn softmax(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();

        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        let shape = a.shape();
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            if dim_idx == ndim - 1 {
                // Softmax over last dimension (optimized)
                launch_softmax(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    outer_size,
                    dim_size,
                )?;
            } else {
                // Softmax over non-last dimension
                launch_softmax_dim(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    outer_size,
                    dim_size,
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }
}
