//! Snake activation for the CUDA runtime.
//!
//! The `ActivationOps` methods in `activation.rs` forward here. Forward is one
//! elementwise launch over the `[outer, C, inner]` view; backward is one
//! elementwise launch for `d_x` plus one block-per-channel reduction launch for
//! `d_alpha` and `d_beta`.

use crate::error::{Error, Result};
use crate::ops::activation_common::{SnakeGeometry, validate_snake_beta};
use crate::runtime::cuda::kernels::{
    SnakeLaunchDims, launch_snake_beta, launch_snake_beta_dparams, launch_snake_beta_dx,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl CudaClient {
    /// Validate, allocate and launch `snake_beta` over the `[outer, C, inner]` view.
    pub(crate) fn snake_beta_cuda(
        &self,
        x: &Tensor<CudaRuntime>,
        alpha: &Tensor<CudaRuntime>,
        beta: &Tensor<CudaRuntime>,
        dim: isize,
        eps: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let geom = validate_snake_beta(x, alpha, beta, dim, eps)?;
        let dtype = x.dtype();
        let x_contig = ensure_contiguous(x)?;
        let alpha_contig = ensure_contiguous(alpha)?;
        let beta_contig = ensure_contiguous(beta)?;
        let out = Tensor::<CudaRuntime>::empty(x.shape(), dtype, &self.device)?;
        unsafe {
            launch_snake_beta(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                x_contig.ptr(),
                alpha_contig.ptr(),
                beta_contig.ptr(),
                out.ptr(),
                snake_dims(geom),
                eps,
            )?;
        }
        Ok(out)
    }

    /// `d_x` by one elementwise launch, `d_alpha`/`d_beta` by one block-per-channel launch.
    #[allow(clippy::type_complexity)]
    pub(crate) fn snake_beta_bwd_cuda(
        &self,
        grad: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        alpha: &Tensor<CudaRuntime>,
        beta: &Tensor<CudaRuntime>,
        dim: isize,
        eps: f64,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let geom = validate_snake_beta(x, alpha, beta, dim, eps)?;
        let dtype = x.dtype();
        if grad.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: grad.shape().to_vec(),
            });
        }
        if grad.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: grad.dtype(),
            });
        }
        let grad_contig = ensure_contiguous(grad)?;
        let x_contig = ensure_contiguous(x)?;
        let alpha_contig = ensure_contiguous(alpha)?;
        let beta_contig = ensure_contiguous(beta)?;
        let d_x = Tensor::<CudaRuntime>::empty(x.shape(), dtype, &self.device)?;
        let d_alpha = Tensor::<CudaRuntime>::empty(alpha.shape(), dtype, &self.device)?;
        let d_beta = Tensor::<CudaRuntime>::empty(beta.shape(), dtype, &self.device)?;
        let dims = snake_dims(geom);
        unsafe {
            launch_snake_beta_dx(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                x_contig.ptr(),
                alpha_contig.ptr(),
                beta_contig.ptr(),
                d_x.ptr(),
                dims,
                eps,
            )?;
            launch_snake_beta_dparams(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                x_contig.ptr(),
                alpha_contig.ptr(),
                beta_contig.ptr(),
                d_alpha.ptr(),
                d_beta.ptr(),
                dims,
                eps,
            )?;
        }
        Ok((d_x, d_alpha, d_beta))
    }
}

fn snake_dims(geom: SnakeGeometry) -> SnakeLaunchDims {
    SnakeLaunchDims {
        outer: geom.outer,
        channels: geom.channels,
        inner: geom.inner,
    }
}
