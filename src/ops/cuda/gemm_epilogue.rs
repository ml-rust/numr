//! CUDA implementation of GEMM epilogue operations.

use crate::error::{Error, Result};
use crate::ops::{
    GemmActivation, GemmEpilogueOps, matmul_bias_output_shape, validate_matmul_bias_dtypes,
};
use crate::runtime::cuda::kernels::{
    launch_gemm_bias_act_batched_kernel, launch_gemm_bias_act_bwd_batched_kernel,
    launch_gemm_bias_act_bwd_kernel, launch_gemm_bias_act_kernel,
    launch_gemm_bias_residual_batched_kernel, launch_gemm_bias_residual_kernel,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl GemmEpilogueOps<CudaRuntime> for CudaClient {
    fn matmul_bias_activation(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        activation: GemmActivation,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

        if bias.shape().len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("bias must be 1D tensor, got shape {:?}", bias.shape()),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let out_shape = matmul_bias_output_shape(a_shape, b_shape, bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            },
        )?;

        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product::<usize>()
            .max(1);

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let bias_contig = ensure_contiguous(bias);

        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        unsafe {
            if batch_size > 1 {
                launch_gemm_bias_act_batched_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    out.ptr(),
                    batch_size,
                    m,
                    n,
                    k,
                    activation,
                )?;
            } else {
                launch_gemm_bias_act_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    out.ptr(),
                    m,
                    n,
                    k,
                    activation,
                )?;
            }
        }

        Ok(out)
    }

    fn matmul_bias_residual(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        residual: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;
        if residual.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: residual.dtype(),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        let out_shape = matmul_bias_output_shape(a_shape, b_shape, bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            },
        )?;

        if residual.shape() != out_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: out_shape.clone(),
                got: residual.shape().to_vec(),
            });
        }

        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product::<usize>()
            .max(1);

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let bias_contig = ensure_contiguous(bias);
        let res_contig = ensure_contiguous(residual);

        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        unsafe {
            if batch_size > 1 {
                launch_gemm_bias_residual_batched_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    res_contig.ptr(),
                    out.ptr(),
                    batch_size,
                    m,
                    n,
                    k,
                )?;
            } else {
                launch_gemm_bias_residual_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    res_contig.ptr(),
                    out.ptr(),
                    m,
                    n,
                    k,
                )?;
            }
        }

        Ok(out)
    }

    fn matmul_bias_activation_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        activation: GemmActivation,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;
        if grad.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: grad.dtype(),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let batch_size: usize = a_shape
            .iter()
            .take(a_shape.len().saturating_sub(2))
            .product::<usize>()
            .max(1);

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let bias_contig = ensure_contiguous(bias);
        let grad_contig = ensure_contiguous(grad);

        let d_a = Tensor::<CudaRuntime>::empty(a_shape, dtype, &self.device);
        let d_b = Tensor::<CudaRuntime>::zeros(b_shape, dtype, &self.device);
        let d_bias = Tensor::<CudaRuntime>::zeros(&[n], dtype, &self.device);

        // Temporary buffer for grad_pre (M * N elements, reused per batch)
        let grad_pre = Tensor::<CudaRuntime>::empty(&[m, n], dtype, &self.device);

        unsafe {
            if batch_size > 1 {
                launch_gemm_bias_act_bwd_batched_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    grad_contig.ptr(),
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    grad_pre.ptr(),
                    d_a.ptr(),
                    d_b.ptr(),
                    d_bias.ptr(),
                    batch_size,
                    m,
                    n,
                    k,
                    activation,
                )?;
            } else {
                launch_gemm_bias_act_bwd_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    grad_contig.ptr(),
                    a_contig.ptr(),
                    b_contig.ptr(),
                    bias_contig.ptr(),
                    grad_pre.ptr(),
                    d_a.ptr(),
                    d_b.ptr(),
                    d_bias.ptr(),
                    m,
                    n,
                    k,
                    activation,
                )?;
            }
        }

        Ok((d_a, d_b, d_bias))
    }
}
