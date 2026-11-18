//! CUDA correctly rounded F64 functions through native lane kernels.

use super::interval_lanes::{cuda_operand, lanes};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{CorrectlyRoundedOps, require_correctly_rounded_contracts};
use crate::runtime::cuda::kernels::{CrKernel, launch_correctly_rounded};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl CorrectlyRoundedOps<CudaRuntime> for CudaClient {
    fn cr_exp(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        map(self, a, CrKernel::Exp, "cr_exp")
    }

    fn cr_log(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        map(self, a, CrKernel::Log, "cr_log")
    }

    fn cr_log1p(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        map(self, a, CrKernel::Log1p, "cr_log1p")
    }

    fn cr_pow_scalar(&self, a: &Tensor<CudaRuntime>, exponent: f64) -> Result<Tensor<CudaRuntime>> {
        map(self, a, CrKernel::PowScalar(exponent), "cr_pow_scalar")
    }
}

fn map(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    kernel: CrKernel,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    if dtype != DType::F64 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    if a.device().index != client.device.index {
        return Err(Error::DeviceMismatch);
    }
    require_correctly_rounded_contracts::<CudaRuntime>(op)?;
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;
    let count = a.numel();
    if count == 0 {
        return Ok(out);
    }
    let n = lanes(count, op)?;
    // A strided view is copied as raw bytes on this stream, so NaN payloads reach the
    // kernel unchanged.
    let input = cuda_operand(client, a, a.shape())?;
    // SAFETY: the input operand holds one element for step 0 or `count` contiguous
    // elements for step 1, and `out` is a fresh `count`-element allocation, both on this
    // client's device.
    unsafe {
        launch_correctly_rounded(
            &client.context,
            &client.stream,
            client.device.index,
            kernel,
            input.addr(),
            input.step,
            out.ptr(),
            n,
        )
    }?;
    Ok(out)
}
