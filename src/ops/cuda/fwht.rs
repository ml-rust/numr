//! CUDA implementation of the Walsh-Hadamard transform.

use crate::error::Result;
use crate::ops::FwhtOps;
use crate::ops::common::validate_fwht_args;
use crate::runtime::cuda::kernels::launch_fwht;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// FwhtOps implementation for CUDA runtime.
impl FwhtOps<CudaRuntime> for CudaClient {
    fn fwht(
        &self,
        x: &Tensor<CudaRuntime>,
        block_size: usize,
        signs: Option<&Tensor<CudaRuntime>>,
    ) -> Result<Tensor<CudaRuntime>> {
        let last_dim = validate_fwht_args(x, block_size, signs)?;
        let shape = x.shape();
        let dtype = x.dtype();

        if x.numel() == 0 {
            return Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        }

        // Ensure contiguous for CUDA kernel
        let x_contig = ensure_contiguous(x)?;
        let signs_contig = signs.map(ensure_contiguous).transpose()?;
        let rows = x.numel() / last_dim;

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        unsafe {
            launch_fwht(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                x_contig.ptr(),
                out.ptr(),
                signs_contig.as_ref().map(|s| s.ptr()),
                rows,
                last_dim,
                block_size,
            )?;
        }

        Ok(out)
    }
}
