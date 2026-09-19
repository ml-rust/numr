//! Index-bounds validation shared by `index_select` and `index_put`.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{launch_fill_with_f64, launch_validate_indices};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Checks every index is in `[0, dim_size)`, or returns `IndexOutOfBounds`.
///
/// The check runs on the device and reads one count back, which is a
/// stream synchronization. Inside a CUDA graph capture a synchronization is
/// not allowed, so the check is skipped there and the kernels' own rule
/// applies: an out-of-range index reads a zero row and writes nothing (see
/// `kernels/index_ops.cuh`). A captured graph therefore never fails on an
/// index; it produces zeros for a bad one.
pub(super) fn validate_indices_or_capture(
    client: &CudaClient,
    index_contig: &Tensor<CudaRuntime>,
    index_len: usize,
    dim_size: usize,
) -> Result<()> {
    if client.is_capturing() {
        return Ok(());
    }
    let error_count_tensor = Tensor::<CudaRuntime>::empty(&[1], DType::U32, &client.device)?;
    unsafe {
        launch_fill_with_f64(
            &client.context,
            &client.stream,
            client.device.index,
            DType::U32,
            0.0,
            error_count_tensor.ptr(),
            1,
        )?;
        launch_validate_indices(
            &client.context,
            &client.stream,
            client.device.index,
            index_contig.ptr(),
            error_count_tensor.ptr(),
            index_len,
            dim_size,
        )?;
    }
    let error_count = error_count_tensor.to_vec::<u32>()[0];
    if error_count > 0 {
        return Err(Error::IndexOutOfBounds {
            index: 0,
            size: dim_size,
        });
    }
    Ok(())
}
