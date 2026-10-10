//! CUDA implementation of the host-to-device write.

use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::HostCopyOps;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::validate_host_write;
use crate::tensor::Tensor;

impl HostCopyOps<CudaRuntime> for CudaClient {
    fn write_host_slice<T: Element>(&self, out: &Tensor<CudaRuntime>, src: &[T]) -> Result<()> {
        validate_host_write(out, T::DTYPE, src.len(), "write_host_slice")?;
        if src.is_empty() {
            return Ok(());
        }
        let bytes: &[u8] = bytemuck::cast_slice(src);
        // Async on the compute stream, so the write is ordered against the
        // kernels that read `out` and records as a memcpy node under graph
        // capture. `src` is pageable host memory, which the driver stages
        // before returning, so the caller may drop it on return.
        let result = {
            let _permit = self.stream().enqueue_permit()?;
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    out.ptr(),
                    bytes.as_ptr() as *const std::ffi::c_void,
                    bytes.len(),
                    self.stream().raw().cu_stream(),
                )
            }
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(Error::Backend(format!(
                "write_host_slice: host-to-device copy of {} bytes failed ({result:?}); check \
                 that the destination tensor is still allocated on this device",
                bytes.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cuda::CudaDevice;
    use crate::runtime::{Runtime, RuntimeClient};

    fn setup() -> Option<CudaClient> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        Some(CudaRuntime::default_client(&CudaDevice::new(0)))
    }

    #[test]
    fn write_lands_in_the_destination_and_keeps_its_address() {
        let Some(client) = setup() else {
            return;
        };
        let out = Tensor::<CudaRuntime>::zeros(&[2, 2], DType::I32, client.device()).unwrap();
        let before = out.ptr();

        client.write_host_slice(&out, &[1i32, 2, 3, 4]).unwrap();

        assert_eq!(out.ptr(), before);
        assert_eq!(out.to_vec::<i32>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn write_carries_f32_as_well() {
        let Some(client) = setup() else {
            return;
        };
        let out = Tensor::<CudaRuntime>::zeros(&[3], DType::F32, client.device()).unwrap();

        client.write_host_slice(&out, &[1.5f32, -2.5, 3.0]).unwrap();

        assert_eq!(out.to_vec::<f32>(), vec![1.5, -2.5, 3.0]);
    }

    #[test]
    fn write_rejects_a_dtype_mismatch() {
        let Some(client) = setup() else {
            return;
        };
        let out = Tensor::<CudaRuntime>::zeros(&[4], DType::F32, client.device()).unwrap();

        assert!(matches!(
            client.write_host_slice(&out, &[1i32, 2, 3, 4]),
            Err(Error::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn write_rejects_a_length_mismatch() {
        let Some(client) = setup() else {
            return;
        };
        let out = Tensor::<CudaRuntime>::zeros(&[4], DType::I32, client.device()).unwrap();

        assert!(matches!(
            client.write_host_slice(&out, &[1i32, 2, 3]),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn write_rejects_a_non_contiguous_destination() {
        let Some(client) = setup() else {
            return;
        };
        let out = Tensor::<CudaRuntime>::zeros(&[2, 3], DType::I32, client.device()).unwrap();
        let view = out.transpose(0, 1).unwrap();
        assert!(!view.is_contiguous());

        assert!(matches!(
            client.write_host_slice(&view, &[1i32, 2, 3, 4, 5, 6]),
            Err(Error::Backend(_))
        ));
    }
}
