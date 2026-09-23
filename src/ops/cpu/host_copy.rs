//! CPU implementation of the host-to-device write.

use crate::dtype::Element;
use crate::error::Result;
use crate::ops::HostCopyOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::runtime::{Runtime, validate_host_write};
use crate::tensor::Tensor;

impl HostCopyOps<CpuRuntime> for CpuClient {
    fn write_host_slice<T: Element>(&self, out: &Tensor<CpuRuntime>, src: &[T]) -> Result<()> {
        validate_host_write(out, T::DTYPE, src.len(), "write_host_slice")?;
        if src.is_empty() {
            return Ok(());
        }
        // One memcpy into the destination's own storage; `ptr()` already
        // folds the view offset in.
        CpuRuntime::copy_to_device(bytemuck::cast_slice(src), out.ptr(), out.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::error::Error;
    use crate::runtime::RuntimeClient;
    use crate::runtime::cpu::CpuDevice;

    fn client() -> CpuClient {
        CpuRuntime::default_client(&CpuDevice::new())
    }

    #[test]
    fn write_lands_in_the_destination_and_keeps_its_address() {
        let client = client();
        let out = Tensor::<CpuRuntime>::zeros(&[2, 2], DType::I32, client.device()).unwrap();
        let before = out.ptr();

        client.write_host_slice(&out, &[1i32, 2, 3, 4]).unwrap();

        assert_eq!(out.ptr(), before);
        assert_eq!(out.to_vec::<i32>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn write_carries_f32_as_well() {
        let client = client();
        let out = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, client.device()).unwrap();

        client.write_host_slice(&out, &[1.5f32, -2.5, 3.0]).unwrap();

        assert_eq!(out.to_vec::<f32>(), vec![1.5, -2.5, 3.0]);
    }

    #[test]
    fn write_rejects_a_dtype_mismatch() {
        let client = client();
        let out = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, client.device()).unwrap();

        assert!(matches!(
            client.write_host_slice(&out, &[1i32, 2, 3, 4]),
            Err(Error::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn write_rejects_a_length_mismatch() {
        let client = client();
        let out = Tensor::<CpuRuntime>::zeros(&[4], DType::I32, client.device()).unwrap();

        assert!(matches!(
            client.write_host_slice(&out, &[1i32, 2, 3]),
            Err(Error::ShapeMismatch { .. })
        ));
        assert_eq!(out.to_vec::<i32>(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn write_rejects_a_non_contiguous_destination() {
        let client = client();
        let out = Tensor::<CpuRuntime>::zeros(&[2, 3], DType::I32, client.device()).unwrap();
        let view = out.transpose(0, 1).unwrap();
        assert!(!view.is_contiguous());

        assert!(matches!(
            client.write_host_slice(&view, &[1i32, 2, 3, 4, 5, 6]),
            Err(Error::Backend(_))
        ));
    }
}
