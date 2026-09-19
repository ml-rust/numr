//! Binary operations for CUDA runtime
use crate::error::{Error, Result};
use crate::ops::BinaryOps;
use crate::runtime::cuda::kernels::{launch_fused_add_mul, launch_fused_mul_add};
use crate::runtime::cuda::ops::helpers::{native_binary_op, native_binary_op_into};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Runtime, ensure_contiguous, validate_copy_into};
use crate::tensor::Tensor;

impl BinaryOps<CudaRuntime> for CudaClient {
    fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "div")
    }

    fn pow(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "max")
    }

    fn minimum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "min")
    }

    fn atan2(
        &self,
        y: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, y, x, "atan2")
    }

    fn fused_mul_add(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        c: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype || c.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if b.dtype() != dtype {
                    b.dtype()
                } else {
                    c.dtype()
                },
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let c_contig = ensure_contiguous(c)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_fused_mul_add(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                c_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn fused_add_mul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        c: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype || c.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if b.dtype() != dtype {
                    b.dtype()
                } else {
                    c.dtype()
                },
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let c_contig = ensure_contiguous(c)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_fused_add_mul(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                c_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn add_into(
        &self,
        out: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<()> {
        native_binary_op_into(self, out, a, b, "add")
    }

    fn copy_into(&self, out: &Tensor<CudaRuntime>, src: &Tensor<CudaRuntime>) -> Result<()> {
        validate_copy_into(out, src, "copy_into")?;
        if out.numel() == 0 {
            return Ok(());
        }
        let elem_size = src.dtype().size_in_bytes();
        if !src.is_contiguous() {
            // Strided read straight into `out`; the kernel takes shape and
            // strides by value, so it captures into a graph.
            return CudaRuntime::copy_strided(
                src.storage().ptr(),
                src.offset() * elem_size,
                out.ptr(),
                src.shape(),
                src.strides(),
                elem_size,
                src.device(),
            );
        }
        let size_bytes = src.numel() * elem_size;
        // Async on the compute stream: stream-ordered after the producer of
        // `src`, and recorded as a memcpy node under graph capture.
        let result = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                out.ptr(),
                src.ptr(),
                size_bytes,
                self.stream.cu_stream(),
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(Error::Backend(format!(
                "copy_into: device-to-device copy of {size_bytes} bytes failed ({result:?})"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::RuntimeClient;
    use crate::runtime::cuda::CudaDevice;

    fn setup() -> Option<CudaClient> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        Some(CudaRuntime::default_client(&CudaDevice::new(0)))
    }

    #[test]
    fn copy_into_keeps_destination_address() {
        let Some(client) = setup() else {
            return;
        };
        let src =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], client.device())
                .unwrap();
        let out = Tensor::<CudaRuntime>::zeros(&[2, 2], DType::F32, client.device()).unwrap();
        let before = out.ptr();
        client.copy_into(&out, &src).unwrap();
        assert_eq!(out.ptr(), before);
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_into_reads_strided_source() {
        let Some(client) = setup() else {
            return;
        };
        let src = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            client.device(),
        )
        .unwrap();
        let transposed = src.transpose(0, 1).unwrap();
        let out = Tensor::<CudaRuntime>::zeros(&[3, 2], DType::F32, client.device()).unwrap();
        client.copy_into(&out, &transposed).unwrap();
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
