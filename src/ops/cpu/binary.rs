//! CPU implementation of binary operations.

use crate::error::Result;
use crate::ops::BinaryOps;
use crate::runtime::Runtime;
use crate::runtime::cpu::strided_copy_into::copy_strided_into_impl;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{
        BinaryOp, binary_op_impl, binary_op_into_impl, fused_add_mul_impl, fused_mul_add_impl,
    },
};
use crate::runtime::{CopyIntoPlan, validate_copy_into};
use crate::tensor::Tensor;

/// BinaryOps implementation for CPU runtime.
impl BinaryOps<CpuRuntime> for CpuClient {
    fn add(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Add, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Sub, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Mul, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Div, a, b, "div")
    }

    fn pow(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Pow, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Max, a, b, "maximum")
    }

    fn minimum(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Min, a, b, "minimum")
    }

    fn atan2(&self, y: &Tensor<CpuRuntime>, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Atan2, y, x, "atan2")
    }

    fn fused_mul_add(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        c: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_mul_add_impl(self, a, b, c)
    }

    fn fused_add_mul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        c: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_add_mul_impl(self, a, b, c)
    }

    fn add_into(
        &self,
        out: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<()> {
        binary_op_into_impl(self, BinaryOp::Add, out, a, b, "add_into")
    }

    fn copy_into(&self, out: &Tensor<CpuRuntime>, src: &Tensor<CpuRuntime>) -> Result<()> {
        if validate_copy_into(out, src, "copy_into")? == CopyIntoPlan::Skip {
            return Ok(());
        }
        let elem_size = src.dtype().size_in_bytes();
        if !out.is_contiguous() {
            // SAFETY: both views lie inside their allocations because the
            // tensors hold valid layouts, `validate_copy_into` matched the
            // shapes, and it rejected overlapping or repeating destinations.
            unsafe {
                copy_strided_into_impl(
                    src.ptr() as *const u8,
                    src.strides(),
                    out.ptr() as *mut u8,
                    out.strides(),
                    src.shape(),
                    elem_size,
                );
            }
            return Ok(());
        }
        if src.is_contiguous() {
            // One memcpy of the viewed bytes; `ptr()` already folds the offset in.
            return CpuRuntime::copy_within_device(
                src.ptr(),
                out.ptr(),
                src.numel() * elem_size,
                src.device(),
            );
        }
        CpuRuntime::copy_strided(
            src.storage().ptr(),
            src.offset() * elem_size,
            out.ptr(),
            src.shape(),
            src.strides(),
            elem_size,
            src.device(),
        )
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
    fn copy_into_keeps_destination_address() {
        let client = client();
        let src =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], client.device())
                .unwrap();
        let out = Tensor::<CpuRuntime>::zeros(&[2, 2], DType::F32, client.device()).unwrap();
        let before = out.ptr();
        client.copy_into(&out, &src).unwrap();
        assert_eq!(out.ptr(), before);
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_into_reads_strided_source() {
        let client = client();
        let src = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            client.device(),
        )
        .unwrap();
        let transposed = src.transpose(0, 1).unwrap();
        assert!(!transposed.is_contiguous());
        let out = Tensor::<CpuRuntime>::zeros(&[3, 2], DType::F32, client.device()).unwrap();
        client.copy_into(&out, &transposed).unwrap();
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn copy_into_rejects_shape_and_dtype_mismatch() {
        let client = client();
        let src = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], client.device())
            .unwrap();

        let wrong_shape =
            Tensor::<CpuRuntime>::zeros(&[2, 2], DType::F32, client.device()).unwrap();
        assert!(matches!(
            client.copy_into(&wrong_shape, &src),
            Err(Error::ShapeMismatch { .. })
        ));

        let wrong_dtype = Tensor::<CpuRuntime>::zeros(&[4], DType::F64, client.device()).unwrap();
        assert!(matches!(
            client.copy_into(&wrong_dtype, &src),
            Err(Error::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn copy_into_writes_a_channel_slice_and_keeps_the_rest() {
        let client = client();
        let buf = Tensor::<CpuRuntime>::from_slice(&[9.0f32; 8], &[2, 4], client.device()).unwrap();
        let slice = buf.narrow(-1, 1, 2).unwrap();
        let src =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], client.device())
                .unwrap();
        client.copy_into(&slice, &src).unwrap();
        assert_eq!(
            buf.to_vec::<f32>(),
            vec![9.0, 1.0, 2.0, 9.0, 9.0, 3.0, 4.0, 9.0]
        );
    }

    #[test]
    fn copy_into_rejects_overlapping_views_of_one_buffer() {
        let client = client();
        let buf = Tensor::<CpuRuntime>::zeros(&[2, 4], DType::F32, client.device()).unwrap();
        let out = buf.narrow(-1, 0, 2).unwrap();
        let src = buf.narrow(-1, 2, 2).unwrap();
        assert!(matches!(
            client.copy_into(&out, &src),
            Err(Error::InvalidArgument { arg: "out", .. })
        ));
    }

    #[test]
    fn copy_into_identical_view_is_a_no_op() {
        let client = client();
        let buf =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], client.device())
                .unwrap();
        let view = buf.narrow(-1, 0, 1).unwrap();
        client.copy_into(&view, &view).unwrap();
        assert_eq!(buf.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
