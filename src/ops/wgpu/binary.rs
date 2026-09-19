//! Binary operations for WebGPU runtime

use crate::error::Result;
use crate::ops::BinaryOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{
    native_binary_op, native_binary_op_into, native_copy_into, native_fused_add_mul,
    native_fused_mul_add,
};
use crate::tensor::Tensor;

impl BinaryOps<WgpuRuntime> for WgpuClient {
    fn add(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "add", a, b)
    }

    fn sub(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "sub", a, b)
    }

    fn mul(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "mul", a, b)
    }

    fn div(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "div", a, b)
    }

    fn pow(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "pow", a, b)
    }

    fn maximum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "maximum", a, b)
    }

    fn minimum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "minimum", a, b)
    }

    fn atan2(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "atan2", y, x)
    }

    fn fused_mul_add(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        c: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_mul_add(self, a, b, c)
    }

    fn fused_add_mul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        c: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_add_mul(self, a, b, c)
    }

    fn add_into(
        &self,
        out: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<()> {
        native_binary_op_into(self, "add", out, a, b)
    }

    fn copy_into(&self, out: &Tensor<WgpuRuntime>, src: &Tensor<WgpuRuntime>) -> Result<()> {
        native_copy_into(self, out, src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::wgpu::WgpuDevice;
    use crate::runtime::{Runtime, RuntimeClient};

    fn create_client() -> WgpuClient {
        WgpuRuntime::default_client(&WgpuDevice::new(0))
    }

    #[test]
    fn copy_into_keeps_destination_address() {
        let client = create_client();
        let src =
            Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], client.device())
                .unwrap();
        let out = Tensor::<WgpuRuntime>::zeros(&[2, 2], DType::F32, client.device()).unwrap();
        let before = out.ptr();
        client.copy_into(&out, &src).unwrap();
        assert_eq!(out.ptr(), before);
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_into_reads_strided_source() {
        let client = create_client();
        let src = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            client.device(),
        )
        .unwrap();
        let transposed = src.transpose(0, 1).unwrap();
        let out = Tensor::<WgpuRuntime>::zeros(&[3, 2], DType::F32, client.device()).unwrap();
        client.copy_into(&out, &transposed).unwrap();
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
