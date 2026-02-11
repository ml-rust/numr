//! Type conversion operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::tensor::Tensor;

impl TypeConversionOps<WgpuRuntime> for WgpuClient {
    fn cast(&self, a: &Tensor<WgpuRuntime>, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        let src_dtype = a.dtype();

        if src_dtype == dtype {
            return Ok(a.clone());
        }

        // WebGPU natively supports 32-bit types only (F32, I32, U32).
        // Casts between native types use WGSL shaders on-device.
        let wgpu_native = [DType::F32, DType::I32, DType::U32];
        let native_cast = wgpu_native.contains(&src_dtype) && wgpu_native.contains(&dtype);

        if native_cast {
            use crate::runtime::wgpu::ops::native::native_cast_op;
            return native_cast_op(self, a, dtype);
        }

        // WebGPU only supports 32-bit types. Reject non-native casts.
        Err(Error::UnsupportedDType {
            dtype: if !wgpu_native.contains(&src_dtype) {
                src_dtype
            } else {
                dtype
            },
            op: "cast (WebGPU supports F32, I32, U32 only)",
        })
    }
}
