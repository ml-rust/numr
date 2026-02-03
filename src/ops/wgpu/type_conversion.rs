//! Type conversion operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TypeConversionOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::tensor::Tensor;

impl TypeConversionOps<WgpuRuntime> for WgpuClient {
    fn cast(&self, a: &Tensor<WgpuRuntime>, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        let src_dtype = a.dtype();

        // Same-type cast is a no-op
        if src_dtype == dtype {
            return Ok(a.clone());
        }

        // Check if both dtypes are natively supported on WebGPU
        let wgpu_supported = [DType::F32, DType::I32, DType::U32];
        let native_cast = wgpu_supported.contains(&src_dtype) && wgpu_supported.contains(&dtype);

        if native_cast {
            // Use native WGSL cast shader
            use crate::runtime::wgpu::ops::native::native_cast_op;
            native_cast_op(self, a, dtype)
        } else {
            // Fall back to CPU for unsupported dtypes (F64, F16, I8, etc.)
            use crate::dispatch_dtype;
            let cpu = crate::runtime::fallback::CpuFallbackContext::new();

            dispatch_dtype!(src_dtype, T => {
                let a_cpu: crate::tensor::Tensor<crate::runtime::cpu::CpuRuntime> =
                    cpu.tensor_from_gpu::<T, WgpuRuntime>(a);
                let result_cpu = cpu.client.cast(&a_cpu, dtype)?;

                dispatch_dtype!(dtype, U => {
                    let result_data: Vec<U> = result_cpu.to_vec();
                    return Ok(Tensor::<WgpuRuntime>::from_slice(&result_data, result_cpu.shape(), &self.device_id));
                }, "cast_output");
            }, "cast_input");
        }
    }
}
