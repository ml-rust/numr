//! Single-scalar GPU-to-host readback for CUDA statistics operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

/// Read a single scalar value from GPU tensor using cuMemcpyDtoH_v2.
/// This is used for reading computed min/max values in histogram and statistics operations.
pub(crate) fn read_scalar_f64(t: &Tensor<CudaRuntime>) -> Result<f64> {
    // Ensure we have a single-element tensor
    if t.numel() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "read_scalar_f64 requires a single-element tensor".to_string(),
        });
    }

    let dtype = t.dtype();

    // Ensure contiguous layout
    let tensor = if t.is_contiguous() {
        t.clone()
    } else {
        t.contiguous()?
    };

    // Get GPU buffer pointer
    let ptr = tensor.ptr();

    // Allocate host memory and copy from GPU based on dtype
    let result = match dtype {
        DType::F32 => {
            let mut val: f32 = 0.0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut f32 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<f32>(),
                );
            }
            val as f64
        }
        DType::F64 => {
            let mut val: f64 = 0.0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut f64 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<f64>(),
                );
            }
            val
        }
        DType::I32 => {
            let mut val: i32 = 0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut i32 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<i32>(),
                );
            }
            val as f64
        }
        DType::I64 => {
            let mut val: i64 = 0;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    &mut val as *mut i64 as *mut std::ffi::c_void,
                    ptr,
                    std::mem::size_of::<i64>(),
                );
            }
            val as f64
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "read_scalar_f64",
            });
        }
    };

    Ok(result)
}
