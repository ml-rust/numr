//! Index dtype bridging for the CUDA sparse kernels.
//!
//! `CsrData`, `CscData` and `CooData` hold I64 indices on every runtime. The
//! SpMV, SpMM and merge kernels read indices as `int`. Each of those entry
//! points casts I64 indices to I32 on the GPU before a launch, and casts the
//! output indices back to I64. No index data leaves the device.

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Return `indices` as a contiguous I32 tensor for an `int*` kernel argument.
///
/// I64 input is cast on the GPU. I32 input passes through. Any other dtype
/// returns `UnsupportedDType` naming `op`.
pub(super) fn indices_to_i32(
    client: &CudaClient,
    indices: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    match indices.dtype() {
        DType::I32 => indices.contiguous(),
        DType::I64 if indices.numel() == 0 => {
            Tensor::<CudaRuntime>::zeros(&[0], DType::I32, client.device())
        }
        DType::I64 => client.cast(&indices.contiguous()?, DType::I32),
        dtype => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Return kernel output indices as the I64 tensor the sparse formats require.
pub(super) fn indices_to_i64(
    client: &CudaClient,
    indices: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    match indices.dtype() {
        DType::I64 => Ok(indices.clone()),
        _ if indices.numel() == 0 => {
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, client.device())
        }
        _ => client.cast(indices, DType::I64),
    }
}

/// Check that `dtype` is a value dtype the CUDA merge kernels instantiate:
/// F32, F64, F16, BF16, I8, I16, I32 or I64.
pub(super) fn check_merge_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Ok(()),
        _ => check_float_dtype(dtype, op),
    }
}

/// Check that `dtype` is a float dtype the CUDA sparse arithmetic kernels
/// instantiate: F32, F64, F16 or BF16.
pub(super) fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::F16 | DType::BF16 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}
