//! Small GPU-to-host scalar readbacks for CUDA control flow.
//!
//! Every read goes through `CudaRuntime::copy_from_device`. That copy is
//! ordered on the compute stream after the kernels that produced the value,
//! makes the client's context current on the calling thread, and reports
//! driver errors.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::{CudaDevice, CudaRuntime};
use crate::tensor::Tensor;

/// Read a single scalar value from a GPU tensor as `f64`.
///
/// Used for min/max values in histogram and statistics operations, and for
/// convergence checks in iterative matrix functions.
pub(crate) fn read_scalar_f64(t: &Tensor<CudaRuntime>) -> Result<f64> {
    if t.numel() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: format!(
                "read_scalar_f64 requires a single-element tensor, got {} elements",
                t.numel()
            ),
        });
    }

    let tensor = if t.is_contiguous() {
        t.clone()
    } else {
        t.contiguous()?
    };

    let mut value = [0.0f64];
    read_scalars_f64(tensor.ptr(), tensor.dtype(), tensor.device(), &mut value)?;
    Ok(value[0])
}

/// Read `out.len()` consecutive scalars of `dtype` at device address `ptr`
/// into `out`, each widened to `f64`.
///
/// Supports F32, F64, I32 and I64.
///
/// # Errors
///
/// Returns `UnsupportedDType` for any other dtype, and the copy's error
/// when the device-to-host copy fails.
pub(crate) fn read_scalars_f64(
    ptr: u64,
    dtype: DType,
    device: &CudaDevice,
    out: &mut [f64],
) -> Result<()> {
    let elem = match dtype {
        DType::F32 | DType::I32 => 4,
        DType::F64 | DType::I64 => 8,
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "read_scalars_f64",
            });
        }
    };

    let mut bytes = vec![0u8; out.len() * elem];
    CudaRuntime::copy_from_device(ptr, &mut bytes, device)?;

    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(elem)) {
        *slot = match dtype {
            DType::F32 => f64::from(f32::from_ne_bytes(four(chunk))),
            DType::I32 => f64::from(i32::from_ne_bytes(four(chunk))),
            DType::F64 => f64::from_ne_bytes(eight(chunk)),
            _ => i64::from_ne_bytes(eight(chunk)) as f64,
        };
    }
    Ok(())
}

/// The first four bytes of `chunk`.
#[inline]
fn four(chunk: &[u8]) -> [u8; 4] {
    let mut b = [0u8; 4];
    b.copy_from_slice(&chunk[..4]);
    b
}

/// The first eight bytes of `chunk`.
#[inline]
fn eight(chunk: &[u8]) -> [u8; 8] {
    let mut b = [0u8; 8];
    b.copy_from_slice(&chunk[..8]);
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cuda::CudaClient;

    fn device() -> Option<CudaDevice> {
        let device = CudaDevice::new(0);
        CudaClient::new(device.clone()).ok().map(|_| device)
    }

    #[test]
    fn reads_each_supported_dtype() {
        let Some(device) = device() else {
            return;
        };
        let f32_t = Tensor::<CudaRuntime>::from_slice(&[1.5f32], &[1], &device).expect("f32");
        let f64_t = Tensor::<CudaRuntime>::from_slice(&[-2.25f64], &[1], &device).expect("f64");
        let i32_t = Tensor::<CudaRuntime>::from_slice(&[-7i32], &[1], &device).expect("i32");
        let i64_t = Tensor::<CudaRuntime>::from_slice(&[1i64 << 40], &[1], &device).expect("i64");
        assert_eq!(read_scalar_f64(&f32_t).expect("read f32"), 1.5);
        assert_eq!(read_scalar_f64(&f64_t).expect("read f64"), -2.25);
        assert_eq!(read_scalar_f64(&i32_t).expect("read i32"), -7.0);
        assert_eq!(
            read_scalar_f64(&i64_t).expect("read i64"),
            (1i64 << 40) as f64
        );
    }

    #[test]
    fn reads_two_f32_scalars_at_their_width() {
        let Some(device) = device() else {
            return;
        };
        let t = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 3.5], &[2], &device).expect("pair");
        let mut got = [0.0f64; 2];
        read_scalars_f64(t.ptr(), DType::F32, &device, &mut got).expect("read pair");
        assert_eq!(got, [1.0, 3.5]);
    }

    #[test]
    fn rejects_multi_element_tensor() {
        let Some(device) = device() else {
            return;
        };
        let t = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).expect("pair");
        assert!(read_scalar_f64(&t).is_err());
    }
}
