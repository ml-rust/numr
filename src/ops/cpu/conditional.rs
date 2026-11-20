//! CPU implementation of conditional operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ConditionalOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels,
};
use crate::tensor::Tensor;

/// ConditionalOps implementation for CPU runtime.
impl ConditionalOps<CpuRuntime> for CpuClient {
    fn where_cond(
        &self,
        cond: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::broadcast_shape;

        // Validate that x and y have the same dtype
        if x.dtype() != y.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: x.dtype(),
                rhs: y.dtype(),
            });
        }
        let dtype = x.dtype();
        let cond_dtype = cond.dtype();

        // Compute broadcast shape (cond, x, y) -> out
        let xy_shape =
            broadcast_shape(x.shape(), y.shape()).ok_or_else(|| Error::BroadcastError {
                lhs: x.shape().to_vec(),
                rhs: y.shape().to_vec(),
            })?;
        let out_shape =
            broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| Error::BroadcastError {
                lhs: cond.shape().to_vec(),
                rhs: xy_shape.clone(),
            })?;

        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;
        let out_ptr = out.ptr();

        // Fast path: all same shape, use simple kernel
        if cond.shape() == x.shape() && x.shape() == y.shape() {
            let cond_contig = ensure_contiguous(cond)?;
            let x_contig = ensure_contiguous(x)?;
            let y_contig = ensure_contiguous(y)?;

            let cond_ptr = cond_contig.ptr();
            let x_ptr = x_contig.ptr();
            let y_ptr = y_contig.ptr();
            let numel = x.numel();

            // Double dispatch: cond dtype and value dtype
            // For U8 condition, use optimized SIMD kernel
            if cond_dtype == DType::U8 {
                dispatch_dtype!(dtype, T => {
                    unsafe {
                        kernels::where_kernel::<T>(
                            cond_ptr as *const u8,
                            x_ptr as *const T,
                            y_ptr as *const T,
                            out_ptr as *mut T,
                            numel,
                        );
                    }
                }, "where_cond");
            } else {
                // Generic kernel for any condition dtype (non-zero = true)
                dispatch_dtype!(cond_dtype, C => {
                    dispatch_dtype!(dtype, T => {
                        unsafe {
                            kernels::where_kernel_generic::<C, T>(
                                cond_ptr as *const C,
                                x_ptr as *const T,
                                y_ptr as *const T,
                                out_ptr as *mut T,
                                numel,
                            );
                        }
                    }, "where_cond");
                }, "where_cond");
            }
        } else {
            // Broadcasting path: use strided kernel
            // Broadcast layouts of all inputs (stride 0 on broadcast dims).
            let cond_layout = cond.broadcast_layout(&out_shape)?;
            let x_layout = x.broadcast_layout(&out_shape)?;
            let y_layout = y.broadcast_layout(&out_shape)?;

            // Strided kernels apply layout offsets to these storage-base pointers.
            let cond_ptr = cond.storage().ptr();
            let x_ptr = x.storage().ptr();
            let y_ptr = y.storage().ptr();

            let cond_strides = cond_layout.strides();
            let x_strides = x_layout.strides();
            let y_strides = y_layout.strides();
            let cond_offset = cond_layout.offset();
            let x_offset = x_layout.offset();
            let y_offset = y_layout.offset();

            // For U8 condition, use optimized kernel
            if cond_dtype == DType::U8 {
                dispatch_dtype!(dtype, T => {
                    unsafe {
                        kernels::where_strided_kernel::<T>(
                            cond_ptr as *const u8,
                            x_ptr as *const T,
                            y_ptr as *const T,
                            out_ptr as *mut T,
                            &out_shape,
                            cond_strides,
                            x_strides,
                            y_strides,
                            cond_offset,
                            x_offset,
                            y_offset,
                        );
                    }
                }, "where_cond");
            } else {
                // Generic kernel for any condition dtype
                dispatch_dtype!(cond_dtype, C => {
                    dispatch_dtype!(dtype, T => {
                        unsafe {
                            kernels::where_strided_kernel_generic::<C, T>(
                                cond_ptr as *const C,
                                x_ptr as *const T,
                                y_ptr as *const T,
                                out_ptr as *mut T,
                                &out_shape,
                                cond_strides,
                                x_strides,
                                y_strides,
                                cond_offset,
                                x_offset,
                                y_offset,
                            );
                        }
                    }, "where_cond");
                }, "where_cond");
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::TypeConversionOps;
    use crate::runtime::cpu::CpuDevice;

    fn select_offset_views(mask_dtype: DType, value_dtype: DType, reverse: bool) -> Result<()> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        // Padding keeps doubled-offset reads inside storage when checking this regression.
        let x_storage = Tensor::<CpuRuntime>::from_slice(
            &[
                91.0f64, 1.0, 2.0, 3.0, 92.0, 93.0, 94.0, 4.0, 5.0, 6.0, 95.0, 96.0, 97.0, 98.0,
                99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
            ],
            &[4, 6],
            &device,
        )?;
        let x_storage = client.cast(&x_storage, value_dtype)?;
        let x = x_storage
            .narrow(0, 0, 2)?
            .narrow(1, 1, 3)?
            .transpose(0, 1)?;
        let y_storage = Tensor::<CpuRuntime>::from_slice(
            &[81.0f64, 10.0, 20.0, 30.0, 82.0, 83.0, 84.0, 85.0],
            &[8, 1],
            &device,
        )?;
        let y_storage = client.cast(&y_storage, value_dtype)?;
        let y = y_storage.narrow(0, 1, 3)?;
        let mask_storage = Tensor::<CpuRuntime>::from_slice(
            &[70.0f64, 1.0, 0.0, 71.0, 72.0, 73.0, 74.0, 75.0],
            &[8],
            &device,
        )?;
        let mask_storage = client.cast(&mask_storage, mask_dtype)?;
        let cond = mask_storage.narrow(0, 1, 2)?;
        let (cond, x, y, expected) = if reverse {
            (
                cond.flip(0)?,
                x.flip(0)?,
                y.flip(0)?,
                [30.0, 6.0, 20.0, 5.0, 10.0, 4.0],
            )
        } else {
            (cond, x, y, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        };
        assert_eq!(x.shape(), &[3, 2]);
        assert_eq!(
            x.layout().strides(),
            if reverse { &[-1, 6] } else { &[1, 6] }
        );
        assert_eq!(x.offset(), if reverse { 3 } else { 1 });
        assert_eq!(y.offset(), if reverse { 3 } else { 1 });
        assert_eq!(cond.offset(), if reverse { 2 } else { 1 });
        let output = client.where_cond(&cond, &x, &y)?;
        assert_eq!(output.shape(), &[3, 2]);
        assert_eq!(output.dtype(), value_dtype);
        let actual = if value_dtype == DType::F32 {
            output
                .try_to_vec::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        } else {
            output.try_to_vec::<f64>()?
        };
        assert_eq!(
            actual.as_slice(),
            expected.as_slice(),
            "mask={mask_dtype:?}, values={value_dtype:?}, reverse={reverse}"
        );
        Ok(())
    }

    #[test]
    fn where_cond_broadcast_applies_each_view_offset_once() -> Result<()> {
        for mask_dtype in [DType::U8, DType::F32, DType::F64] {
            for value_dtype in [DType::F32, DType::F64] {
                select_offset_views(mask_dtype, value_dtype, false)?;
            }
        }
        Ok(())
    }

    #[test]
    fn where_cond_broadcast_preserves_reversed_offset_views() -> Result<()> {
        for mask_dtype in [DType::U8, DType::F32, DType::F64] {
            for value_dtype in [DType::F32, DType::F64] {
                select_offset_views(mask_dtype, value_dtype, true)?;
            }
        }
        Ok(())
    }
}
