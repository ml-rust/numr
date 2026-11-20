//! Compare operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::CompareOp;
use crate::runtime::{compute_broadcast_shape, validate_binary_dtypes};
use crate::tensor::Tensor;

/// Helper for comparison operations (eq, ne, lt, le, gt, ge)
pub fn compare_op_impl(
    client: &CpuClient,
    op: CompareOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;
    let out_ptr = out.ptr();

    // Fast path for same shapes, both contiguous
    let same_shapes = a.shape() == b.shape() && a.shape() == out_shape.as_slice();
    let both_contiguous = a.is_contiguous() && b.is_contiguous();

    if same_shapes && both_contiguous {
        let len = a.numel();
        let a_ptr = a.ptr();
        let b_ptr = b.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::compare_op_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                );
            }
        }, op_name);
    } else {
        // Broadcasting path: use strided kernel. Broadcasting keeps each
        // input's offset, so its pointer is unchanged; only the strides change.
        let a_layout = a.broadcast_layout(&out_shape)?;
        let b_layout = b.broadcast_layout(&out_shape)?;
        let a_ptr = a.ptr();
        let b_ptr = b.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::compare_op_strided_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    &out_shape,
                    a_layout.strides(),
                    b_layout.strides(),
                    0,
                    0,
                );
            }
        }, op_name);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::ops::CompareOps;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    /// The broadcast path reads each input through its own pointer, so a view
    /// at a nonzero offset must start at its first element, not the buffer's.
    #[test]
    fn broadcast_compare_reads_offset_views_from_their_first_element() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let rows =
            Tensor::<CpuRuntime>::from_slice(&[9.0f64, 9.0, 1.0, 5.0, 3.0, 2.0], &[3, 2], &device)
                .expect("base")
                .narrow(0, 1, 2)
                .expect("narrow"); // [[1,5],[3,2]]
        let row = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 2.0, 4.0], &[3], &device)
            .expect("row")
            .narrow(0, 1, 2)
            .expect("narrow"); // [2,4]

        let lt = client.lt(&rows, &row).expect("lt");
        assert_eq!(lt.shape(), &[2, 2]);
        assert_eq!(lt.to_vec::<f64>(), vec![1.0, 0.0, 0.0, 1.0]);
    }
}
