//! Binary operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::{BinaryOp, Kernel};
use crate::runtime::{compute_broadcast_shape, validate_binary_dtypes};
use crate::tensor::Tensor;

/// Helper for binary operations (add, sub, mul, div, pow, max, min)
pub fn binary_op_impl(
    client: &CpuClient,
    op: BinaryOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // Create output tensor
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
    let out_ptr = out.storage().ptr();

    // Check if we can use the fast path (same shapes, both contiguous)
    let same_shapes = a.shape() == b.shape() && a.shape() == out_shape.as_slice();
    let both_contiguous = a.is_contiguous() && b.is_contiguous();

    if same_shapes && both_contiguous {
        // Fast path: no broadcasting needed, use contiguous kernel
        let len = a.numel();
        let a_ptr = a.storage().ptr();
        let b_ptr = b.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::binary_op::<T>(
                    client, op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                );
            }
        }, op_name);
    } else {
        // Broadcasting path: use strided kernel
        // Broadcast both inputs to output shape (zero-copy views with stride 0 for broadcast dims)
        let a_broadcast = a.broadcast_to(&out_shape)?;
        let b_broadcast = b.broadcast_to(&out_shape)?;

        let a_ptr = a_broadcast.storage().ptr();
        let b_ptr = b_broadcast.storage().ptr();

        // Get strides from broadcast layouts
        let a_strides: Vec<isize> = a_broadcast.layout().strides().to_vec();
        let b_strides: Vec<isize> = b_broadcast.layout().strides().to_vec();
        let a_offset = a_broadcast.layout().offset();
        let b_offset = b_broadcast.layout().offset();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::binary_op_strided_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    &out_shape,
                    &a_strides,
                    &b_strides,
                    a_offset,
                    b_offset,
                );
            }
        }, op_name);
    }

    Ok(out)
}
