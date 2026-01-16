//! Comparison CUDA kernel launchers
//!
//! Provides launchers for element-wise comparison operations (eq, ne, lt, le, gt, ge)
//! on two tensors of the same shape.

use std::sync::Arc;
use cudarc::driver::safe::{CudaContext, CudaStream};

use super::loader::{kernel_names, launch_binary_kernel};
use crate::dtype::DType;
use crate::error::Result;

/// Launch a comparison operation kernel.
///
/// Performs element-wise comparison: `output[i] = a[i] op b[i]`
/// Output is the same dtype as input (typically should be bool, but follows input for now).
///
/// # Supported Operations
///
/// - `eq`: Equal (==)
/// - `ne`: Not equal (!=)
/// - `lt`: Less than (<)
/// - `le`: Less than or equal (<=)
/// - `gt`: Greater than (>)
/// - `ge`: Greater than or equal (>=)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` elements
/// - `a` and `b` must have the same dtype
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Comparison operation name (e.g., "eq", "lt")
/// * `dtype` - Data type of the input tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_compare_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> { unsafe {
    launch_binary_kernel(
        context,
        stream,
        device_index,
        kernel_names::COMPARE_MODULE,
        op,
        dtype,
        a_ptr,
        b_ptr,
        out_ptr,
        numel,
    )
}}
