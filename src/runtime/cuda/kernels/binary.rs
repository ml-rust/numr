//! Binary operation CUDA kernel launchers
//!
//! Provides launchers for element-wise binary operations (add, sub, mul, div, etc.)
//! on two tensors of the same shape.

use std::sync::Arc;
use cudarc::driver::safe::{CudaContext, CudaStream};

use super::loader::{kernel_names, launch_binary_kernel};
use crate::dtype::DType;
use crate::error::Result;

/// Launch a binary operation kernel.
///
/// Performs element-wise operation: `output[i] = op(a[i], b[i])`
///
/// # Supported Operations
///
/// - `add`: Element-wise addition
/// - `sub`: Element-wise subtraction
/// - `mul`: Element-wise multiplication
/// - `div`: Element-wise division
/// - `pow`: Element-wise power
/// - `max`: Element-wise maximum
/// - `min`: Element-wise minimum
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
/// * `op` - Operation name (e.g., "add", "mul")
/// * `dtype` - Data type of the tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_binary_op(
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
        kernel_names::BINARY_MODULE,
        op,
        dtype,
        a_ptr,
        b_ptr,
        out_ptr,
        numel,
    )
}}
