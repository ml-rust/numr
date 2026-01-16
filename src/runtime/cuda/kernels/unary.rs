//! Unary operation CUDA kernel launchers
//!
//! Provides launchers for element-wise unary operations (neg, abs, sqrt, exp, etc.)
//! on a single tensor.

use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{kernel_names, launch_unary_kernel};
use crate::dtype::DType;
use crate::error::Result;

/// Launch a unary operation kernel.
///
/// Performs element-wise operation: `output[i] = op(input[i])`
///
/// # Supported Operations
///
/// - `neg`: Negation (-x)
/// - `abs`: Absolute value
/// - `sqrt`: Square root
/// - `exp`: Exponential (e^x)
/// - `log`: Natural logarithm
/// - `sin`, `cos`, `tan`: Trigonometric functions
/// - `tanh`: Hyperbolic tangent
/// - `recip`: Reciprocal (1/x)
/// - `square`: Square (x*x)
/// - `floor`, `ceil`, `round`: Rounding functions
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Operation name (e.g., "neg", "sqrt")
/// * `dtype` - Data type of the tensor
/// * `a_ptr` - Device pointer to input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_unary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_kernel(
            context,
            stream,
            device_index,
            kernel_names::UNARY_MODULE,
            op,
            dtype,
            a_ptr,
            out_ptr,
            numel,
        )
    }
}
