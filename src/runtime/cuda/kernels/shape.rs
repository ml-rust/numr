//! Shape operation CUDA kernel launchers
//!
//! Provides launchers for shape operations: cat, stack
//! split and chunk are zero-copy operations using narrow() and don't need kernels.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for shape operations
pub const SHAPE_MODULE: &str = "shape";

// ============================================================================
// Cat Copy Kernel
// ============================================================================

/// Launch cat_copy kernel to copy one source tensor into the output at the correct offset.
///
/// This kernel copies data from a source tensor to the appropriate position in the
/// concatenated output tensor. It's called once per input tensor in the cat operation.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - GPU device index
/// * `dtype` - Data type of the tensors
/// * `src_ptr` - Pointer to source tensor data (must be contiguous)
/// * `dst_ptr` - Pointer to output tensor data
/// * `outer_size` - Product of dimensions before cat dimension
/// * `src_cat_size` - Size of source tensor along cat dimension
/// * `dst_cat_size` - Size of output tensor along cat dimension (sum of all inputs)
/// * `cat_offset` - Offset in cat dimension where this source starts
/// * `inner_size` - Product of dimensions after cat dimension
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - src_ptr must point to contiguous data with total elements = outer_size * src_cat_size * inner_size
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_cat_copy(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    outer_size: usize,
    src_cat_size: usize,
    dst_cat_size: usize,
    cat_offset: usize,
    inner_size: usize,
) -> Result<()> {
    let total_elements = outer_size * src_cat_size * inner_size;
    if total_elements == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, SHAPE_MODULE)?;
        let func_name = kernel_name("cat_copy", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total_elements);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let src_cat_u32 = src_cat_size as u32;
        let dst_cat_u32 = dst_cat_size as u32;
        let cat_offset_u32 = cat_offset as u32;
        let inner_u32 = inner_size as u32;
        let total_u32 = total_elements as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&outer_u32);
        builder.arg(&src_cat_u32);
        builder.arg(&dst_cat_u32);
        builder.arg(&cat_offset_u32);
        builder.arg(&inner_u32);
        builder.arg(&total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA cat_copy kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}
