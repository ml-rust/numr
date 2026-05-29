//! Binary operation CUDA kernel launchers
//!
//! Provides launchers for element-wise binary operations (add, sub, mul, div, etc.)
//! on two tensors of the same shape.
//!
//! Also supports broadcasting operations using strided access patterns.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_binary_kernel, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::CudaDevice;

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
) -> Result<()> {
    unsafe {
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
    }
}

/// Launch a logical_and kernel.
///
/// Performs element-wise logical AND: `output[i] = a[i] && b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_and_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_and_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_and kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch a logical_or kernel.
///
/// Performs element-wise logical OR: `output[i] = a[i] || b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_or_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_or_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_or kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch a logical_xor kernel.
///
/// Performs element-wise logical XOR: `output[i] = a[i] ^ b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_xor_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_xor_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_xor kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Compute broadcast strides for a tensor shape relative to the output shape.
///
/// For each dimension in the output shape:
/// - If the input dimension matches, use the original stride
/// - If the input dimension is 1 (broadcast), use stride 0
/// - If the input doesn't have this dimension (prepended), use stride 0
pub fn compute_broadcast_strides(input_shape: &[usize], output_shape: &[usize]) -> Vec<u32> {
    let mut strides = vec![0u32; output_shape.len()];
    let input_ndim = input_shape.len();
    let output_ndim = output_shape.len();

    // Compute input strides (row-major)
    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim.saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Map input dimensions to output dimensions (right-aligned)
    let offset = output_ndim - input_ndim;
    for i in 0..output_ndim {
        if i < offset {
            // Dimension doesn't exist in input, broadcast with stride 0
            strides[i] = 0;
        } else {
            let input_idx = i - offset;
            if input_shape[input_idx] == 1 {
                // Broadcasting dimension, stride 0
                strides[i] = 0;
            } else {
                // Normal dimension, use input stride
                strides[i] = input_strides[input_idx] as u32;
            }
        }
    }

    strides
}

/// Maximum number of dimensions supported by the inline broadcast kernel.
///
/// Must match `MAX_BROADCAST_DIMS` in `binary.cu`.
pub const MAX_BROADCAST_DIMS: usize = 8;

/// Compute magic-number fast-division constants for divisor `d`.
///
/// Returns `(magic, shift)` encoding. The CUDA kernel must use:
///   if (magic == 0) { q = remaining >> shift; }   // d==1 (shift=0) or power-of-2 (shift=k)
///   else            { q = __umulhi(remaining, magic) >> shift; }  // general case
/// Then: coord = remaining - q * shape[d]; remaining = q;
///
/// - d == 0: (0, 0) — unused dim, kernel skips via ndim guard
/// - d == 1: (0, 0) — q = remaining >> 0 = remaining; coord = remaining - remaining = 0 ✓
/// - d == 2^k: (0, k) — q = remaining >> k (exact); coord = remaining - q*d ✓
/// - d general: __umulhi(x, magic) >> shift == floor(x/d) for all x in [0, 2^32) ✓
pub fn compute_magic_divisor(d: u32) -> (u32, u32) {
    if d <= 1 {
        // d==0: unused sentinel. d==1: q = remaining >> 0 = remaining; coord = 0.
        return (0u32, 0u32);
    }
    if d.is_power_of_two() {
        let shift = d.trailing_zeros();
        return (0u32, shift);
    }
    // General case d >= 3, not power-of-2:
    // magic = ceil(2^(32+p) / d), shift = p = floor(log2(d))
    // Guarantees: __umulhi(x, magic) >> p == floor(x/d) for all x in [0, 2^32).
    let p = 31u32 - d.leading_zeros();
    let numerator: u64 = 1u64 << (32 + p);
    let magic_full = (numerator + (d as u64) - 1) / (d as u64);
    // For non-power-of-2 d>=3, magic_full always fits in u32.
    debug_assert!(magic_full <= 0xFFFF_FFFFu64, "magic overflow for d={d}");
    (magic_full as u32, p)
}

/// Check whether `a` and `b` satisfy the fast trailing-broadcast preconditions:
/// - `a` must be contiguous with the same shape as `out_shape` (a_strides == natural strides)
/// - `b` must be a contiguous trailing-broadcast of `out_shape`: all leading dims of `b`
///   that differ from `out_shape` must be 1, and the remaining trailing dims must match.
///   The b_numel (product of b's non-broadcast dims) must be a contiguous suffix of out_shape.
///
/// Returns `Some(b_numel)` if the fast path applies, `None` otherwise.
pub fn detect_fast_trailing_broadcast(
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Option<usize> {
    // a must exactly match out_shape (no broadcasting on a side)
    if a_shape != out_shape {
        return None;
    }

    // b must be a trailing suffix of out_shape.
    // Aligned right: b_shape right-pads with 1s if shorter.
    // For each position, b must either be 1 (broadcast) or equal to out.
    // The non-1 dimensions of b must form a contiguous SUFFIX of out_shape.
    let ndim = out_shape.len();
    let b_ndim = b_shape.len();
    let offset = ndim.saturating_sub(b_ndim);

    // Find where b's non-trivial (non-1) dimensions start
    let mut b_start = b_ndim; // index in b_shape where first non-1 dim is
    for i in 0..b_ndim {
        if b_shape[i] != 1 {
            b_start = i;
            break;
        }
    }

    // All dims in b from b_start onward must match out_shape
    for i in b_start..b_ndim {
        let out_i = offset + i;
        if b_shape[i] != out_shape[out_i] {
            return None;
        }
    }

    // All dims in b before b_start must be 1 (already guaranteed by construction)
    // and all corresponding out dims before offset+b_start must be non-trivial
    // (but that's fine, a covers them linearly).

    // b_numel = product of b's non-1 suffix
    let b_numel: usize = b_shape[b_start..].iter().product();
    if b_numel == 0 {
        return None;
    }

    Some(b_numel)
}

/// Launch a broadcast binary operation kernel.
///
/// Performs element-wise operation with broadcasting:
/// `output[i] = op(a[broadcast_idx], b[broadcast_idx])`
///
/// # CUDA Graph Compatibility
///
/// This function uses the `*_broadcast_*_inline` kernel variants that accept
/// strides and shape as individual scalar u32 arguments baked into the
/// kernel-parameter block.  Unlike the pointer-based variants, the inline
/// kernels do NOT trigger H2D memcpy nodes during CUDA graph capture, so the
/// graph's kernel nodes never contain stale host-side pointers.
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
/// - `out_shape.len()` must be ≤ `MAX_BROADCAST_DIMS` (= 8)
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
/// * `a_shape` - Shape of tensor a
/// * `b_shape` - Shape of tensor b
/// * `out_shape` - Shape of output tensor (broadcast result)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_broadcast_binary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    _device: &CudaDevice,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Result<()> {
    let numel: usize = out_shape.iter().product();
    if numel == 0 {
        return Ok(());
    }

    let ndim = out_shape.len();
    if ndim > MAX_BROADCAST_DIMS {
        return Err(Error::Internal(format!(
            "launch_broadcast_binary_op: ndim={ndim} exceeds MAX_BROADCAST_DIMS={MAX_BROADCAST_DIMS}"
        )));
    }

    let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
    let dtype_str = kernel_name("", dtype).trim_start_matches('_').to_owned();
    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    // ----------------------------------------------------------------
    // FAST PATH: contiguous trailing-broadcast
    //
    // When a is contiguous and has the same shape as out, and b is a
    // contiguous tensor that just repeats along the leading dimensions
    // (b_index = idx % b_numel), we dispatch a specialized 3-arg kernel
    // that avoids multi-dim coordinate decomposition entirely.
    // ----------------------------------------------------------------
    if let Some(b_numel) = detect_fast_trailing_broadcast(a_shape, b_shape, out_shape) {
        let func_name = format!("{}_broadcast_fast_trailing_{}", op, dtype_str);
        if let Ok(func) = get_kernel_function(&module, &func_name) {
            let (b_magic, b_shift) = compute_magic_divisor(b_numel as u32);
            let b_numel_u32 = b_numel as u32;
            unsafe {
                let mut builder = stream.launch_builder(&func);
                builder.arg(&a_ptr);
                builder.arg(&b_ptr);
                builder.arg(&out_ptr);
                builder.arg(&b_magic);
                builder.arg(&b_shift);
                builder.arg(&b_numel_u32);
                builder.arg(&n);
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!(
                        "CUDA broadcast fast-trailing kernel '{}' launch failed: {:?}",
                        func_name, e
                    ))
                })?;
            }
            return Ok(());
        }
        // If the fast-trailing kernel is missing for some reason, fall through to general path.
    }

    // ----------------------------------------------------------------
    // GENERAL PATH: magic-number inline broadcast
    //
    // Compute broadcast strides and magic-divisor constants for each
    // output dimension. Pass all 40 scalar args inline (CUDA-graph safe).
    // ----------------------------------------------------------------

    // Compute broadcast strides.
    let a_strides_vec = compute_broadcast_strides(a_shape, out_shape);
    let b_strides_vec = compute_broadcast_strides(b_shape, out_shape);
    let shape_vec: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();

    // Pack into fixed-size arrays (zero-padded to MAX_BROADCAST_DIMS).
    let mut a_strides = [0u32; MAX_BROADCAST_DIMS];
    let mut b_strides = [0u32; MAX_BROADCAST_DIMS];
    let mut shape = [0u32; MAX_BROADCAST_DIMS];
    let mut magic = [0u32; MAX_BROADCAST_DIMS];
    let mut pshift = [0u32; MAX_BROADCAST_DIMS];
    for i in 0..ndim {
        a_strides[i] = a_strides_vec[i];
        b_strides[i] = b_strides_vec[i];
        shape[i] = shape_vec[i];
        let (m, s) = compute_magic_divisor(shape_vec[i]);
        magic[i] = m;
        pshift[i] = s;
    }
    // Zero-padded dims: shape=0 means magic=0, shift=0. The kernel skips them via ndim.

    let func_name = format!("{}_broadcast_{}_inline", op, dtype_str);
    let func = get_kernel_function(&module, &func_name)?;
    let ndim_u32 = ndim as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        // a_strides[0..7]
        builder.arg(&a_strides[0]);
        builder.arg(&a_strides[1]);
        builder.arg(&a_strides[2]);
        builder.arg(&a_strides[3]);
        builder.arg(&a_strides[4]);
        builder.arg(&a_strides[5]);
        builder.arg(&a_strides[6]);
        builder.arg(&a_strides[7]);
        // b_strides[0..7]
        builder.arg(&b_strides[0]);
        builder.arg(&b_strides[1]);
        builder.arg(&b_strides[2]);
        builder.arg(&b_strides[3]);
        builder.arg(&b_strides[4]);
        builder.arg(&b_strides[5]);
        builder.arg(&b_strides[6]);
        builder.arg(&b_strides[7]);
        // shape[0..7]
        builder.arg(&shape[0]);
        builder.arg(&shape[1]);
        builder.arg(&shape[2]);
        builder.arg(&shape[3]);
        builder.arg(&shape[4]);
        builder.arg(&shape[5]);
        builder.arg(&shape[6]);
        builder.arg(&shape[7]);
        // magic[0..7]
        builder.arg(&magic[0]);
        builder.arg(&magic[1]);
        builder.arg(&magic[2]);
        builder.arg(&magic[3]);
        builder.arg(&magic[4]);
        builder.arg(&magic[5]);
        builder.arg(&magic[6]);
        builder.arg(&magic[7]);
        // pshift[0..7]
        builder.arg(&pshift[0]);
        builder.arg(&pshift[1]);
        builder.arg(&pshift[2]);
        builder.arg(&pshift[3]);
        builder.arg(&pshift[4]);
        builder.arg(&pshift[5]);
        builder.arg(&pshift[6]);
        builder.arg(&pshift[7]);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA broadcast binary kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}
