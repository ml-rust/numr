//! Helper functions for CPU tensor operations
//!
//! This module contains shared helper functions used by TensorOps, ScalarOps,
//! and CompareOps implementations.

use super::{CpuClient, CpuRuntime, kernels};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOp, CompareOp, Kernel, ReduceOp, UnaryOp, broadcast_shape, reduce_output_shape,
};
use crate::tensor::Tensor;

// ============================================================================
// DType Dispatch Macro
// ============================================================================

/// Macro for dtype dispatch to typed kernel calls
///
/// This macro matches on dtype and executes the code block with the appropriate type.
/// Usage: `dispatch_dtype!(dtype, T => { code using T }, "op_name")`
///
/// F16 and BF16 are supported when the "f16" feature is enabled.
/// FP8E4M3 and FP8E5M2 are supported when the "fp8" feature is enabled.
/// Without the respective feature, these types return `UnsupportedDType` error.
macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident => $body:block, $error_op:expr) => {
        match $dtype {
            DType::F64 => {
                type $T = f64;
                $body
            }
            DType::F32 => {
                type $T = f32;
                $body
            }
            DType::F16 => {
                #[cfg(feature = "f16")]
                {
                    type $T = half::f16;
                    $body
                }
                #[cfg(not(feature = "f16"))]
                {
                    return Err(Error::UnsupportedDType {
                        dtype: $dtype,
                        op: $error_op,
                    });
                }
            }
            DType::BF16 => {
                #[cfg(feature = "f16")]
                {
                    type $T = half::bf16;
                    $body
                }
                #[cfg(not(feature = "f16"))]
                {
                    return Err(Error::UnsupportedDType {
                        dtype: $dtype,
                        op: $error_op,
                    });
                }
            }
            DType::FP8E4M3 => {
                #[cfg(feature = "fp8")]
                {
                    type $T = crate::dtype::FP8E4M3;
                    $body
                }
                #[cfg(not(feature = "fp8"))]
                {
                    return Err(Error::UnsupportedDType {
                        dtype: $dtype,
                        op: $error_op,
                    });
                }
            }
            DType::FP8E5M2 => {
                #[cfg(feature = "fp8")]
                {
                    type $T = crate::dtype::FP8E5M2;
                    $body
                }
                #[cfg(not(feature = "fp8"))]
                {
                    return Err(Error::UnsupportedDType {
                        dtype: $dtype,
                        op: $error_op,
                    });
                }
            }
            DType::I64 => {
                type $T = i64;
                $body
            }
            DType::I32 => {
                type $T = i32;
                $body
            }
            DType::I16 => {
                type $T = i16;
                $body
            }
            DType::I8 => {
                type $T = i8;
                $body
            }
            DType::U64 => {
                type $T = u64;
                $body
            }
            DType::U32 => {
                type $T = u32;
                $body
            }
            DType::U16 => {
                type $T = u16;
                $body
            }
            DType::U8 => {
                type $T = u8;
                $body
            }
            DType::Bool => {
                return Err(Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $error_op,
                })
            }
        }
    };
}

pub(super) use dispatch_dtype;

// ============================================================================
// Helper Functions
// ============================================================================

/// Ensure a tensor is contiguous, cloning if already contiguous or copying if not.
///
/// This is a common pattern used before kernel dispatch to ensure memory layout
/// is suitable for efficient computation.
#[inline]
pub(super) fn ensure_contiguous(tensor: &Tensor<CpuRuntime>) -> Tensor<CpuRuntime> {
    if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()
    }
}

/// Validate that two tensors have matching dtypes for binary operations.
#[inline]
pub(super) fn validate_binary_dtypes(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<DType> {
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(a.dtype())
}

/// Compute broadcast shape for binary operations.
///
/// Returns the output shape after broadcasting, or an error if shapes are incompatible.
#[inline]
pub(super) fn compute_broadcast_shape(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Vec<usize>> {
    broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

// ============================================================================
// Operation Implementation Helpers
// ============================================================================

pub(super) fn binary_op_impl(
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

pub(super) fn unary_op_impl(
    client: &CpuClient,
    op: UnaryOp,
    a: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let len = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            <CpuClient as Kernel<CpuRuntime>>::unary_op::<T>(
                client, op,
                a_ptr as *const T,
                out_ptr as *mut T,
                len,
            );
        }
    }, op_name);

    Ok(out)
}

pub(super) fn scalar_op_impl(
    client: &CpuClient,
    op: BinaryOp,
    a: &Tensor<CpuRuntime>,
    scalar: f64,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let len = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scalar_op_kernel::<T>(
                op,
                a_ptr as *const T,
                scalar,
                out_ptr as *mut T,
                len,
            );
        }
    }, op_name);

    Ok(out)
}

pub(super) fn compare_op_impl(
    client: &CpuClient,
    op: CompareOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
    let out_ptr = out.storage().ptr();

    // Fast path for same shapes, both contiguous
    let same_shapes = a.shape() == b.shape() && a.shape() == out_shape.as_slice();
    let both_contiguous = a.is_contiguous() && b.is_contiguous();

    if same_shapes && both_contiguous {
        let len = a.numel();
        let a_ptr = a.storage().ptr();
        let b_ptr = b.storage().ptr();

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
        // Broadcasting path: use strided kernel
        let a_broadcast = a.broadcast_to(&out_shape)?;
        let b_broadcast = b.broadcast_to(&out_shape)?;

        let a_strides: Vec<isize> = a_broadcast.layout().strides().to_vec();
        let b_strides: Vec<isize> = b_broadcast.layout().strides().to_vec();
        let a_offset = a_broadcast.layout().offset();
        let b_offset = b_broadcast.layout().offset();
        let a_ptr = a_broadcast.storage().ptr();
        let b_ptr = b_broadcast.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::compare_op_strided_kernel::<T>(
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

// ============================================================================
// Activation Helpers
// ============================================================================

/// Activation operation kind for kernel dispatch
#[derive(Copy, Clone)]
pub(super) enum ActivationOp {
    ReLU,
    Sigmoid,
    SiLU,
    GELU,
}

/// Helper for activation operations (relu, sigmoid)
pub(super) fn activation_op_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    op: ActivationOp,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let len = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            match op {
                ActivationOp::ReLU => kernels::relu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Sigmoid => kernels::sigmoid_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::SiLU => kernels::silu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::GELU => kernels::gelu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
            }
        }
    }, op_name);

    Ok(out)
}

// ============================================================================
// Reduction Helpers
// ============================================================================

pub(super) fn reduce_impl(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimensions
    for &d in dims {
        if d >= ndim {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim,
            });
        }
    }

    // For now, only support reducing the last dimension when contiguous
    // Full reduction support requires permuting data or strided kernels
    if dims.len() == 1 && dims[0] == ndim - 1 && a.is_contiguous() {
        // Simple case: reduce last dimension
        let reduce_size = shape[ndim - 1];
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = outer_size.max(1);

        let out_shape = reduce_output_shape(shape, dims, keepdim);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::reduce::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, op_name);

        Ok(out)
    } else if dims.is_empty() {
        // No dimensions to reduce - return a copy
        Ok(a.clone())
    } else {
        // General case: need to handle arbitrary dimensions
        // For now, make contiguous and reduce sequentially
        let a_contig = ensure_contiguous(a);

        // Reduce one dimension at a time, from highest to lowest
        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse();

        let mut current = a_contig;
        for &dim in &sorted_dims {
            current = reduce_single_dim(client, op, &current, dim, keepdim, op_name)?;
        }

        Ok(current)
    }
}

/// Reduce a single dimension
fn reduce_single_dim(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim + 1..].iter().product();
    let inner_size = inner_size.max(1);

    let out_shape = reduce_output_shape(shape, &[dim], keepdim);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    // If reducing non-last dimension, we need special handling
    if dim == ndim - 1 {
        // Reducing last dimension - can use kernel directly
        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::reduce::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, op_name);
    } else {
        // Reducing non-last dimension - iterate manually
        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                reduce_non_last_dim::<T>(
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    outer_size,
                    reduce_size,
                    inner_size,
                );
            }
        }, op_name);
    }

    Ok(out)
}

/// Reduce a non-last dimension
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => T::zero(),
                ReduceOp::Prod => T::one(),
                ReduceOp::Max => {
                    let idx = outer * reduce_size * inner_size + inner;
                    *a.add(idx)
                }
                ReduceOp::Min => {
                    let idx = outer * reduce_size * inner_size + inner;
                    *a.add(idx)
                }
                ReduceOp::All => T::one(),
                ReduceOp::Any => T::zero(),
            };

            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = *a.add(idx);

                acc = match op {
                    ReduceOp::Sum | ReduceOp::Mean => acc + val,
                    ReduceOp::Prod => acc * val,
                    ReduceOp::Max => {
                        if val > acc {
                            val
                        } else {
                            acc
                        }
                    }
                    ReduceOp::Min => {
                        if val < acc {
                            val
                        } else {
                            acc
                        }
                    }
                    ReduceOp::All => {
                        if val.to_f64() != 0.0 && acc.to_f64() != 0.0 {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }
                    ReduceOp::Any => {
                        if val.to_f64() != 0.0 || acc.to_f64() != 0.0 {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }
                };
            }

            // Apply mean scaling if needed
            if matches!(op, ReduceOp::Mean) {
                acc = T::from_f64(acc.to_f64() / reduce_size as f64);
            }

            let out_idx = outer * inner_size + inner;
            *out.add(out_idx) = acc;
        }
    }
}
