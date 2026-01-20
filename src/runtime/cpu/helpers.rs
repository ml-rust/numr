//! Helper functions for CPU tensor operations
//!
//! This module contains shared helper functions used by TensorOps, ScalarOps,
//! and CompareOps implementations.

use super::kernels::Accumulator;
use super::{CpuClient, CpuRuntime, kernels};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{
    AccumulationPrecision, BinaryOp, CompareOp, Kernel, ReduceOp, UnaryOp, broadcast_shape,
    reduce_output_shape,
};
use crate::tensor::Tensor;

// ============================================================================
// DType Dispatch Macro (re-exported from crate root)
// ============================================================================
//
// Use the canonical dispatch_dtype! macro from src/ops/dispatch.rs.
// This avoids duplicating the macro logic and ensures a single source of truth.

pub(super) use crate::dispatch_dtype;
pub(super) use crate::runtime::ensure_contiguous;

// ============================================================================
// Helper Functions
// ============================================================================

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
    Relu,
    Sigmoid,
    Silu,
    Gelu,
}

/// Parametric activation operation kind (activations that take a scalar parameter)
#[derive(Copy, Clone)]
pub(super) enum ParametricActivationOp {
    /// LeakyReLU: x if x > 0, else negative_slope * x
    LeakyRelu,
    /// ELU: x if x > 0, else alpha * (exp(x) - 1)
    Elu,
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
                ActivationOp::Relu => kernels::relu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Sigmoid => kernels::sigmoid_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Silu => kernels::silu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Gelu => kernels::gelu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
            }
        }
    }, op_name);

    Ok(out)
}

/// Helper for parametric activation operations (leaky_relu, elu)
///
/// These activations take a single f64 parameter in addition to the input tensor.
pub(super) fn parametric_activation_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    op: ParametricActivationOp,
    param: f64,
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
                ParametricActivationOp::LeakyRelu => kernels::leaky_relu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                    param,
                ),
                ParametricActivationOp::Elu => kernels::elu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                    param,
                ),
            }
        }
    }, op_name);

    Ok(out)
}

/// Helper for leaky_relu activation
#[inline]
pub(super) fn leaky_relu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    negative_slope: f64,
) -> Result<Tensor<CpuRuntime>> {
    parametric_activation_impl(
        client,
        a,
        ParametricActivationOp::LeakyRelu,
        negative_slope,
        "leaky_relu",
    )
}

/// Helper for ELU activation
#[inline]
pub(super) fn elu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    alpha: f64,
) -> Result<Tensor<CpuRuntime>> {
    parametric_activation_impl(client, a, ParametricActivationOp::Elu, alpha, "elu")
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

/// Reduce a single dimension of a tensor using native precision.
///
/// This is an optimized path for single-dimension reductions. Uses chunked
/// iteration for non-last dimensions to handle strided memory access.
///
/// # Arguments
/// * `op` - Reduction operation (Sum, Max, Min)
/// * `dim` - Dimension to reduce
/// * `keepdim` - Whether to keep the reduced dimension as size 1
/// * `op_name` - Name for error messages
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

// ============================================================================
// Precision-Aware Reduction Helpers
// ============================================================================

/// Reduce implementation with explicit accumulation precision
pub(super) fn reduce_impl_with_precision(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    precision: AccumulationPrecision,
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
                kernels::reduce_kernel_with_precision::<T>(
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                    precision,
                );
            }
        }, op_name);

        Ok(out)
    } else if dims.is_empty() {
        // No dimensions to reduce - return a copy
        Ok(a.clone())
    } else {
        // General case: need to handle arbitrary dimensions
        let a_contig = ensure_contiguous(a);

        // Reduce one dimension at a time, from highest to lowest
        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse();

        let mut current = a_contig;
        for &dim in &sorted_dims {
            current = reduce_single_dim_with_precision(
                client, op, &current, dim, keepdim, precision, op_name,
            )?;
        }

        Ok(current)
    }
}

/// Reduce a single dimension with explicit accumulation precision.
///
/// Similar to [`reduce_single_dim`], but allows specifying higher precision
/// for accumulation. This is important for numerical stability when summing
/// many small values (e.g., F16 tensors accumulated in F32).
///
/// # Arguments
/// * `op` - Reduction operation (Sum, Max, Min)
/// * `dim` - Dimension to reduce
/// * `keepdim` - Whether to keep the reduced dimension as size 1
/// * `precision` - Accumulation precision (Native, F32, or F64)
/// * `op_name` - Name for error messages
fn reduce_single_dim_with_precision(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
    precision: AccumulationPrecision,
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

    let a_ptr = a.storage().ptr();
    let out_ptr = out.storage().ptr();

    if dim == ndim - 1 {
        // Reducing last dimension - use kernel directly
        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::reduce_kernel_with_precision::<T>(
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                    precision,
                );
            }
        }, op_name);
    } else {
        // Reducing non-last dimension - iterate manually with precision
        dispatch_dtype!(dtype, T => {
            unsafe {
                reduce_non_last_dim_with_precision::<T>(
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    outer_size,
                    reduce_size,
                    inner_size,
                    precision,
                );
            }
        }, op_name);
    }

    Ok(out)
}

/// Reduce a non-last dimension with explicit precision
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim_with_precision<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    precision: AccumulationPrecision,
) {
    match precision {
        AccumulationPrecision::Native => {
            reduce_non_last_dim(op, a, out, outer_size, reduce_size, inner_size);
        }
        AccumulationPrecision::FP32 | AccumulationPrecision::BF16 => {
            // For CPU, both FP32 and BF16 accumulation use f32 internally
            reduce_non_last_dim_f32_acc(op, a, out, outer_size, reduce_size, inner_size);
        }
        AccumulationPrecision::FP64 => {
            // Maximum precision for math/science applications
            reduce_non_last_dim_f64_acc(op, a, out, outer_size, reduce_size, inner_size);
        }
    }
}

/// Generic reduce for non-last dimension with configurable accumulation precision.
///
/// Uses the Accumulator trait to abstract over f32/f64 accumulation types.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim_acc<T: Element, A: Accumulator>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let first_idx = outer * reduce_size * inner_size + inner;
            let first_val = A::acc_in((*a.add(first_idx)).to_f64());

            let mut acc: A = match op {
                ReduceOp::Sum | ReduceOp::Mean => A::ZERO,
                ReduceOp::Prod => A::ONE,
                ReduceOp::Max | ReduceOp::Min => first_val,
                ReduceOp::All => A::ONE,
                ReduceOp::Any => A::ZERO,
            };

            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = A::acc_in((*a.add(idx)).to_f64());

                acc = match op {
                    ReduceOp::Sum | ReduceOp::Mean => acc.acc_add(val),
                    ReduceOp::Prod => acc.acc_mul(val),
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
                        if val != A::ZERO && acc != A::ZERO {
                            A::ONE
                        } else {
                            A::ZERO
                        }
                    }
                    ReduceOp::Any => {
                        if val != A::ZERO || acc != A::ZERO {
                            A::ONE
                        } else {
                            A::ZERO
                        }
                    }
                };
            }

            // Apply mean scaling if needed
            if matches!(op, ReduceOp::Mean) {
                acc = acc.acc_div(reduce_size);
            }

            let out_idx = outer * inner_size + inner;
            *out.add(out_idx) = T::from_f64(acc.into());
        }
    }
}

/// Reduce a non-last dimension with f32 accumulation (convenience wrapper)
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn reduce_non_last_dim_f32_acc<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    reduce_non_last_dim_acc::<T, f32>(op, a, out, outer_size, reduce_size, inner_size)
}

/// Reduce a non-last dimension with f64 accumulation (convenience wrapper)
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn reduce_non_last_dim_f64_acc<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    reduce_non_last_dim_acc::<T, f64>(op, a, out, outer_size, reduce_size, inner_size)
}

// ============================================================================
// Indexing Operation Helpers
// ============================================================================

/// Gather elements along a dimension using an index tensor.
pub(super) fn gather_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Validate index dimensions
    if index.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index.shape().to_vec(),
        });
    }

    // Output shape is same as index shape
    let out_shape = index.shape().to_vec();

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let a_ptr = a_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::gather_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                out_ptr as *mut T,
                shape,
                &out_shape,
                dim,
            );
        }
    }, "gather");

    Ok(out)
}

/// Scatter values into a tensor at positions specified by an index tensor.
pub(super) fn scatter_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate dtypes
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Validate shapes
    if index.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let src_contig = ensure_contiguous(src);
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);

    let a_ptr = a_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let src_ptr = src_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scatter_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                src_ptr as *const T,
                out_ptr as *mut T,
                shape,
                index.shape(),
                dim,
            );
        }
    }, "scatter");

    Ok(out)
}

/// Select elements along a dimension using a 1D index tensor.
pub(super) fn index_select_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Index must be 1D
    if index.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index.numel()],
            got: index.shape().to_vec(),
        });
    }

    let index_len = index.shape()[0];

    // Output shape: replace dimension `dim` with index length
    let mut out_shape = shape.to_vec();
    out_shape[dim] = index_len;

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let a_ptr = a_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::index_select_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                out_ptr as *mut T,
                shape,
                dim,
                index_len,
            );
        }
    }, "index_select");

    Ok(out)
}

/// Select elements where mask is true, returning a flattened 1D tensor.
pub(super) fn masked_select_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    mask: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();

    // Validate mask dtype
    if mask.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to input shape
    let mask_broadcast = mask.broadcast_to(a.shape())?;

    let a_contig = ensure_contiguous(a);
    let mask_contig = ensure_contiguous(&mask_broadcast);

    let numel = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let mask_ptr = mask_contig.storage().ptr();

    // Count true elements first
    let count = unsafe { kernels::masked_count_kernel(mask_ptr as *const u8, numel) };

    // Allocate output with correct size
    let out = Tensor::<CpuRuntime>::empty(&[count], dtype, &client.device);
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::masked_select_kernel::<T>(
                a_ptr as *const T,
                mask_ptr as *const u8,
                out_ptr as *mut T,
                numel,
            );
        }
    }, "masked_select");

    Ok(out)
}

/// Fill elements where mask is true with a scalar value.
pub(super) fn masked_fill_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    mask: &Tensor<CpuRuntime>,
    value: f64,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();

    // Validate mask dtype
    if mask.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to input shape
    let mask_broadcast = mask.broadcast_to(a.shape())?;

    let a_contig = ensure_contiguous(a);
    let mask_contig = ensure_contiguous(&mask_broadcast);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let numel = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let mask_ptr = mask_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::masked_fill_kernel::<T>(
                a_ptr as *const T,
                mask_ptr as *const u8,
                out_ptr as *mut T,
                numel,
                value,
            );
        }
    }, "masked_fill");

    Ok(out)
}

// ============================================================================
// Shape Operation Helpers
// ============================================================================

use crate::runtime::shape_ops::{self, validate_cat, validate_stack};

/// Concatenate tensors along a dimension
pub(super) fn cat_impl(
    client: &CpuClient,
    tensors: &[&Tensor<CpuRuntime>],
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = validate_cat(tensors, dim)?;

    // Allocate output
    let out = Tensor::<CpuRuntime>::empty(&params.out_shape, params.dtype, &client.device);
    let out_ptr = out.storage().ptr();

    // Copy data from each tensor
    dispatch_dtype!(params.dtype, T => {
        unsafe {
            let mut cat_offset = 0usize;
            for &tensor in tensors {
                let tensor_contig = ensure_contiguous(tensor);
                let src_ptr = tensor_contig.storage().ptr() as *const T;
                let src_cat_size = tensor.shape()[params.dim_idx];

                // Copy each row-block
                for outer in 0..params.outer_size {
                    for cat_i in 0..src_cat_size {
                        let src_base = outer * src_cat_size * params.inner_size + cat_i * params.inner_size;
                        let dst_base = outer * params.cat_dim_total * params.inner_size + (cat_offset + cat_i) * params.inner_size;

                        std::ptr::copy_nonoverlapping(
                            src_ptr.add(src_base),
                            (out_ptr as *mut T).add(dst_base),
                            params.inner_size,
                        );
                    }
                }

                cat_offset += src_cat_size;
            }
        }
    }, "cat");

    Ok(out)
}

/// Stack tensors along a new dimension
pub(super) fn stack_impl(
    client: &CpuClient,
    tensors: &[&Tensor<CpuRuntime>],
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let _dim_idx = validate_stack(tensors, dim)?;

    // Unsqueeze each tensor and then cat
    // stack(tensors, dim) = cat([t.unsqueeze(dim) for t in tensors], dim)
    let unsqueezed: Vec<Tensor<CpuRuntime>> = tensors
        .iter()
        .map(|t| t.unsqueeze(dim))
        .collect::<Result<_>>()?;

    let unsqueezed_refs: Vec<&Tensor<CpuRuntime>> = unsqueezed.iter().collect();
    cat_impl(client, &unsqueezed_refs, dim)
}

/// Split a tensor into chunks of a given size along a dimension (zero-copy)
pub(super) fn split_impl(
    tensor: &Tensor<CpuRuntime>,
    split_size: usize,
    dim: isize,
) -> Result<Vec<Tensor<CpuRuntime>>> {
    shape_ops::split_impl(tensor, split_size, dim)
}

/// Split a tensor into a specific number of chunks (zero-copy)
pub(super) fn chunk_impl(
    tensor: &Tensor<CpuRuntime>,
    chunks: usize,
    dim: isize,
) -> Result<Vec<Tensor<CpuRuntime>>> {
    shape_ops::chunk_impl(tensor, chunks, dim)
}

/// Repeat tensor along each dimension
pub(super) fn repeat_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    repeats: &[usize],
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = shape_ops::validate_repeat(tensor, repeats)?;

    // Handle case where all repeats are 1 (no-op)
    if repeats.iter().all(|&r| r == 1) {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let in_shape = tensor.shape();
    let out = Tensor::<CpuRuntime>::empty(&params.out_shape, dtype, &client.device);

    // Make input contiguous
    let tensor_contig = ensure_contiguous(tensor);
    let src_ptr = tensor_contig.storage().ptr();
    let dst_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            repeat_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                in_shape,
                &params.out_shape,
                repeats,
            );
        }
    }, "repeat");

    Ok(out)
}

/// Kernel for repeat operation
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn repeat_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    in_shape: &[usize],
    out_shape: &[usize],
    _repeats: &[usize],
) {
    let ndim = in_shape.len();
    let out_numel: usize = out_shape.iter().product();

    // Compute strides for input and output
    let mut in_strides = vec![1usize; ndim];
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // For each output element, compute the corresponding input element
    for out_idx in 0..out_numel {
        // Convert flat index to multi-dimensional indices for output
        let mut remaining = out_idx;
        let mut src_offset = 0usize;

        for d in 0..ndim {
            let out_coord = remaining / out_strides[d];
            remaining %= out_strides[d];

            // Map output coordinate to input coordinate using modulo
            let in_coord = out_coord % in_shape[d];
            src_offset += in_coord * in_strides[d];
        }

        *dst.add(out_idx) = *src.add(src_offset);
    }
}

/// Pad tensor with a constant value
pub(super) fn pad_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    padding: &[usize],
    value: f64,
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = shape_ops::validate_pad(tensor, padding)?;

    // Handle case where no padding is added
    if params.pad_per_dim.iter().all(|&(b, a)| b == 0 && a == 0) {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let in_shape = tensor.shape();

    // Create output filled with padding value
    let out = Tensor::<CpuRuntime>::full_scalar(&params.out_shape, dtype, value, &client.device);

    // Make input contiguous
    let tensor_contig = ensure_contiguous(tensor);
    let src_ptr = tensor_contig.storage().ptr();
    let dst_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            pad_copy_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                in_shape,
                &params.out_shape,
                &params.pad_per_dim,
            );
        }
    }, "pad");

    Ok(out)
}

/// Kernel for copying input data into padded output
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pad_copy_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    in_shape: &[usize],
    out_shape: &[usize],
    pad_per_dim: &[(usize, usize)],
) {
    let ndim = in_shape.len();
    let in_numel: usize = in_shape.iter().product();

    // Compute strides for input and output
    let mut in_strides = vec![1usize; ndim];
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Copy each input element to its padded position
    for in_idx in 0..in_numel {
        let mut remaining = in_idx;
        let mut dst_offset = 0usize;

        for d in 0..ndim {
            let in_coord = remaining / in_strides[d];
            remaining %= in_strides[d];

            // Add padding offset
            let out_coord = in_coord + pad_per_dim[d].0;
            dst_offset += out_coord * out_strides[d];
        }

        *dst.add(dst_offset) = *src.add(in_idx);
    }
}

/// Roll tensor elements along a dimension
pub(super) fn roll_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    shift: isize,
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = shape_ops::validate_roll(tensor, shift, dim)?;

    // Handle case where shift is 0 (no-op)
    if params.shift == 0 {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let shape = tensor.shape();
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);

    // Make input contiguous
    let tensor_contig = ensure_contiguous(tensor);
    let src_ptr = tensor_contig.storage().ptr();
    let dst_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            roll_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                shape,
                params.dim_idx,
                params.shift,
                params.dim_size,
            );
        }
    }, "roll");

    Ok(out)
}

/// Kernel for roll operation
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn roll_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    shape: &[usize],
    dim_idx: usize,
    shift: usize,
    dim_size: usize,
) {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();

    // Compute strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // For each element, compute rolled position
    for idx in 0..numel {
        let mut remaining = idx;
        let mut dst_offset = 0usize;

        for d in 0..ndim {
            let coord = remaining / strides[d];
            remaining %= strides[d];

            if d == dim_idx {
                // Apply circular shift
                let new_coord = (coord + shift) % dim_size;
                dst_offset += new_coord * strides[d];
            } else {
                dst_offset += coord * strides[d];
            }
        }

        *dst.add(dst_offset) = *src.add(idx);
    }
}

// ============================================================================
// Cumulative Operations (cumsum, cumprod, logsumexp)
// ============================================================================

/// Normalize a dimension index, allowing negative indexing
#[inline]
fn normalize_dim(ndim: usize, dim: isize) -> Option<usize> {
    if dim >= 0 {
        let d = dim as usize;
        if d < ndim { Some(d) } else { None }
    } else {
        let d = dim + ndim as isize;
        if d >= 0 { Some(d as usize) } else { None }
    }
}

/// Cumulative sum along a dimension
pub(super) fn cumsum_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    let dim_idx = normalize_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

    // Handle empty tensor
    if a.numel() == 0 {
        return Ok(Tensor::<CpuRuntime>::empty(shape, dtype, &client.device));
    }

    // Make contiguous for simplicity
    let a_contig = ensure_contiguous(a);

    // Output has same shape as input
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);

    // Compute sizes for the scan
    let scan_size = shape[dim_idx];
    let outer_size: usize = shape[..dim_idx].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim_idx + 1..].iter().product();
    let inner_size = inner_size.max(1);

    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            if inner_size == 1 {
                // Fast path: scan dimension is last (or tensor is 1D)
                kernels::cumsum_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                );
            } else {
                // General path: strided access
                kernels::cumsum_strided_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                    inner_size,
                );
            }
        }
    }, "cumsum");

    Ok(out)
}

/// Cumulative product along a dimension
pub(super) fn cumprod_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    let dim_idx = normalize_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

    // Handle empty tensor
    if a.numel() == 0 {
        return Ok(Tensor::<CpuRuntime>::empty(shape, dtype, &client.device));
    }

    // Make contiguous for simplicity
    let a_contig = ensure_contiguous(a);

    // Output has same shape as input
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);

    // Compute sizes for the scan
    let scan_size = shape[dim_idx];
    let outer_size: usize = shape[..dim_idx].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim_idx + 1..].iter().product();
    let inner_size = inner_size.max(1);

    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            if inner_size == 1 {
                // Fast path: scan dimension is last (or tensor is 1D)
                kernels::cumprod_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                );
            } else {
                // General path: strided access
                kernels::cumprod_strided_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                    inner_size,
                );
            }
        }
    }, "cumprod");

    Ok(out)
}

/// Log-sum-exp along specified dimensions (numerically stable)
///
/// Only supports floating-point dtypes (F32, F64, F16, BF16).
pub(super) fn logsumexp_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Logsumexp only makes sense for floating-point types
    if !dtype.is_float() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "logsumexp",
        });
    }

    // Validate dimensions
    for &d in dims {
        if d >= ndim {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim,
            });
        }
    }

    // For single last-dimension reduction on contiguous tensor, use fast path
    if dims.len() == 1 && dims[0] == ndim - 1 && a.is_contiguous() {
        let reduce_size = shape[ndim - 1];
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = outer_size.max(1);

        let out_shape = reduce_output_shape(shape, dims, keepdim);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::logsumexp_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, "logsumexp");

        return Ok(out);
    }

    // For empty dims, return copy
    if dims.is_empty() {
        return Ok(a.clone());
    }

    // General case: reduce one dimension at a time
    let a_contig = ensure_contiguous(a);

    let mut sorted_dims: Vec<usize> = dims.to_vec();
    sorted_dims.sort_unstable();
    sorted_dims.reverse();

    let mut current = a_contig;
    for &dim in &sorted_dims {
        current = logsumexp_single_dim(client, &current, dim, keepdim)?;
    }

    Ok(current)
}

/// Log-sum-exp along a single dimension
fn logsumexp_single_dim(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
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

    // Output shape: remove the reduced dimension (or keep as 1)
    let out_shape: Vec<usize> = if keepdim {
        shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == dim { 1 } else { s })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if i != dim { Some(s) } else { None })
            .collect()
    };

    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let a_ptr = a.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            if inner_size == 1 {
                // Fast path: reduce dimension is last
                kernels::logsumexp_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            } else {
                // General path: strided access
                kernels::logsumexp_strided_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                    inner_size,
                    inner_size, // in_stride
                    inner_size, // out_stride
                );
            }
        }
    }, "logsumexp");

    Ok(out)
}
