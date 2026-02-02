//! Reduction operation helpers for CPU tensors

use super::super::kernels;
use super::super::kernels::Accumulator;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{AccumulationPrecision, Kernel, ReduceOp, reduce_output_shape};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Reduce implementation with native precision
pub fn reduce_impl(
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
pub fn reduce_impl_with_precision(
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
