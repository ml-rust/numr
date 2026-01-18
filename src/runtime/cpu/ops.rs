//! TensorOps, ScalarOps, and CompareOps implementations for CPU runtime
//!
//! This module contains the high-level operation trait implementations that
//! dispatch to the low-level kernels.

use super::helpers::{
    ActivationOp, activation_op_impl, binary_op_impl, compare_op_impl, dispatch_dtype,
    ensure_contiguous, reduce_impl, reduce_impl_with_precision, scalar_op_impl, unary_op_impl,
};
use super::{CpuClient, CpuRuntime, kernels};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{
    AccumulationPrecision, BinaryOp, CompareOp, CompareOps, Kernel, LogicalOps, ReduceOp,
    ScalarOps, TensorOps, UnaryOp, compute_reduce_strides, normalize_softmax_dim,
    reduce_dim_output_shape,
};
use crate::tensor::Tensor;

// ============================================================================
// TensorOps Implementation
// ============================================================================

/// TensorOps implementation for CPU runtime.
impl TensorOps<CpuRuntime> for CpuClient {
    // ===== Binary Operations =====

    fn add(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Add, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Sub, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Mul, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Div, a, b, "div")
    }

    // ===== Unary Operations =====

    fn neg(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Neg, a, "neg")
    }

    fn abs(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Abs, a, "abs")
    }

    fn sqrt(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sqrt, a, "sqrt")
    }

    fn exp(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Exp, a, "exp")
    }

    fn log(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Log, a, "log")
    }

    fn sin(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sin, a, "sin")
    }

    fn cos(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Cos, a, "cos")
    }

    fn tanh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Tanh, a, "tanh")
    }

    fn tan(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Tan, a, "tan")
    }

    fn recip(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Recip, a, "recip")
    }

    fn square(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Square, a, "square")
    }

    fn floor(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Floor, a, "floor")
    }

    fn ceil(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Ceil, a, "ceil")
    }

    fn round(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Round, a, "round")
    }

    fn sign(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sign, a, "sign")
    }

    fn isnan(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();
        let numel = a.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::isnan_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut u8,
                    numel,
                );
            }
        }, "isnan");

        Ok(out)
    }

    fn isinf(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();
        let numel = a.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::isinf_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut u8,
                    numel,
                );
            }
        }, "isinf");

        Ok(out)
    }

    // ===== Element-wise Binary (extended) =====

    fn pow(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Pow, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Max, a, b, "maximum")
    }

    fn minimum(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Min, a, b, "minimum")
    }

    // ===== Matrix Operations =====

    fn matmul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::matmul_output_shape;

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        let dtype = a.dtype();

        // Compute output shape
        let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        })?;

        // Get matrix dimensions (last two dims)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // For now, require contiguous tensors
        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);

        // Calculate batch size
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Create output tensor
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let b_ptr = b_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        // Leading dimensions for contiguous row-major matrices
        let lda = k;
        let ldb = n;
        let ldc = n;

        // Dispatch based on dtype
        dispatch_dtype!(dtype, T => {
            unsafe {
                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let out_offset = batch * m * n;

                    <Self as Kernel<CpuRuntime>>::matmul::<T>(
                        self,
                        (a_ptr as *const T).add(a_offset),
                        (b_ptr as *const T).add(b_offset),
                        (out_ptr as *mut T).add(out_offset),
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                }
            }
        }, "matmul");

        Ok(out)
    }

    // ===== Reductions =====

    fn sum(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Sum, a, dims, keepdim, "sum")
    }

    fn sum_with_precision(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl_with_precision(self, ReduceOp::Sum, a, dims, keepdim, precision, "sum")
    }

    fn mean(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Mean, a, dims, keepdim, "mean")
    }

    fn max(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Max, a, dims, keepdim, "max")
    }

    fn max_with_precision(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl_with_precision(self, ReduceOp::Max, a, dims, keepdim, precision, "max")
    }

    fn min(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Min, a, dims, keepdim, "min")
    }

    fn min_with_precision(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl_with_precision(self, ReduceOp::Min, a, dims, keepdim, precision, "min")
    }

    // ===== Activations =====

    fn relu(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Relu, "relu")
    }

    fn sigmoid(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Sigmoid, "sigmoid")
    }

    fn silu(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Silu, "silu")
    }

    fn gelu(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Gelu, "gelu")
    }

    fn softmax(&self, a: &Tensor<CpuRuntime>, dim: isize) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();

        // Normalize dimension
        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &self.device);

        let shape = a.shape();

        // Calculate outer_size (product of dims before softmax dim)
        // and dim_size (size of softmax dim)
        // and inner_size (product of dims after softmax dim)
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        // For softmax, we need the data laid out so that the softmax dimension is contiguous
        // If dim is the last dimension, we can use the simple kernel
        // Otherwise, we need to iterate

        if dim_idx == ndim - 1 {
            // Simple case: softmax over last dimension
            let a_ptr = a_contig.storage().ptr();
            let out_ptr = out.storage().ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::softmax_kernel::<T>(
                        a_ptr as *const T,
                        out_ptr as *mut T,
                        outer_size,
                        dim_size,
                    );
                }
            }, "softmax");
        } else {
            // General case: softmax over non-last dimension
            // Pre-allocate buffer outside loops to avoid repeated allocations
            let a_ptr = a_contig.storage().ptr();
            let out_ptr = out.storage().ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    softmax_non_last_dim::<T>(
                        a_ptr as *const T,
                        out_ptr as *mut T,
                        outer_size,
                        dim_size,
                        inner_size,
                    );
                }
            }, "softmax");
        }

        Ok(out)
    }

    // ===== Normalization =====

    fn rms_norm(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: weight.dtype(),
            });
        }

        // Weight must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1); // Handle 1D case

        let input_contig = ensure_contiguous(input);
        let weight_contig = ensure_contiguous(weight);
        let out = Tensor::<CpuRuntime>::empty(input_shape, dtype, &self.device);

        let input_ptr = input_contig.storage().ptr();
        let weight_ptr = weight_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::rms_norm_kernel::<T>(
                    input_ptr as *const T,
                    weight_ptr as *const T,
                    out_ptr as *mut T,
                    batch_size,
                    hidden_size,
                    eps,
                );
            }
        }, "rms_norm");

        Ok(out)
    }

    fn layer_norm(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype || bias.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if weight.dtype() != dtype {
                    weight.dtype()
                } else {
                    bias.dtype()
                },
            });
        }

        // Weight and bias must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }
        if bias.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: bias.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1); // Handle 1D case

        let input_contig = ensure_contiguous(input);
        let weight_contig = ensure_contiguous(weight);
        let bias_contig = ensure_contiguous(bias);
        let out = Tensor::<CpuRuntime>::empty(input_shape, dtype, &self.device);

        let input_ptr = input_contig.storage().ptr();
        let weight_ptr = weight_contig.storage().ptr();
        let bias_ptr = bias_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::layer_norm_kernel::<T>(
                    input_ptr as *const T,
                    weight_ptr as *const T,
                    bias_ptr as *const T,
                    out_ptr as *mut T,
                    batch_size,
                    hidden_size,
                    eps,
                );
            }
        }, "layer_norm");

        Ok(out)
    }

    // ===== Index Operations =====

    fn argmax(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        keepdim: bool,
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

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::argmax_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut i64,
                    outer_size,
                    reduce_size,
                    inner_size,
                );
            }
        }, "argmax");

        Ok(out)
    }

    fn argmin(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        keepdim: bool,
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

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::argmin_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut i64,
                    outer_size,
                    reduce_size,
                    inner_size,
                );
            }
        }, "argmin");

        Ok(out)
    }

    // ===== Type Conversion =====

    fn cast(&self, a: &Tensor<CpuRuntime>, target_dtype: DType) -> Result<Tensor<CpuRuntime>> {
        let src_dtype = a.dtype();

        // No-op if types match
        if src_dtype == target_dtype {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(shape, target_dtype, &self.device);

        let src_ptr = a_contig.storage().ptr() as *const u8;
        let dst_ptr = out.storage().ptr() as *mut u8;

        unsafe {
            kernels::cast_kernel(src_ptr, dst_ptr, numel, src_dtype, target_dtype)?;
        }

        Ok(out)
    }

    // ===== Conditional Operations =====

    fn where_cond(
        &self,
        cond: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::broadcast_shape;

        // Validate that x and y have the same dtype
        if x.dtype() != y.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: x.dtype(),
                rhs: y.dtype(),
            });
        }
        let dtype = x.dtype();

        // Validate condition tensor is U8 (boolean)
        if cond.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: cond.dtype(),
            });
        }

        // Compute broadcast shape (cond, x, y) -> out
        let xy_shape =
            broadcast_shape(x.shape(), y.shape()).ok_or_else(|| Error::BroadcastError {
                lhs: x.shape().to_vec(),
                rhs: y.shape().to_vec(),
            })?;
        let out_shape =
            broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| Error::BroadcastError {
                lhs: cond.shape().to_vec(),
                rhs: xy_shape.clone(),
            })?;

        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);
        let out_ptr = out.storage().ptr();

        // Fast path: all same shape, use simple kernel
        if cond.shape() == x.shape() && x.shape() == y.shape() {
            let cond_contig = ensure_contiguous(cond);
            let x_contig = ensure_contiguous(x);
            let y_contig = ensure_contiguous(y);

            let cond_ptr = cond_contig.storage().ptr();
            let x_ptr = x_contig.storage().ptr();
            let y_ptr = y_contig.storage().ptr();
            let numel = x.numel();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::where_kernel::<T>(
                        cond_ptr as *const u8,
                        x_ptr as *const T,
                        y_ptr as *const T,
                        out_ptr as *mut T,
                        numel,
                    );
                }
            }, "where_cond");
        } else {
            // Broadcasting path: use strided kernel
            // Broadcast all inputs to output shape (zero-copy views with stride 0 for broadcast dims)
            let cond_broadcast = cond.broadcast_to(&out_shape)?;
            let x_broadcast = x.broadcast_to(&out_shape)?;
            let y_broadcast = y.broadcast_to(&out_shape)?;

            let cond_ptr = cond_broadcast.storage().ptr();
            let x_ptr = x_broadcast.storage().ptr();
            let y_ptr = y_broadcast.storage().ptr();

            // Get strides from broadcast layouts
            let cond_strides: Vec<isize> = cond_broadcast.layout().strides().to_vec();
            let x_strides: Vec<isize> = x_broadcast.layout().strides().to_vec();
            let y_strides: Vec<isize> = y_broadcast.layout().strides().to_vec();
            let cond_offset = cond_broadcast.layout().offset();
            let x_offset = x_broadcast.layout().offset();
            let y_offset = y_broadcast.layout().offset();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::where_strided_kernel::<T>(
                        cond_ptr as *const u8,
                        x_ptr as *const T,
                        y_ptr as *const T,
                        out_ptr as *mut T,
                        &out_shape,
                        &cond_strides,
                        &x_strides,
                        &y_strides,
                        cond_offset,
                        x_offset,
                        y_offset,
                    );
                }
            }, "where_cond");
        }

        Ok(out)
    }

    // ===== Utility Operations =====

    fn clamp(
        &self,
        a: &Tensor<CpuRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();
        let numel = a.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::clamp_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    numel,
                    min_val,
                    max_val,
                );
            }
        }, "clamp");

        Ok(out)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<CpuRuntime>> {
        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device);
        let out_ptr = out.storage().ptr();
        let numel = out.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::fill_kernel::<T>(
                    out_ptr as *mut T,
                    T::from_f64(value),
                    numel,
                );
            }
        }, "fill");

        Ok(out)
    }

    // ===== Statistical Operations =====

    fn var(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        // For multi-dimensional reduction, we need to handle the dims properly
        // For simplicity, we support single dimension reduction here
        // Multi-dim can be decomposed into multiple single-dim reductions
        if dims.is_empty() {
            // Reduce over all dimensions - return scalar variance
            let numel = a.numel();
            let a_contig = ensure_contiguous(a);
            let a_ptr = a_contig.storage().ptr();

            let variance = dispatch_dtype!(dtype, T => {
                unsafe {
                    let slice = std::slice::from_raw_parts(a_ptr as *const T, numel);
                    // Compute mean
                    let mut sum = 0.0f64;
                    for &val in slice {
                        sum += val.to_f64();
                    }
                    let mean = sum / (numel as f64);

                    // Compute variance
                    let mut var_sum = 0.0f64;
                    for &val in slice {
                        let diff = val.to_f64() - mean;
                        var_sum += diff * diff;
                    }
                    let divisor = (numel.saturating_sub(correction)).max(1) as f64;
                    var_sum / divisor
                }
            }, "var");

            let out_shape = if keepdim { vec![1; ndim] } else { vec![] };
            let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);
            let out_ptr = out.storage().ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    *(out_ptr as *mut T) = T::from_f64(variance);
                }
            }, "var");

            return Ok(out);
        }

        // Single dimension case
        if dims.len() == 1 {
            let dim = dims[0];
            if dim >= ndim {
                return Err(Error::InvalidDimension {
                    dim: dim as isize,
                    ndim,
                });
            }

            let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
            let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

            let a_contig = ensure_contiguous(a);
            let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);

            let a_ptr = a_contig.storage().ptr();
            let out_ptr = out.storage().ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::variance_kernel::<T>(
                        a_ptr as *const T,
                        out_ptr as *mut T,
                        outer_size,
                        reduce_size,
                        inner_size,
                        correction,
                    );
                }
            }, "var");

            return Ok(out);
        }

        // Multi-dimension case: compute variance iteratively
        // First, compute along the last dimension, then continue
        let mut result = a.clone();
        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a)); // Sort descending so we reduce from last to first

        for dim in sorted_dims {
            result = self.var(&result, &[dim], true, correction)?;
        }

        // Remove keepdim dimensions if not requested
        if !keepdim {
            let final_shape: Vec<usize> = result
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| !dims.contains(i))
                .map(|(_, &s)| s)
                .collect();
            result = result.reshape(&final_shape)?;
        }

        Ok(result)
    }

    fn std(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        // Standard deviation is sqrt of variance
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    // ===== Random Operations =====

    fn rand(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType { dtype, op: "rand" });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device);
        let out_ptr = out.storage().ptr();
        let numel = out.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::rand_uniform_kernel::<T>(out_ptr as *mut T, numel);
            }
        }, "rand");

        Ok(out)
    }

    fn randn(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType { dtype, op: "randn" });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device);
        let out_ptr = out.storage().ptr();
        let numel = out.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::rand_normal_kernel::<T>(out_ptr as *mut T, numel);
            }
        }, "randn");

        Ok(out)
    }
}

/// Softmax over non-last dimension
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn softmax_non_last_dim<T: crate::dtype::Element>(
    a_ptr: *const T,
    out_ptr: *mut T,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) {
    // Pre-allocate reusable buffer for softmax computation
    let mut slice = vec![0.0f64; dim_size];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Elements are at: outer * dim_size * inner_size + d * inner_size + inner
            let base_idx = outer * dim_size * inner_size + inner;
            let stride = inner_size;

            // Read slice into buffer
            for (d, slot) in slice.iter_mut().enumerate() {
                let idx = base_idx + d * stride;
                *slot = (*a_ptr.add(idx)).to_f64();
            }

            // Compute softmax with numerical stability
            let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut exp_sum = 0.0f64;
            for val in &mut slice {
                *val = (*val - max_val).exp();
                exp_sum += *val;
            }

            // Handle edge case: avoid division by zero
            let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };

            // Write normalized values back
            for (d, &val) in slice.iter().enumerate() {
                let idx = base_idx + d * stride;
                *out_ptr.add(idx) = T::from_f64(val * inv_sum);
            }
        }
    }
}

// ============================================================================
// ScalarOps Implementation
// ============================================================================

impl ScalarOps<CpuRuntime> for CpuClient {
    fn add_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Add, a, scalar, "add_scalar")
    }

    fn sub_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Sub, a, scalar, "sub_scalar")
    }

    fn mul_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Mul, a, scalar, "mul_scalar")
    }

    fn div_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Div, a, scalar, "div_scalar")
    }

    fn pow_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Pow, a, scalar, "pow_scalar")
    }
}

// ============================================================================
// CompareOps Implementation
// ============================================================================

impl CompareOps<CpuRuntime> for CpuClient {
    fn eq(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Eq, a, b, "eq")
    }

    fn ne(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Ne, a, b, "ne")
    }

    fn lt(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Lt, a, b, "lt")
    }

    fn le(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Le, a, b, "le")
    }

    fn gt(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Gt, a, b, "gt")
    }

    fn ge(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Ge, a, b, "ge")
    }
}

// ============================================================================
// LogicalOps Implementation
// ============================================================================

impl LogicalOps<CpuRuntime> for CpuClient {
    fn logical_and(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate both tensors are U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }
        if b.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: b.dtype(),
            });
        }

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device);

        let a_ptr = a_contig.storage().ptr() as *const u8;
        let b_ptr = b_contig.storage().ptr() as *const u8;
        let out_ptr = out.storage().ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_and_kernel(a_ptr, b_ptr, out_ptr, numel);
        }

        Ok(out)
    }

    fn logical_or(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate both tensors are U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }
        if b.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: b.dtype(),
            });
        }

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device);

        let a_ptr = a_contig.storage().ptr() as *const u8;
        let b_ptr = b_contig.storage().ptr() as *const u8;
        let out_ptr = out.storage().ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_or_kernel(a_ptr, b_ptr, out_ptr, numel);
        }

        Ok(out)
    }

    fn logical_xor(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate both tensors are U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }
        if b.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: b.dtype(),
            });
        }

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device);

        let a_ptr = a_contig.storage().ptr() as *const u8;
        let b_ptr = b_contig.storage().ptr() as *const u8;
        let out_ptr = out.storage().ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_xor_kernel(a_ptr, b_ptr, out_ptr, numel);
        }

        Ok(out)
    }

    fn logical_not(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        // Validate tensor is U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device);

        let a_ptr = a_contig.storage().ptr() as *const u8;
        let out_ptr = out.storage().ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_not_kernel(a_ptr, out_ptr, numel);
        }

        Ok(out)
    }
}
