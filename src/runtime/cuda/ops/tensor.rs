//! TensorOps implementation for CUDA runtime

use super::super::kernels::{
    AccumulationPrecision, launch_cast, launch_cat_copy, launch_cumprod, launch_cumprod_strided,
    launch_cumsum, launch_cumsum_strided, launch_elu, launch_embedding_lookup,
    launch_fill_with_f64, launch_gather, launch_gelu, launch_index_select, launch_isinf_op,
    launch_isnan_op, launch_layer_norm, launch_leaky_relu, launch_logsumexp,
    launch_logsumexp_strided, launch_masked_count, launch_masked_fill, launch_masked_prefix_sum,
    launch_masked_select, launch_pad, launch_relu, launch_repeat, launch_rms_norm, launch_roll,
    launch_scatter, launch_sigmoid, launch_silu, launch_softmax, launch_softmax_dim,
    launch_where_broadcast_op, launch_where_op,
};
use super::super::{CudaClient, CudaRuntime};
use super::helpers::{
    matmul_batched_native, matmul_native, native_binary_op, native_reduce_op, native_unary_op,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    ScalarOps, TensorOps, compute_reduce_strides, matmul_output_shape, normalize_softmax_dim,
    reduce_dim_output_shape, reduce_output_shape,
};
use crate::runtime::fallback::{compute_broadcast_shape, matmul_fallback, validate_binary_dtypes};
use crate::runtime::shape_ops;
use crate::runtime::{
    Runtime, compute_contiguous_strides, ensure_contiguous, validate_arange, validate_eye,
};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

impl TensorOps<CudaRuntime> for CudaClient {
    // ===== Binary Operations (Native CUDA Kernels) =====

    fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "div")
    }

    fn pow(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "max")
    }

    fn minimum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "min")
    }

    // ===== Unary Operations (Native CUDA Kernels) =====

    fn neg(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "neg")
    }

    fn abs(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "abs")
    }

    fn sqrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sqrt")
    }

    fn exp(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "exp")
    }

    fn log(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log")
    }

    fn sin(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sin")
    }

    fn cos(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "cos")
    }

    fn tan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "tan")
    }

    fn tanh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "tanh")
    }

    fn recip(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "recip")
    }

    fn square(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "square")
    }

    fn floor(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "floor")
    }

    fn ceil(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "ceil")
    }

    fn round(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "round")
    }

    fn sign(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sign")
    }

    fn isnan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        // Output is always U8 (boolean)
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_isnan_op(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn isinf(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        // Output is always U8 (boolean)
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_isinf_op(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    // ===== Matrix Operations (Native CUDA Kernels) =====

    fn matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_binary_dtypes(a, b)?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let k_b = if b_shape.len() >= 2 {
            b_shape[b_shape.len() - 2]
        } else {
            b_shape[b_shape.len() - 1]
        };
        if k != k_b {
            return Err(Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }

        let out_shape = matmul_output_shape(a_shape, b_shape).ok_or(Error::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        })?;

        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Native tiled CUDA kernel
        match dtype {
            DType::F32 | DType::F64 => {
                if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    matmul_native(self, a, b, dtype, m, k, n)
                }
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    matmul_native(self, a, b, dtype, m, k, n)
                }
            }
            _ => matmul_fallback(a, b, &out_shape, &self.device, "matmul"),
        }
    }

    // ===== Reductions (Native CUDA Kernels) =====

    fn sum(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "sum", dims, keepdim, None)
    }

    fn sum_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "sum", dims, keepdim, Some(precision))
    }

    fn mean(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        // Mean = sum / count
        // When dims is empty, reduce over all dimensions
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| a.shape()[d]).product()
        };

        // For empty dims, we need to reduce all dimensions
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..a.shape().len()).collect()
        } else {
            dims.to_vec()
        };

        let sum_result = self.sum(a, &actual_dims, keepdim)?;
        self.div_scalar(&sum_result, count as f64)
    }

    fn max(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "max", dims, keepdim, None)
    }

    fn max_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "max", dims, keepdim, Some(precision))
    }

    fn min(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "min", dims, keepdim, None)
    }

    fn min_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "min", dims, keepdim, Some(precision))
    }

    fn prod(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "prod", dims, keepdim, None)
    }

    fn prod_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "prod", dims, keepdim, Some(precision))
    }

    fn any(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "any", dims, keepdim, None)
    }

    fn all(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "all", dims, keepdim, None)
    }

    // ===== Activations (Native CUDA Kernels) =====

    fn relu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_relu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn sigmoid(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_sigmoid(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn silu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_silu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn gelu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_gelu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn leaky_relu(
        &self,
        a: &Tensor<CudaRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_leaky_relu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
                negative_slope as f32,
            )?;
        }

        Ok(out)
    }

    fn elu(&self, a: &Tensor<CudaRuntime>, alpha: f64) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_elu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
                alpha as f32,
            )?;
        }

        Ok(out)
    }

    fn softmax(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();

        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        let shape = a.shape();
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            if dim_idx == ndim - 1 {
                // Softmax over last dimension (optimized)
                launch_softmax(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    outer_size,
                    dim_size,
                )?;
            } else {
                // Softmax over non-last dimension
                launch_softmax_dim(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    outer_size,
                    dim_size,
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    // ===== Normalization =====

    fn rms_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device);

        unsafe {
            launch_rms_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.storage().ptr(),
                weight_contig.storage().ptr(),
                out.storage().ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    fn layer_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device);

        unsafe {
            launch_layer_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.storage().ptr(),
                weight_contig.storage().ptr(),
                bias_contig.storage().ptr(),
                out.storage().ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    // ===== Index Operations =====

    fn argmax(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_argmax_dim(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                reduce_size,
                inner_size,
            )?;
        }

        Ok(out)
    }

    fn argmin(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_argmin_dim(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                reduce_size,
                inner_size,
            )?;
        }

        Ok(out)
    }

    // ===== Indexing Operations =====

    fn gather(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate index dtype
        if index.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: index.dtype(),
            });
        }

        // Validate dimension
        let ndim = a.ndim();
        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Validate index tensor has same number of dimensions
        if index.ndim() != ndim {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: index.shape().to_vec(),
            });
        }

        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let index_contig = ensure_contiguous(index);

        // Output has same shape as index
        let out_shape = index.shape().to_vec();
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        // Prepare shape and stride arrays for GPU
        let input_shape: Vec<u32> = a.shape().iter().map(|&s| s as u32).collect();
        let input_strides: Vec<u32> = compute_contiguous_strides(a.shape())
            .iter()
            .map(|&s| s as u32)
            .collect();
        let output_shape: Vec<u32> = out_shape.iter().map(|&s| s as u32).collect();
        let output_strides: Vec<u32> = compute_contiguous_strides(&out_shape)
            .iter()
            .map(|&s| s as u32)
            .collect();

        // Allocate device memory for shape/stride arrays
        let shape_bytes = ndim * std::mem::size_of::<u32>();
        let input_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let input_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let output_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let output_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);

        // Copy shape/stride data to device
        let input_shape_bytes: &[u8] = bytemuck::cast_slice(&input_shape);
        let input_strides_bytes: &[u8] = bytemuck::cast_slice(&input_strides);
        let output_shape_bytes: &[u8] = bytemuck::cast_slice(&output_shape);
        let output_strides_bytes: &[u8] = bytemuck::cast_slice(&output_strides);

        CudaRuntime::copy_to_device(input_shape_bytes, input_shape_ptr, &self.device);
        CudaRuntime::copy_to_device(input_strides_bytes, input_strides_ptr, &self.device);
        CudaRuntime::copy_to_device(output_shape_bytes, output_shape_ptr, &self.device);
        CudaRuntime::copy_to_device(output_strides_bytes, output_strides_ptr, &self.device);

        let result = unsafe {
            launch_gather(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                index_contig.storage().ptr(),
                out.storage().ptr(),
                ndim,
                dim,
                input_shape_ptr,
                input_strides_ptr,
                output_shape_ptr,
                output_strides_ptr,
                out.numel(),
            )
        };

        // Clean up temporary device allocations
        CudaRuntime::deallocate(input_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(input_strides_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(output_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(output_strides_ptr, shape_bytes, &self.device);

        result?;
        Ok(out)
    }

    fn scatter(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
        src: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate index dtype
        if index.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: index.dtype(),
            });
        }

        // Validate dimension
        let ndim = a.ndim();
        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Validate src has same dtype as input
        let dtype = a.dtype();
        if src.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: src.dtype(),
            });
        }

        // Index and src must have same shape
        if index.shape() != src.shape() {
            return Err(Error::ShapeMismatch {
                expected: index.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let index_contig = ensure_contiguous(index);
        let src_contig = ensure_contiguous(src);

        // Output has same shape as input
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        // First, copy input to output (scatter modifies output in-place)
        unsafe {
            super::super::kernels::launch_copy(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        // Prepare shape and stride arrays for GPU
        let output_shape: Vec<u32> = a.shape().iter().map(|&s| s as u32).collect();
        let output_strides: Vec<u32> = compute_contiguous_strides(a.shape())
            .iter()
            .map(|&s| s as u32)
            .collect();
        let src_shape: Vec<u32> = src.shape().iter().map(|&s| s as u32).collect();
        let src_strides: Vec<u32> = compute_contiguous_strides(src.shape())
            .iter()
            .map(|&s| s as u32)
            .collect();

        // Allocate device memory for shape/stride arrays
        let shape_bytes = ndim * std::mem::size_of::<u32>();
        let output_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let output_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let src_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let src_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);

        // Copy shape/stride data to device
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&output_shape),
            output_shape_ptr,
            &self.device,
        );
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&output_strides),
            output_strides_ptr,
            &self.device,
        );
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&src_shape),
            src_shape_ptr,
            &self.device,
        );
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&src_strides),
            src_strides_ptr,
            &self.device,
        );

        let result = unsafe {
            launch_scatter(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                index_contig.storage().ptr(),
                src_contig.storage().ptr(),
                out.storage().ptr(),
                ndim,
                dim,
                output_shape_ptr,
                output_strides_ptr,
                src_shape_ptr,
                src_strides_ptr,
                src.numel(),
            )
        };

        // Clean up temporary device allocations
        CudaRuntime::deallocate(output_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(output_strides_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(src_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(src_strides_ptr, shape_bytes, &self.device);

        result?;
        Ok(out)
    }

    fn index_select(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate index dtype
        if index.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: index.dtype(),
            });
        }

        // Validate index is 1D
        if index.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![index.numel()],
                got: index.shape().to_vec(),
            });
        }

        // Validate dimension
        let shape = a.shape();
        let ndim = shape.len();
        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let index_contig = ensure_contiguous(index);

        // Compute output shape: same as input but dim[dim] = index.len()
        let index_len = index.numel();
        let mut out_shape = shape.to_vec();
        out_shape[dim] = index_len;

        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        // Compute outer/dim/inner sizes
        let outer_size: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            launch_index_select(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                index_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                dim_size,
                inner_size,
                index_len,
            )?;
        }

        Ok(out)
    }

    fn masked_select(
        &self,
        a: &Tensor<CudaRuntime>,
        mask: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate mask dtype
        if mask.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: mask.dtype(),
            });
        }

        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let mask_contig = ensure_contiguous(mask);
        let numel = a.numel();

        // Both tensors must have same shape (or mask must broadcast to a's shape)
        // For simplicity, require same shape for now
        if a.shape() != mask.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: mask.shape().to_vec(),
            });
        }

        // Phase 1: Count true elements in mask
        let count_bytes = std::mem::size_of::<u32>();
        let count_ptr = CudaRuntime::allocate(count_bytes, &self.device);

        // Initialize count to 0
        let zero: u32 = 0;
        CudaRuntime::copy_to_device(bytemuck::bytes_of(&zero), count_ptr, &self.device);

        unsafe {
            launch_masked_count(
                &self.context,
                &self.stream,
                self.device.index,
                mask_contig.storage().ptr(),
                count_ptr,
                numel,
            )?;
        }

        // Read count back to host
        let mut count_buf = [0u32; 1];
        CudaRuntime::copy_from_device(
            count_ptr,
            bytemuck::bytes_of_mut(&mut count_buf),
            &self.device,
        );
        let count = count_buf[0] as usize;

        CudaRuntime::deallocate(count_ptr, count_bytes, &self.device);

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(&[count], dtype, &self.device);

        if count == 0 {
            return Ok(out);
        }

        // Phase 2: Compute prefix sum
        let prefix_sum_bytes = numel * std::mem::size_of::<u32>();
        let prefix_sum_ptr = CudaRuntime::allocate(prefix_sum_bytes, &self.device);

        unsafe {
            launch_masked_prefix_sum(
                &self.context,
                &self.stream,
                self.device.index,
                mask_contig.storage().ptr(),
                prefix_sum_ptr,
                numel,
            )?;
        }

        // Phase 3: Gather selected elements
        unsafe {
            launch_masked_select(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                mask_contig.storage().ptr(),
                out.storage().ptr(),
                prefix_sum_ptr,
                numel,
            )?;
        }

        CudaRuntime::deallocate(prefix_sum_ptr, prefix_sum_bytes, &self.device);

        Ok(out)
    }

    fn masked_fill(
        &self,
        a: &Tensor<CudaRuntime>,
        mask: &Tensor<CudaRuntime>,
        value: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate mask dtype
        if mask.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: mask.dtype(),
            });
        }

        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let mask_contig = ensure_contiguous(mask);

        // Both tensors must have same shape (or mask must broadcast to a's shape)
        // For simplicity, require same shape for now
        if a.shape() != mask.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: mask.shape().to_vec(),
            });
        }

        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_masked_fill(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                mask_contig.storage().ptr(),
                out.storage().ptr(),
                value,
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn embedding_lookup(
        &self,
        embeddings: &Tensor<CudaRuntime>,
        indices: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = embeddings.dtype();
        let emb_shape = embeddings.shape();

        // Validate embeddings is 2D
        if emb_shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                expected: vec![0, 0], // Indicates 2D expected
                got: emb_shape.to_vec(),
            });
        }

        // Validate indices dtype
        if indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: indices.dtype(),
            });
        }

        let vocab_size = emb_shape[0];
        let embedding_dim = emb_shape[1];
        let num_indices = indices.numel();

        // Output shape: indices.shape() + [embedding_dim]
        let mut out_shape = indices.shape().to_vec();
        out_shape.push(embedding_dim);

        let emb_contig = ensure_contiguous(embeddings);
        let idx_contig = ensure_contiguous(indices);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        unsafe {
            launch_embedding_lookup(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                emb_contig.storage().ptr(),
                idx_contig.storage().ptr(),
                out.storage().ptr(),
                num_indices,
                vocab_size,
                embedding_dim,
            )?;
        }

        Ok(out)
    }

    // ===== Type Casting =====

    fn cast(&self, a: &Tensor<CudaRuntime>, target_dtype: DType) -> Result<Tensor<CudaRuntime>> {
        let src_dtype = a.dtype();

        // No-op if types match
        if src_dtype == target_dtype {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(shape, target_dtype, &self.device);

        unsafe {
            launch_cast(
                &self.context,
                &self.stream,
                self.device.index,
                src_dtype,
                target_dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    // ===== Conditional Operations =====

    fn where_cond(
        &self,
        cond: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate that x and y have the same dtype
        let dtype = validate_binary_dtypes(x, y)?;

        // Validate condition tensor is U8 (boolean)
        if cond.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: cond.dtype(),
            });
        }

        // For same shapes, use optimized element-wise kernel on GPU
        if cond.shape() == x.shape() && x.shape() == y.shape() {
            let cond_contig = ensure_contiguous(cond);
            let x_contig = ensure_contiguous(x);
            let y_contig = ensure_contiguous(y);
            let out = Tensor::<CudaRuntime>::empty(x.shape(), dtype, &self.device);

            unsafe {
                launch_where_op(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    cond_contig.storage().ptr(),
                    x_contig.storage().ptr(),
                    y_contig.storage().ptr(),
                    out.storage().ptr(),
                    out.numel(),
                )?;
            }

            return Ok(out);
        }

        // For different shapes, use the broadcast kernel (stays on GPU)
        // Compute broadcast shape for all three tensors
        let xy_shape = compute_broadcast_shape(x, y)?;
        let out_shape = crate::ops::broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| {
            Error::BroadcastError {
                lhs: cond.shape().to_vec(),
                rhs: xy_shape.clone(),
            }
        })?;

        let cond_contig = ensure_contiguous(cond);
        let x_contig = ensure_contiguous(x);
        let y_contig = ensure_contiguous(y);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        unsafe {
            launch_where_broadcast_op(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                dtype,
                cond_contig.storage().ptr(),
                x_contig.storage().ptr(),
                y_contig.storage().ptr(),
                out.storage().ptr(),
                cond.shape(),
                x.shape(),
                y.shape(),
                &out_shape,
            )?;
        }

        Ok(out)
    }

    // ===== Utility Operations =====

    fn clamp(
        &self,
        a: &Tensor<CudaRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use native CUDA implementation via composition of maximum and minimum
        // clamp(x, min, max) = min(max(x, min), max)
        // This approach uses existing optimized kernels

        // Create scalar tensors for min and max
        let min_scalar = self.fill(&[], min_val, a.dtype())?;
        let max_scalar = self.fill(&[], max_val, a.dtype())?;

        // First: max(x, min_val)
        let clamped_low = self.maximum(a, &min_scalar)?;

        // Then: min(result, max_val)
        self.minimum(&clamped_low, &max_scalar)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<CudaRuntime>> {
        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        // Launch native CUDA fill kernel
        unsafe {
            launch_fill_with_f64(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                value,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn arange(
        &self,
        start: f64,
        stop: f64,
        step: f64,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use shared validation
        let numel = validate_arange(start, stop, step)?;

        // Handle empty tensor case
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(&[0], dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(&[numel], dtype, &self.device);

        unsafe {
            super::super::kernels::launch_arange(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                start,
                step,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn linspace(
        &self,
        start: f64,
        stop: f64,
        steps: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        // linspace supports all numeric dtypes - computation is done in higher precision,
        // then converted to the output dtype. This matches NumPy behavior.

        // Handle edge cases
        if steps == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(&[0], dtype, &self.device));
        }

        if steps == 1 {
            return self.fill(&[1], start, dtype);
        }

        let out = Tensor::<CudaRuntime>::empty(&[steps], dtype, &self.device);

        unsafe {
            super::super::kernels::launch_linspace(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                start,
                stop,
                out.storage().ptr(),
                steps,
            )?;
        }

        Ok(out)
    }

    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // Use shared validation
        let (rows, cols) = validate_eye(n, m);

        // Handle edge cases
        if rows == 0 || cols == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(
                &[rows, cols],
                dtype,
                &self.device,
            ));
        }

        let out = Tensor::<CudaRuntime>::empty(&[rows, cols], dtype, &self.device);

        unsafe {
            super::super::kernels::launch_eye(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                rows,
                cols,
                out.storage().ptr(),
            )?;
        }

        Ok(out)
    }

    // ===== Statistical Operations =====

    fn var(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Variance implementation using existing ops
        // var(x) = mean((x - mean(x))^2) * N / (N - correction)

        let shape = a.shape();

        // When dims is empty, reduce over all dimensions
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..shape.len()).collect()
        } else {
            dims.to_vec()
        };

        // Compute count of elements being reduced
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| shape[d]).product()
        };

        // Compute mean (mean already handles empty dims internally)
        let mean_val = self.mean(a, dims, true)?;

        // Compute (x - mean)
        let diff = self.sub(a, &mean_val)?;

        // Compute (x - mean)^2
        let diff_squared = self.square(&diff)?;

        // Compute sum of squared differences over all dims when dims is empty
        let sum_sq = self.sum(&diff_squared, &actual_dims, keepdim)?;

        // Divide by (N - correction)
        let divisor = (count.saturating_sub(correction)).max(1) as f64;
        self.div_scalar(&sum_sq, divisor)
    }

    fn std(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Standard deviation is sqrt of variance
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    // ===== Cumulative Operations =====

    fn cumsum(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let shape = a.shape();
        let ndim = shape.len();

        // Normalize dimension (handle negative indexing)
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Handle empty tensor
        if a.numel() == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device));
        }

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a);

        // Calculate dimensions for kernel launch
        let scan_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device);

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Scan along last dimension or effectively contiguous
            let outer = outer_size.max(1);
            unsafe {
                launch_cumsum(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer,
                )?;
            }
        } else {
            // Strided scan for non-last dimension
            unsafe {
                launch_cumsum_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer_size.max(1),
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    fn cumprod(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let shape = a.shape();
        let ndim = shape.len();

        // Normalize dimension (handle negative indexing)
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Handle empty tensor
        if a.numel() == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device));
        }

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a);

        // Calculate dimensions for kernel launch
        let scan_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device);

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Scan along last dimension or effectively contiguous
            let outer = outer_size.max(1);
            unsafe {
                launch_cumprod(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer,
                )?;
            }
        } else {
            // Strided scan for non-last dimension
            unsafe {
                launch_cumprod_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer_size.max(1),
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    fn logsumexp(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        // Only support floating point types
        if !matches!(a.dtype(), DType::F32 | DType::F64) {
            return Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "logsumexp",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();

        // Handle empty dims (reduce over all dimensions)
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..ndim).collect()
        } else {
            dims.to_vec()
        };

        // Validate dimensions
        for &dim in &actual_dims {
            if dim >= ndim {
                return Err(Error::InvalidDimension {
                    dim: dim as isize,
                    ndim,
                });
            }
        }

        // Handle empty tensor
        if a.numel() == 0 {
            let out_shape = reduce_output_shape(shape, &actual_dims, keepdim);
            return Ok(Tensor::<CudaRuntime>::empty(
                &out_shape,
                a.dtype(),
                &self.device,
            ));
        }

        // For multi-dimensional reduction, reduce one dimension at a time
        if actual_dims.len() > 1 {
            let mut result = a.clone();
            // Sort dims in descending order to avoid index invalidation
            let mut sorted_dims = actual_dims.clone();
            sorted_dims.sort_by(|a, b| b.cmp(a));

            for &dim in &sorted_dims {
                result = self.logsumexp(&result, &[dim], true)?;
            }

            // Remove keepdim if not requested
            if !keepdim {
                let out_shape = reduce_output_shape(shape, &actual_dims, false);
                result = result.reshape(&out_shape)?;
            }

            return Ok(result);
        }

        // Single dimension reduction
        let dim = actual_dims[0];

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a);

        // Calculate dimensions for kernel launch
        let reduce_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Calculate output shape
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);
        let out_numel: usize = out_shape.iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(&out_shape, a.dtype(), &self.device);

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Reduction along last dimension
            let outer = outer_size.max(1);
            unsafe {
                launch_logsumexp(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    reduce_size,
                    outer,
                )?;
            }
        } else {
            // Strided reduction for non-last dimension
            unsafe {
                launch_logsumexp_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    reduce_size,
                    outer_size.max(1),
                    inner_size,
                )?;
            }
        }

        // Handle keepdim reshape if needed
        if keepdim && out.numel() == out_numel {
            Ok(out)
        } else {
            Ok(out)
        }
    }

    // ===== Random Operations =====

    fn rand(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // Supported: F32, F64, F16, BF16
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "rand" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();

        // Launch native CUDA rand kernel
        unsafe {
            super::super::kernels::launch_rand(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn randn(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // Supported: F32, F64, F16, BF16
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "randn" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();

        // Launch native CUDA randn kernel (uses Box-Muller transform)
        unsafe {
            super::super::kernels::launch_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn randint(
        &self,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate dtype is integer
        if !dtype.is_int() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "randint",
            });
        }

        // Validate range
        if high <= low {
            return Err(Error::InvalidArgument {
                arg: "high",
                reason: format!(
                    "randint requires high > low, got low={}, high={}",
                    low, high
                ),
            });
        }

        // Validate range fits in unsigned dtype
        if dtype.is_unsigned_int() && low < 0 {
            return Err(Error::InvalidArgument {
                arg: "low",
                reason: format!(
                    "randint with unsigned dtype {} requires low >= 0, got low={}",
                    dtype, low
                ),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();
        let range = high - low;

        // Launch native CUDA randint kernel
        unsafe {
            super::super::kernels::launch_randint(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                low,
                range,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    // ===== Shape Operations =====

    fn cat(&self, tensors: &[&Tensor<CudaRuntime>], dim: isize) -> Result<Tensor<CudaRuntime>> {
        let params = crate::runtime::shape_ops::validate_cat(tensors, dim)?;

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(&params.out_shape, params.dtype, &self.device);

        // Copy data from each tensor using CUDA kernel
        let mut cat_offset = 0usize;
        for &tensor in tensors {
            let tensor_contig = ensure_contiguous(tensor);
            let src_cat_size = tensor.shape()[params.dim_idx];

            unsafe {
                launch_cat_copy(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    params.dtype,
                    tensor_contig.storage().ptr(),
                    out.storage().ptr(),
                    params.outer_size,
                    src_cat_size,
                    params.cat_dim_total,
                    cat_offset,
                    params.inner_size,
                )?;
            }

            cat_offset += src_cat_size;
        }

        Ok(out)
    }

    fn stack(&self, tensors: &[&Tensor<CudaRuntime>], dim: isize) -> Result<Tensor<CudaRuntime>> {
        // Validate tensors and get normalized dimension
        let _ = crate::runtime::shape_ops::validate_stack(tensors, dim)?;

        // stack(tensors, dim) = cat([t.unsqueeze(dim) for t in tensors], dim)
        let unsqueezed: Vec<Tensor<CudaRuntime>> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<_>>()?;

        let refs: Vec<&Tensor<CudaRuntime>> = unsqueezed.iter().collect();
        self.cat(&refs, dim)
    }

    fn split(
        &self,
        tensor: &Tensor<CudaRuntime>,
        split_size: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<CudaRuntime>>> {
        shape_ops::split_impl(tensor, split_size, dim)
    }

    fn chunk(
        &self,
        tensor: &Tensor<CudaRuntime>,
        chunks: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<CudaRuntime>>> {
        shape_ops::chunk_impl(tensor, chunks, dim)
    }

    fn repeat(
        &self,
        tensor: &Tensor<CudaRuntime>,
        repeats: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        let params = shape_ops::validate_repeat(tensor, repeats)?;

        // Handle no-op case (all repeats are 1)
        if repeats.iter().all(|&r| r == 1) {
            return Ok(tensor.contiguous());
        }

        let tensor_contig = ensure_contiguous(tensor);
        let out = Tensor::<CudaRuntime>::empty(&params.out_shape, tensor.dtype(), &self.device);

        unsafe {
            launch_repeat(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                tensor.dtype(),
                tensor_contig.storage().ptr(),
                out.storage().ptr(),
                tensor.shape(),
                &params.out_shape,
            )?;
        }

        Ok(out)
    }

    fn pad(
        &self,
        tensor: &Tensor<CudaRuntime>,
        padding: &[usize],
        value: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let params = shape_ops::validate_pad(tensor, padding)?;

        // Handle no-op case (all padding is zero)
        if params.pad_per_dim.iter().all(|&(b, a)| b == 0 && a == 0) {
            return Ok(tensor.contiguous());
        }

        let tensor_contig = ensure_contiguous(tensor);
        let out = Tensor::<CudaRuntime>::empty(&params.out_shape, tensor.dtype(), &self.device);

        // Extract pad_before from pad_per_dim
        let pad_before: Vec<usize> = params.pad_per_dim.iter().map(|(b, _)| *b).collect();

        unsafe {
            launch_pad(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                tensor.dtype(),
                tensor_contig.storage().ptr(),
                out.storage().ptr(),
                value,
                tensor.shape(),
                &params.out_shape,
                &pad_before,
            )?;
        }

        Ok(out)
    }

    fn roll(
        &self,
        tensor: &Tensor<CudaRuntime>,
        shift: isize,
        dim: isize,
    ) -> Result<Tensor<CudaRuntime>> {
        let params = shape_ops::validate_roll(tensor, shift, dim)?;

        // Handle no-op case (shift is 0 or multiple of dim_size)
        if params.shift == 0 {
            return Ok(tensor.contiguous());
        }

        let tensor_contig = ensure_contiguous(tensor);
        let out = Tensor::<CudaRuntime>::empty(tensor.shape(), tensor.dtype(), &self.device);

        // Compute outer/inner sizes
        let outer_size: usize = tensor.shape()[..params.dim_idx].iter().product();
        let inner_size: usize = tensor.shape()[params.dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            launch_roll(
                &self.context,
                &self.stream,
                self.device.index,
                tensor.dtype(),
                tensor_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                params.dim_size,
                inner_size,
                params.shift,
            )?;
        }

        Ok(out)
    }
}

// ============================================================================
// Random Seed Generation
// ============================================================================

/// Global atomic counter for generating unique seeds
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a random seed combining atomic counter and system time.
///
/// This provides good entropy for parallel random number generation:
/// - Atomic counter ensures uniqueness across calls
/// - System time adds unpredictability
#[inline]
fn generate_random_seed() -> u64 {
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time_component = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Combine counter and time using splitmix64-style mixing
    let mut z = counter.wrapping_add(time_component);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
