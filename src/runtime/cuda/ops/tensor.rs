//! TensorOps implementation for CUDA runtime

// Note: Some kernel launchers are imported but accessed via full path for clarity.
// The imports are kept for future use and consistency.
#[allow(unused_imports)]
use super::super::kernels::{
    AccumulationPrecision, launch_angle, launch_angle_real, launch_argsort, launch_cast,
    launch_cat_copy, launch_conj, launch_count_nonzero, launch_count_unique, launch_cumprod,
    launch_cumprod_strided, launch_cumsum, launch_cumsum_strided, launch_elu,
    launch_embedding_lookup, launch_extract_unique, launch_fill_with_f64,
    launch_flat_to_multi_index, launch_gather, launch_gather_nonzero, launch_gelu, launch_imag,
    launch_index_put, launch_index_select, launch_isinf_op, launch_isnan_op, launch_layer_norm,
    launch_leaky_relu, launch_logsumexp, launch_logsumexp_strided, launch_masked_count,
    launch_masked_fill, launch_masked_prefix_sum, launch_masked_select, launch_pad, launch_real,
    launch_relu, launch_repeat, launch_rms_norm, launch_roll, launch_scatter, launch_searchsorted,
    launch_sigmoid, launch_silu, launch_softmax, launch_softmax_dim, launch_sort,
    launch_sort_values_only, launch_topk, launch_validate_indices,
    launch_where_broadcast_generic_op, launch_where_broadcast_op, launch_where_generic_op,
    launch_where_op,
};
use super::super::{CudaClient, CudaRuntime};
use super::helpers::{
    matmul_batched_native, matmul_bias_batched_native, matmul_bias_native, matmul_native,
    native_binary_op, native_reduce_op, native_unary_op,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    ActivationOps, ComplexOps, ConditionalOps, CumulativeOps, IndexingOps, LinalgOps, MatmulOps,
    NormalizationOps, RandomOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, StatisticalOps,
    TensorOps, TypeConversionOps, UtilityOps, compute_reduce_strides, matmul_bias_output_shape,
    matmul_output_shape, normalize_softmax_dim, reduce_dim_output_shape, reduce_output_shape,
};
use crate::runtime::fallback::{compute_broadcast_shape, matmul_fallback, validate_binary_dtypes};
use crate::runtime::shape_ops;
use crate::runtime::{
    Runtime, compute_contiguous_strides, ensure_contiguous, normalize_dim, validate_arange,
    validate_eye,
};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

impl TypeConversionOps<CudaRuntime> for CudaClient {
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
}

// ============================================================================
// ComplexOps Implementation
// ============================================================================

impl ComplexOps<CudaRuntime> for CudaClient {
    fn conj(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // For real types, conjugate is identity
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_conj(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn real(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // For real types, return copy
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        // Determine output dtype: Complex64 → F32, Complex128 → F64
        let out_dtype = match dtype {
            DType::Complex64 => DType::F32,
            DType::Complex128 => DType::F64,
            _ => return Err(Error::UnsupportedDType { dtype, op: "real" }),
        };

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device);

        unsafe {
            launch_real(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn imag(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // Determine output dtype
        let out_dtype = if dtype.is_complex() {
            match dtype {
                DType::Complex64 => DType::F32,
                DType::Complex128 => DType::F64,
                _ => return Err(Error::UnsupportedDType { dtype, op: "imag" }),
            }
        } else {
            // For real types, return zeros with same dtype
            dtype
        };

        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device);

        // For real types, fill with zeros
        if !dtype.is_complex() {
            unsafe {
                super::super::kernels::launch_fill_with_f64(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    out_dtype,
                    0.0,
                    out.storage().ptr(),
                    out.numel(),
                )?;
            }
            return Ok(out);
        }

        // For complex types, extract imaginary part
        let a_contig = ensure_contiguous(a);

        unsafe {
            launch_imag(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn angle(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // Determine output dtype
        let out_dtype = if dtype.is_complex() {
            match dtype {
                DType::Complex64 => DType::F32,
                DType::Complex128 => DType::F64,
                _ => return Err(Error::UnsupportedDType { dtype, op: "angle" }),
            }
        } else {
            // For real types, return zeros with same dtype
            dtype
        };

        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device);
        let a_contig = ensure_contiguous(a);

        // For real types: angle(x) = 0 if x >= 0, π if x < 0
        if !dtype.is_complex() {
            match dtype {
                DType::F32 | DType::F64 => unsafe {
                    super::super::kernels::launch_angle_real(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        dtype,
                        a_contig.storage().ptr(),
                        out.storage().ptr(),
                        a.numel(),
                    )?;
                },
                _ => {
                    // For integer types, return zeros (π as integer doesn't make mathematical sense)
                    unsafe {
                        super::super::kernels::launch_fill_with_f64(
                            &self.context,
                            &self.stream,
                            self.device.index,
                            out_dtype,
                            0.0,
                            out.storage().ptr(),
                            out.numel(),
                        )?;
                    }
                }
            }
            return Ok(out);
        }

        // For complex types, compute phase angle

        unsafe {
            launch_angle(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }
}

impl NormalizationOps<CudaRuntime> for CudaClient {
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
}

// ============================================================================
// MatmulOps Implementation
// ============================================================================

impl MatmulOps<CudaRuntime> for CudaClient {
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

    fn matmul_bias(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::ops::validate_matmul_bias_dtypes;

        // Validate dtypes using unified helper (ensures consistent error handling across backends)
        let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

        // Validate bias is 1D
        if bias.shape().len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("bias must be 1D tensor, got shape {:?}", bias.shape()),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let bias_shape = bias.shape();

        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Validate inner dimensions
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

        // Validate bias length matches N
        if bias_shape[0] != n {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!(
                    "bias length {} must match output columns {}",
                    bias_shape[0], n
                ),
            });
        }

        let out_shape =
            matmul_bias_output_shape(a_shape, b_shape, bias_shape).ok_or(Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            })?;

        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Native tiled CUDA kernel with fused bias
        match dtype {
            DType::F32 | DType::F64 => {
                if batch_size > 1 {
                    matmul_bias_batched_native(self, a, b, bias, dtype, batch_size, m, k, n)
                } else {
                    matmul_bias_native(self, a, b, bias, dtype, m, k, n)
                }
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                if batch_size > 1 {
                    matmul_bias_batched_native(self, a, b, bias, dtype, batch_size, m, k, n)
                } else {
                    matmul_bias_native(self, a, b, bias, dtype, m, k, n)
                }
            }
            _ => {
                // For unsupported dtypes, return error instead of silent fallback
                // (matmul_bias requires fused kernel for efficiency - non-fused defeats the purpose)
                Err(Error::UnsupportedDType {
                    dtype,
                    op: "matmul_bias",
                })
            }
        }
    }
}

// ============================================================================
// CumulativeOps Implementation
// ============================================================================

impl CumulativeOps<CudaRuntime> for CudaClient {
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
}

// ============================================================================
// ActivationOps Implementation
// ============================================================================

impl ActivationOps<CudaRuntime> for CudaClient {
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
}

// ============================================================================
// TensorOps Implementation
// ============================================================================

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

    fn atan2(
        &self,
        y: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, y, x, "atan2")
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

    fn atan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "atan")
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

    fn rsqrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "rsqrt")
    }

    fn cbrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "cbrt")
    }

    fn exp2(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "exp2")
    }

    fn expm1(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "expm1")
    }

    fn log2(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log2")
    }

    fn log10(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log10")
    }

    fn log1p(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log1p")
    }

    fn asin(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "asin")
    }

    fn acos(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "acos")
    }

    fn sinh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sinh")
    }

    fn cosh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "cosh")
    }

    fn asinh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "asinh")
    }

    fn acosh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "acosh")
    }

    fn atanh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "atanh")
    }

    fn trunc(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "trunc")
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

    // ===== Reductions (Native CUDA Kernels) =====
    // Moved to ReduceOps trait implementation below

    // ===== Index Operations =====
    // Moved to IndexingOps trait implementation below
}

// ============================================================================
// IndexingOps Implementation
// ============================================================================

/// IndexingOps implementation for CUDA runtime.
impl IndexingOps<CudaRuntime> for CudaClient {
    // Index operations methods go here
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

        // Compute dim_size for validation
        let dim_size = shape[dim];

        // Validate indices on GPU (only costs copying 4 bytes back)
        let error_count_tensor = Tensor::<CudaRuntime>::empty(&[1], DType::U32, &self.device);
        unsafe {
            // Initialize error count to 0
            launch_fill_with_f64(
                &self.context,
                &self.stream,
                self.device.index,
                DType::U32,
                0.0,
                error_count_tensor.storage().ptr(),
                1,
            )?;

            // Run validation kernel
            launch_validate_indices(
                &self.context,
                &self.stream,
                self.device.index,
                index_contig.storage().ptr(),
                error_count_tensor.storage().ptr(),
                index_len,
                dim_size,
            )?;
        }

        // Check validation result
        let error_count = error_count_tensor.to_vec::<u32>()[0];
        if error_count > 0 {
            return Err(Error::IndexOutOfBounds {
                index: 0, // We don't know which specific index failed
                size: dim_size,
            });
        }

        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        // Compute outer/dim/inner sizes
        let outer_size: usize = shape[..dim].iter().product();
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

    fn index_put(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
        src: &Tensor<CudaRuntime>,
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

        // Validate src dtype matches
        if src.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: src.dtype(),
            });
        }

        let index_len = index.numel();

        // Validate src shape: must match a's shape except at dim where it equals index_len
        let mut expected_src_shape = shape.to_vec();
        expected_src_shape[dim] = index_len;
        if src.shape() != expected_src_shape {
            return Err(Error::ShapeMismatch {
                expected: expected_src_shape,
                got: src.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let index_contig = ensure_contiguous(index);
        let src_contig = ensure_contiguous(src);

        // Compute dim_size for validation
        let dim_size = shape[dim];

        // Validate indices on GPU (only costs copying 4 bytes back)
        let error_count_tensor = Tensor::<CudaRuntime>::empty(&[1], DType::U32, &self.device);
        unsafe {
            // Initialize error count to 0
            launch_fill_with_f64(
                &self.context,
                &self.stream,
                self.device.index,
                DType::U32,
                0.0,
                error_count_tensor.storage().ptr(),
                1,
            )?;

            // Run validation kernel
            launch_validate_indices(
                &self.context,
                &self.stream,
                self.device.index,
                index_contig.storage().ptr(),
                error_count_tensor.storage().ptr(),
                index_len,
                dim_size,
            )?;
        }

        // Check validation result
        let error_count = error_count_tensor.to_vec::<u32>()[0];
        if error_count > 0 {
            return Err(Error::IndexOutOfBounds {
                index: 0, // We don't know which specific index failed
                size: dim_size,
            });
        }

        // Clone a to output first
        let out = a_contig.clone();

        // Compute outer/dim/inner sizes
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            launch_index_put(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                index_contig.storage().ptr(),
                src_contig.storage().ptr(),
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
}

// ============================================================================
// StatisticalOps Implementation
// ============================================================================

impl StatisticalOps<CudaRuntime> for CudaClient {
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

    fn quantile(
        &self,
        a: &Tensor<CudaRuntime>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<CudaRuntime>> {
        super::statistics::quantile_impl(self, a, q, dim, keepdim, interpolation)
    }

    fn percentile(
        &self,
        a: &Tensor<CudaRuntime>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        super::statistics::percentile_impl(self, a, p, dim, keepdim)
    }

    fn median(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        super::statistics::median_impl(self, a, dim, keepdim)
    }

    fn histogram(
        &self,
        a: &Tensor<CudaRuntime>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        super::statistics::histogram_impl(self, a, bins, range)
    }

    fn cov(&self, a: &Tensor<CudaRuntime>, ddof: Option<usize>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CudaRuntime>>::cov(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CudaRuntime>>::corrcoef(self, a)
    }

    fn skew(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        super::statistics::skew_impl(self, a, dims, keepdim, correction)
    }

    fn kurtosis(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        super::statistics::kurtosis_impl(self, a, dims, keepdim, correction)
    }

    fn mode(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        super::statistics::mode_impl(self, a, dim, keepdim)
    }

    // ===== Random Operations =====
    // Moved to RandomOps trait implementation below
}

// ============================================================================
// RandomOps Implementation
// ============================================================================

/// RandomOps implementation for CUDA runtime.
impl RandomOps<CudaRuntime> for CudaClient {
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

    fn multinomial(
        &self,
        probs: &Tensor<CudaRuntime>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = probs.dtype();

        // Validate probs is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "multinomial",
            });
        }

        // Validate num_samples
        if num_samples == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: "num_samples must be > 0".to_string(),
            });
        }

        let shape = probs.shape();
        if shape.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: "probs tensor must have at least 1 dimension".to_string(),
            });
        }

        let num_categories = *shape.last().unwrap();
        if num_categories == 0 {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: "probs tensor must have at least 1 category (last dim > 0)".to_string(),
            });
        }

        // Without replacement: can't sample more than we have
        if !replacement && num_samples > num_categories {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: format!(
                    "cannot sample {} items without replacement from {} categories",
                    num_samples, num_categories
                ),
            });
        }

        // Compute number of distributions (product of all dims except last)
        let num_distributions: usize = shape[..shape.len() - 1].iter().product();
        let num_distributions = num_distributions.max(1); // At least 1 for 1D input

        // Ensure probs is contiguous
        let probs = ensure_contiguous(probs);

        // Output shape: [..., num_samples]
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(num_samples);
        if out_shape.is_empty() {
            out_shape.push(num_samples);
        }

        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        // Generate seed
        let seed = generate_random_seed();

        // Launch CUDA kernel
        unsafe {
            if replacement {
                super::super::kernels::launch_multinomial_with_replacement(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    probs.storage().ptr(),
                    out.storage().ptr(),
                    seed,
                    num_distributions,
                    num_categories,
                    num_samples,
                )?;
            } else {
                super::super::kernels::launch_multinomial_without_replacement(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    probs.storage().ptr(),
                    out.storage().ptr(),
                    seed,
                    num_distributions,
                    num_categories,
                    num_samples,
                )?;
            }
        }

        Ok(out)
    }

    fn bernoulli(&self, p: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bernoulli",
            });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidArgument {
                arg: "p",
                reason: format!("bernoulli requires p in [0, 1], got {}", p),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_bernoulli(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                p,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn beta(
        &self,
        alpha: f64,
        beta: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "beta" });
        }
        if alpha <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "alpha",
                reason: format!("beta requires alpha > 0, got {}", alpha),
            });
        }
        if beta <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "beta",
                reason: format!("beta requires beta > 0, got {}", beta),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_beta_dist(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                alpha,
                beta,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "gamma" });
        }
        if shape_param <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "shape_param",
                reason: format!("gamma requires shape_param > 0, got {}", shape_param),
            });
        }
        if scale <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "scale",
                reason: format!("gamma requires scale > 0, got {}", scale),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_gamma_dist(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                shape_param,
                scale,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn exponential(&self, rate: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "exponential",
            });
        }
        if rate <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "rate",
                reason: format!("exponential requires rate > 0, got {}", rate),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_exponential(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                rate,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn poisson(&self, lambda: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "poisson",
            });
        }
        if lambda <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "lambda",
                reason: format!("poisson requires lambda > 0, got {}", lambda),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_poisson(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                lambda,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn binomial(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "binomial",
            });
        }
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: "binomial requires n > 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidArgument {
                arg: "p",
                reason: format!("binomial requires p in [0, 1], got {}", p),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_binomial(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                n,
                p,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn laplace(
        &self,
        loc: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "laplace",
            });
        }
        if scale <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "scale",
                reason: format!("laplace requires scale > 0, got {}", scale),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_laplace(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                loc,
                scale,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn chi_squared(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "chi_squared",
            });
        }
        if df <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df",
                reason: format!("chi_squared requires df > 0, got {}", df),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_chi_squared(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                df,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn student_t(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "student_t",
            });
        }
        if df <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df",
                reason: format!("student_t requires df > 0, got {}", df),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_student_t(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                df,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn f_distribution(
        &self,
        df1: f64,
        df2: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "f_distribution",
            });
        }
        if df1 <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df1",
                reason: format!("f_distribution requires df1 > 0, got {}", df1),
            });
        }
        if df2 <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df2",
                reason: format!("f_distribution requires df2 > 0, got {}", df2),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let seed = generate_random_seed();

        unsafe {
            super::super::kernels::launch_f_distribution(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                df1,
                df2,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }
}

// ============================================================================
// Continuing TensorOps Implementation
// ============================================================================

impl TensorOps<CudaRuntime> for CudaClient {
    // ===== Shape Operations =====
    // Moved to ShapeOps trait implementation below

    // ===== Linear Algebra Operations =====
    // Moved to LinalgOps trait implementation below

    // ===== Complex Number Operations =====
    // Moved to ComplexOps trait in ops/traits/complex.rs

    // ===== Sorting and Search Operations =====
    // Moved to SortingOps trait implementation below
}

// ============================================================================
// LinalgOps Implementation
// ============================================================================

/// LinalgOps implementation for CUDA runtime.
impl LinalgOps<CudaRuntime> for CudaClient {
    fn solve(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::solve(self, a, b)
    }

    fn lstsq(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::lstsq(self, a, b)
    }

    fn pinverse(&self, a: &Tensor<CudaRuntime>, rcond: Option<f64>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::pinverse(self, a, rcond)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<CudaRuntime>,
        ord: crate::algorithm::linalg::MatrixNormOrder,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_norm(self, a, ord)
    }

    fn inverse(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::inverse(self, a)
    }

    fn det(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::det(self, a)
    }

    fn trace(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::trace(self, a)
    }

    fn diag(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diag(self, a)
    }

    fn diagflat(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diagflat(self, a)
    }

    fn matrix_rank(
        &self,
        a: &Tensor<CudaRuntime>,
        tol: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_rank(self, a, tol)
    }
}

// ============================================================================
// ReduceOps Implementation
// ============================================================================

/// ReduceOps implementation for CUDA runtime.
impl ReduceOps<CudaRuntime> for CudaClient {
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
}

// ============================================================================
// ShapeOps Implementation
// ============================================================================

/// ShapeOps implementation for CUDA runtime.
impl ShapeOps<CudaRuntime> for CudaClient {
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
// SortingOps Implementation
// ============================================================================

/// SortingOps implementation for CUDA runtime.
impl SortingOps<CudaRuntime> for CudaClient {
    fn sort(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(a.clone());
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        unsafe {
            super::super::kernels::launch_sort_values_only(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                sort_size,
                inner_size,
                descending,
            )?;
        }

        Ok(out)
    }

    fn sort_with_indices(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            let indices = Tensor::<CudaRuntime>::zeros(shape, DType::I64, &self.device);
            return Ok((a.clone(), indices));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a);
        let out_values = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);
        let out_indices = Tensor::<CudaRuntime>::empty(shape, DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_sort(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out_values.storage().ptr(),
                out_indices.storage().ptr(),
                outer_size,
                sort_size,
                inner_size,
                descending,
            )?;
        }

        Ok((out_values, out_indices))
    }

    fn argsort(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(Tensor::<CudaRuntime>::zeros(
                shape,
                DType::I64,
                &self.device,
            ));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(shape, DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_argsort(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                sort_size,
                inner_size,
                descending,
            )?;
        }

        Ok(out)
    }

    fn topk(
        &self,
        a: &Tensor<CudaRuntime>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            if k > 1 {
                return Err(Error::InvalidArgument {
                    arg: "k",
                    reason: "k cannot be greater than 1 for scalar tensors".to_string(),
                });
            }
            let indices = Tensor::<CudaRuntime>::zeros(shape, DType::I64, &self.device);
            return Ok((a.clone(), indices));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let dim_size = shape[dim_idx];
        if k > dim_size {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!(
                    "k ({}) cannot be greater than dimension size ({})",
                    k, dim_size
                ),
            });
        }

        if k == 0 {
            let mut out_shape = shape.to_vec();
            out_shape[dim_idx] = 0;
            let out_values = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);
            let out_indices = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);
            return Ok((out_values, out_indices));
        }

        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a);

        let mut out_shape = shape.to_vec();
        out_shape[dim_idx] = k;

        let out_values = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);
        let out_indices = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_topk(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out_values.storage().ptr(),
                out_indices.storage().ptr(),
                outer_size,
                sort_size,
                inner_size,
                k,
                largest,
                sorted,
            )?;
        }

        Ok((out_values, out_indices))
    }

    fn unique(&self, a: &Tensor<CudaRuntime>, _sorted: bool) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let numel = a.numel();

        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(&[0], dtype, &self.device));
        }

        // Flatten and make contiguous
        let a_flat = a.reshape(&[numel])?;
        let a_contig = ensure_contiguous(&a_flat);

        // Sort first
        let sorted_tensor = self.sort(&a_contig, 0, false)?;

        // Allocate counter on device (using U32)
        let counter = Tensor::<CudaRuntime>::zeros(&[1], DType::U32, &self.device);

        // Count unique elements
        unsafe {
            super::super::kernels::launch_count_unique(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                sorted_tensor.storage().ptr(),
                counter.storage().ptr(),
                numel,
            )?;
        }

        // Synchronize and read count
        self.stream
            .synchronize()
            .map_err(|e| Error::Internal(format!("CUDA sync failed: {:?}", e)))?;
        let count_data = counter.to_vec::<u32>();
        let unique_count = count_data[0] as usize;

        if unique_count == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(&[0], dtype, &self.device));
        }

        // Reset counter and allocate output
        let counter = Tensor::<CudaRuntime>::zeros(&[1], DType::U32, &self.device);
        let out = Tensor::<CudaRuntime>::empty(&[unique_count], dtype, &self.device);

        // Extract unique elements
        unsafe {
            super::super::kernels::launch_extract_unique(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                sorted_tensor.storage().ptr(),
                out.storage().ptr(),
                counter.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn unique_with_counts(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let dtype = a.dtype();
        let numel = a.numel();

        if numel == 0 {
            let unique = Tensor::<CudaRuntime>::empty(&[0], dtype, &self.device);
            let inverse = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &self.device);
            let counts = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &self.device);
            return Ok((unique, inverse, counts));
        }

        // Get unique values (GPU-native)
        let unique = self.unique(a, true)?;
        let unique_count = unique.numel();

        // Compute inverse indices via searchsorted (GPU-native)
        let a_flat = a.reshape(&[numel])?;
        let inverse = self.searchsorted(&unique, &a_flat, false)?;

        // Count occurrences using GPU bincount kernel (no CPU round-trip)
        let counts = Tensor::<CudaRuntime>::zeros(&[unique_count], DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_bincount(
                &self.context,
                &self.stream,
                self.device.index,
                inverse.storage().ptr(),
                counts.storage().ptr(),
                numel,
                unique_count,
            )?;
        }

        Ok((unique, inverse, counts))
    }

    fn nonzero(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();
        let numel = a.numel();

        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(
                &[0, ndim],
                DType::I64,
                &self.device,
            ));
        }

        let a_contig = ensure_contiguous(a);

        // Phase 1: Count nonzero elements
        let counter = Tensor::<CudaRuntime>::zeros(&[1], DType::U32, &self.device);

        unsafe {
            super::super::kernels::launch_count_nonzero(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                counter.storage().ptr(),
                numel,
            )?;
        }

        // Synchronize and read count
        self.stream
            .synchronize()
            .map_err(|e| Error::Internal(format!("CUDA sync failed: {:?}", e)))?;
        let count_data = counter.to_vec::<u32>();
        let nnz = count_data[0] as usize;

        if nnz == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(
                &[0, ndim],
                DType::I64,
                &self.device,
            ));
        }

        if ndim == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(
                &[1, 0],
                DType::I64,
                &self.device,
            ));
        }

        // Phase 2: Gather flat indices
        let counter = Tensor::<CudaRuntime>::zeros(&[1], DType::U32, &self.device);
        let flat_indices = Tensor::<CudaRuntime>::empty(&[nnz], DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_gather_nonzero(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                flat_indices.storage().ptr(),
                counter.storage().ptr(),
                numel,
            )?;
        }

        // Phase 3: Convert flat indices to multi-indices
        let shape_tensor = Tensor::<CudaRuntime>::from_slice(
            &shape.iter().map(|&s| s as u32).collect::<Vec<_>>(),
            &[ndim],
            &self.device,
        );
        let out = Tensor::<CudaRuntime>::empty(&[nnz, ndim], DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_flat_to_multi_index(
                &self.context,
                &self.stream,
                self.device.index,
                flat_indices.storage().ptr(),
                out.storage().ptr(),
                nnz,
                ndim,
                shape_tensor.storage().ptr(),
            )?;
        }

        Ok(out)
    }

    fn searchsorted(
        &self,
        sorted_sequence: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        right: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        if sorted_sequence.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![sorted_sequence.numel()],
                got: sorted_sequence.shape().to_vec(),
            });
        }

        if sorted_sequence.dtype() != values.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: sorted_sequence.dtype(),
                rhs: values.dtype(),
            });
        }

        let dtype = sorted_sequence.dtype();
        let seq_len = sorted_sequence.numel();
        let num_values = values.numel();

        if num_values == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(
                values.shape(),
                DType::I64,
                &self.device,
            ));
        }

        let seq_contig = ensure_contiguous(sorted_sequence);
        let values_contig = ensure_contiguous(values);
        let out = Tensor::<CudaRuntime>::empty(values.shape(), DType::I64, &self.device);

        unsafe {
            super::super::kernels::launch_searchsorted(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seq_contig.storage().ptr(),
                values_contig.storage().ptr(),
                out.storage().ptr(),
                seq_len,
                num_values,
                right,
            )?;
        }

        Ok(out)
    }
}

// ============================================================================
// ConditionalOps Implementation
// ============================================================================

/// ConditionalOps implementation for CUDA runtime.
impl ConditionalOps<CudaRuntime> for CudaClient {
    fn where_cond(
        &self,
        cond: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate that x and y have the same dtype
        let dtype = validate_binary_dtypes(x, y)?;
        let cond_dtype = cond.dtype();

        // For same shapes, use optimized element-wise kernel on GPU
        if cond.shape() == x.shape() && x.shape() == y.shape() {
            let cond_contig = ensure_contiguous(cond);
            let x_contig = ensure_contiguous(x);
            let y_contig = ensure_contiguous(y);
            let out = Tensor::<CudaRuntime>::empty(x.shape(), dtype, &self.device);

            unsafe {
                if cond_dtype == DType::U8 {
                    // Optimized U8 kernel
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
                } else {
                    // Generic kernel for F32, F64, I32, I64, U32 conditions
                    launch_where_generic_op(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        cond_dtype,
                        dtype,
                        cond_contig.storage().ptr(),
                        x_contig.storage().ptr(),
                        y_contig.storage().ptr(),
                        out.storage().ptr(),
                        out.numel(),
                    )?;
                }
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
            if cond_dtype == DType::U8 {
                // Optimized U8 broadcast kernel
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
            } else {
                // Generic broadcast kernel for non-U8 conditions
                launch_where_broadcast_generic_op(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    &self.device,
                    cond_dtype,
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
        }

        Ok(out)
    }
}

// ============================================================================
// UtilityOps Implementation
// ============================================================================

/// UtilityOps implementation for CUDA runtime.
impl UtilityOps<CudaRuntime> for CudaClient {
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
