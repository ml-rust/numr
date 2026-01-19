//! TensorOps trait implementation for WebGPU runtime.

use super::super::shaders::shape;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::*;
use super::native::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{AccumulationPrecision, ScalarOps, TensorOps};
use crate::runtime::shape_ops::{validate_cat, validate_stack};
use crate::runtime::{RuntimeClient, shape_ops, validate_arange, validate_eye};
use crate::tensor::Tensor;

impl TensorOps<WgpuRuntime> for WgpuClient {
    // --- Binary Operations ---

    fn add(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "add", a, b)
    }

    fn sub(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "sub", a, b)
    }

    fn mul(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "mul", a, b)
    }

    fn div(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "div", a, b)
    }

    fn pow(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "pow", a, b)
    }

    fn maximum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "maximum", a, b)
    }

    fn minimum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "minimum", a, b)
    }

    // --- Unary Operations ---

    fn neg(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "neg", a)
    }

    fn abs(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "abs", a)
    }

    fn sqrt(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sqrt", a)
    }

    fn exp(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "exp", a)
    }

    fn log(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "log", a)
    }

    fn sin(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sin", a)
    }

    fn cos(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "cos", a)
    }

    fn tan(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "tan", a)
    }

    fn tanh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "tanh", a)
    }

    fn recip(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "recip", a)
    }

    fn square(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "square", a)
    }

    fn floor(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "floor", a)
    }

    fn ceil(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "ceil", a)
    }

    fn round(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "round", a)
    }

    // --- Matrix Multiplication ---

    fn matmul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_matmul(self, a, b)
    }

    // --- Reduction Operations ---

    fn sum(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "sum", a, dims, keepdim)
    }

    fn mean(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "mean", a, dims, keepdim)
    }

    fn max(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "max", a, dims, keepdim)
    }

    fn min(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "min", a, dims, keepdim)
    }

    // --- Activation Functions ---

    fn relu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "relu", a)
    }

    fn sigmoid(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sigmoid", a)
    }

    fn softmax(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        native_softmax(self, a, dim)
    }

    fn silu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "silu", a)
    }

    fn gelu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "gelu", a)
    }

    fn leaky_relu(
        &self,
        a: &Tensor<WgpuRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_parametric_activation(self, "leaky_relu", a, negative_slope)
    }

    fn elu(&self, a: &Tensor<WgpuRuntime>, alpha: f64) -> Result<Tensor<WgpuRuntime>> {
        native_parametric_activation(self, "elu", a, alpha)
    }

    // --- Additional Unary Operations ---

    fn sign(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sign", a)
    }

    fn isnan(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "isnan", a)
    }

    fn isinf(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "isinf", a)
    }

    // --- Precision-Aware Reductions ---

    fn sum_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        self.sum(a, dims, keepdim)
    }

    fn max_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        self.max(a, dims, keepdim)
    }

    fn min_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        self.min(a, dims, keepdim)
    }

    // --- Normalization ---

    fn rms_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_rms_norm(self, a, weight, eps)
    }

    fn layer_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_layer_norm(self, a, weight, bias, eps)
    }

    // --- Argmax/Argmin ---

    fn argmax(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_argreduce_op(self, "argmax", a, dim, keepdim)
    }

    fn argmin(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_argreduce_op(self, "argmin", a, dim, keepdim)
    }

    // --- Cast ---

    fn cast(&self, a: &Tensor<WgpuRuntime>, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        let src_dtype = a.dtype();

        // Same-type cast is a no-op
        if src_dtype == dtype {
            return Ok(a.clone());
        }

        // Check if both dtypes are natively supported on WebGPU
        let wgpu_supported = [DType::F32, DType::I32, DType::U32];
        let native_cast = wgpu_supported.contains(&src_dtype) && wgpu_supported.contains(&dtype);

        if native_cast {
            // Use native WGSL cast shader
            native_cast_op(self, a, dtype)
        } else {
            // Fall back to CPU for unsupported dtypes (F64, F16, I8, etc.)
            use crate::dispatch_dtype;
            let cpu = crate::runtime::fallback::CpuFallbackContext::new();

            dispatch_dtype!(src_dtype, T => {
                let a_cpu: crate::tensor::Tensor<crate::runtime::cpu::CpuRuntime> =
                    cpu.tensor_from_gpu::<T, WgpuRuntime>(a);
                let result_cpu = cpu.client.cast(&a_cpu, dtype)?;

                dispatch_dtype!(dtype, U => {
                    let result_data: Vec<U> = result_cpu.to_vec();
                    return Ok(Tensor::<WgpuRuntime>::from_slice(&result_data, result_cpu.shape(), &self.device_id));
                }, "cast_output");
            }, "cast_input");
        }
    }

    // --- Where/Conditional ---

    fn where_cond(
        &self,
        cond: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_where_cond(self, cond, x, y)
    }

    // --- Utility Operations ---

    fn clamp(
        &self,
        a: &Tensor<WgpuRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_clamp(self, a, min_val, max_val)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        let zeros = Tensor::zeros(shape, dtype, self.device());
        self.add_scalar(&zeros, value)
    }

    fn arange(
        &self,
        start: f64,
        stop: f64,
        step: f64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Use shared validation
        let numel = validate_arange(start, stop, step)?;

        // Handle empty tensor case
        if numel == 0 {
            return Ok(Tensor::empty(&[0], dtype, self.device()));
        }

        // WebGPU only supports F32, I32, U32 natively (no F64, F16, I64, etc.)
        // This is a hardware limitation of WGSL.
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "arange",
            });
        }

        // Allocate output
        let out = alloc_output(self, &[numel], dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Create params (f32 precision - WebGPU limitation)
        let params = ArangeParams {
            numel: numel as u32,
            start: start as f32,
            step: step as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_arange(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn linspace(
        &self,
        start: f64,
        stop: f64,
        steps: usize,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU linspace only supports F32 because:
        // 1. WGSL has no F64 support, so computation must be in F32
        // 2. Integer linspace with F32 intermediate would lose precision
        // Use CPU backend for integer linspace if needed.
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "linspace (WebGPU only supports F32; use CPU for integer linspace)",
            });
        }

        if steps == 0 {
            return Ok(Tensor::empty(&[0], dtype, self.device()));
        }

        if steps == 1 {
            return Ok(Tensor::from_slice(&[start as f32], &[1], &self.device_id));
        }

        // Allocate output
        let out = alloc_output(self, &[steps], dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Create params
        let params = LinspaceParams {
            steps: steps as u32,
            start: start as f32,
            stop: stop as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_linspace(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            steps,
            dtype,
        )?;

        Ok(out)
    }

    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        // Use shared validation
        let (rows, cols) = validate_eye(n, m);

        if rows == 0 || cols == 0 {
            return Ok(Tensor::empty(&[rows, cols], dtype, self.device()));
        }

        // WebGPU only supports F32, I32, U32 natively (no F64, F16, I64, etc.)
        // This is a hardware limitation of WGSL.
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "eye" });
        }

        let numel = rows * cols;

        // Allocate output
        let out = alloc_output(self, &[rows, cols], dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Create params
        let params = EyeParams {
            n: rows as u32,
            m: cols as u32,
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_eye(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    // --- Statistical Operations ---

    fn var(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let shape = a.shape();
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| shape[d]).product()
        };

        let mean_val = self.mean(a, dims, true)?;
        let diff = self.sub(a, &mean_val)?;
        let diff_squared = self.square(&diff)?;
        let sum_sq = self.sum(&diff_squared, dims, keepdim)?;
        let divisor = (count.saturating_sub(correction)).max(1) as f64;
        self.div_scalar(&sum_sq, divisor)
    }

    fn std(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    // --- Random Operations ---

    fn rand(&self, _shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::UnsupportedDType {
            dtype,
            op: "rand (WebGPU native PRNG not implemented)",
        })
    }

    fn randn(&self, _shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::UnsupportedDType {
            dtype,
            op: "randn (WebGPU native PRNG not implemented)",
        })
    }

    // --- Indexing Operations ---

    fn gather(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_gather(self, a, dim, index)
    }

    fn scatter(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
        src: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_scatter(self, a, dim, index, src)
    }

    fn index_select(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_index_select(self, a, dim, index)
    }

    fn masked_select(
        &self,
        a: &Tensor<WgpuRuntime>,
        mask: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_masked_select(self, a, mask)
    }

    fn masked_fill(
        &self,
        a: &Tensor<WgpuRuntime>,
        mask: &Tensor<WgpuRuntime>,
        value: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_masked_fill(self, a, mask, value)
    }

    // --- Shape Operations ---

    fn cat(&self, tensors: &[&Tensor<WgpuRuntime>], dim: isize) -> Result<Tensor<WgpuRuntime>> {
        let cat_params = validate_cat(tensors, dim)?;

        // Check dtype is supported by WebGPU (F32, I32, U32 are natively supported)
        if !matches!(cat_params.dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype: cat_params.dtype,
                op: "cat",
            });
        }

        // Allocate output
        let out = alloc_output(self, &cat_params.out_shape, cat_params.dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Copy data from each tensor using WGSL kernel
        let mut cat_offset = 0usize;
        for &tensor in tensors {
            let tensor_contig = if tensor.is_contiguous() {
                tensor.clone()
            } else {
                tensor.contiguous()
            };
            let src_cat_size = tensor.shape()[cat_params.dim_idx];
            let total_elements = cat_params.outer_size * src_cat_size * cat_params.inner_size;

            let src_buf = get_tensor_buffer(&tensor_contig)?;

            let shader_params = CatShaderParams {
                outer_size: cat_params.outer_size as u32,
                src_cat_size: src_cat_size as u32,
                dst_cat_size: cat_params.cat_dim_total as u32,
                cat_offset: cat_offset as u32,
                inner_size: cat_params.inner_size as u32,
                total_elements: total_elements as u32,
            };
            let params_buf = create_params_buffer(self, &shader_params);

            shape::launch_cat_copy(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &src_buf,
                &out_buf,
                &params_buf,
                total_elements,
                cat_params.dtype,
            )?;

            cat_offset += src_cat_size;
        }

        Ok(out)
    }

    fn stack(&self, tensors: &[&Tensor<WgpuRuntime>], dim: isize) -> Result<Tensor<WgpuRuntime>> {
        // Validate tensors and get normalized dimension
        let _ = validate_stack(tensors, dim)?;

        // stack(tensors, dim) = cat([t.unsqueeze(dim) for t in tensors], dim)
        let unsqueezed: Vec<Tensor<WgpuRuntime>> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<_>>()?;

        let refs: Vec<&Tensor<WgpuRuntime>> = unsqueezed.iter().collect();
        self.cat(&refs, dim)
    }

    fn split(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        split_size: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<WgpuRuntime>>> {
        shape_ops::split_impl(tensor, split_size, dim)
    }

    fn chunk(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        chunks: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<WgpuRuntime>>> {
        shape_ops::chunk_impl(tensor, chunks, dim)
    }
}
