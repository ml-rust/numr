//! TensorOps trait implementation for WebGPU runtime.

use super::super::shaders::{distributions, shape};
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::*;
use super::native::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{AccumulationPrecision, ScalarOps, TensorOps};
use crate::runtime::shape_ops::{validate_cat, validate_stack};
use crate::runtime::{
    RuntimeClient, ensure_contiguous, normalize_dim, shape_ops, validate_arange, validate_eye,
};
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

    fn matmul_bias(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_matmul_bias(self, a, b, bias)
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

    fn prod(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "prod", a, dims, keepdim)
    }

    fn any(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "any", a, dims, keepdim)
    }

    fn all(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "all", a, dims, keepdim)
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

    fn prod_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU currently uses native precision; precision parameter reserved for future use
        self.prod(a, dims, keepdim)
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

    fn quantile(
        &self,
        a: &Tensor<WgpuRuntime>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::super::statistics::quantile_impl(self, a, q, dim, keepdim, interpolation)
    }

    fn percentile(
        &self,
        a: &Tensor<WgpuRuntime>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::super::statistics::percentile_impl(self, a, p, dim, keepdim)
    }

    fn median(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::super::statistics::median_impl(self, a, dim, keepdim)
    }

    fn histogram(
        &self,
        a: &Tensor<WgpuRuntime>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        super::super::statistics::histogram_impl(self, a, bins, range)
    }

    fn cov(&self, a: &Tensor<WgpuRuntime>, ddof: Option<usize>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<WgpuRuntime>>::cov(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<WgpuRuntime>>::corrcoef(self, a)
    }

    fn skew(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::super::statistics::skew_impl(self, a, dims, keepdim, correction)
    }

    fn kurtosis(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::super::statistics::kurtosis_impl(self, a, dims, keepdim, correction)
    }

    // --- Cumulative Operations ---

    fn cumsum(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        native_cumsum(self, a, dim)
    }

    fn cumprod(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        native_cumprod(self, a, dim)
    }

    fn logsumexp(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_logsumexp(self, a, dims, keepdim)
    }

    // --- Random Operations ---

    fn rand(&self, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU rand only supports F32
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType { dtype, op: "rand" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        // Allocate output
        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Create params with random seed
        // Note: WGSL doesn't support u64 natively, so we use u32 seed (truncated from timestamp).
        // This limits the seed space but is sufficient for most use cases.
        // For reproducible results, users should use explicit seeding (future API).
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(12345u32);
        let seed = time_seed.wrapping_add(counter);

        let params = RandParams {
            numel: numel as u32,
            seed,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_rand(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn randn(&self, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU randn only supports F32
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType { dtype, op: "randn" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        // Allocate output
        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Create params with random seed (see rand() for seed design notes)
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(12345u32);
        let seed = time_seed.wrapping_add(counter);

        let params = RandnParams {
            numel: numel as u32,
            seed,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_randn(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn randint(
        &self,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU randint only supports I32 and U32
        if !matches!(dtype, DType::I32 | DType::U32) {
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

        // For unsigned types, validate low >= 0
        if matches!(dtype, DType::U32) && low < 0 {
            return Err(Error::InvalidArgument {
                arg: "low",
                reason: format!(
                    "randint with unsigned dtype requires low >= 0, got low={}",
                    low
                ),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        // Allocate output
        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Create params with random seed (see rand() for seed design notes)
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(12345u32);
        let seed = time_seed.wrapping_add(counter);

        let range = (high - low) as u32;

        // Use dtype-specific param struct to ensure correct type handling
        // I32 uses signed low (i32), U32 uses unsigned low (u32)
        let params_buf = match dtype {
            DType::I32 => {
                let params = RandintParamsI32 {
                    numel: numel as u32,
                    low: low as i32, // Preserve sign for negative bounds
                    range,
                    seed,
                };
                create_params_buffer(self, &params)
            }
            DType::U32 => {
                let params = RandintParamsU32 {
                    numel: numel as u32,
                    low: low as u32,
                    range,
                    seed,
                };
                create_params_buffer(self, &params)
            }
            _ => unreachable!("randint only supports I32 and U32, validated above"),
        };

        // Launch kernel
        shape::launch_randint(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn multinomial(
        &self,
        probs: &Tensor<WgpuRuntime>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Validate input dtype - must be float for probabilities
        let dtype = probs.dtype();
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "multinomial (WebGPU only supports F32 probabilities)",
            });
        }

        // Validate num_samples > 0
        if num_samples == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: "multinomial requires num_samples > 0".to_string(),
            });
        }

        let shape = probs.shape();
        let ndim = shape.len();

        // Determine shape: 1D tensor [K] or 2D tensor [N, K]
        let (num_distributions, num_categories) = match ndim {
            1 => (1usize, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "probs",
                    reason: format!(
                        "multinomial requires 1D or 2D probability tensor, got {}D",
                        ndim
                    ),
                });
            }
        };

        // Validate num_categories > 0
        if num_categories == 0 {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: "multinomial requires at least 1 category".to_string(),
            });
        }

        // For without replacement, num_samples must not exceed num_categories
        if !replacement && num_samples > num_categories {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: format!(
                    "multinomial without replacement: num_samples ({}) cannot exceed num_categories ({})",
                    num_samples, num_categories
                ),
            });
        }

        // Check category limit for without replacement (shared memory limit)
        const MAX_CATEGORIES_WITHOUT_REPLACEMENT: usize = 1024;
        if !replacement && num_categories > MAX_CATEGORIES_WITHOUT_REPLACEMENT {
            return Err(Error::backend_limitation(
                "WebGPU",
                "multinomial",
                format!(
                    "without replacement supports max {} categories, got {}",
                    MAX_CATEGORIES_WITHOUT_REPLACEMENT, num_categories
                ),
            ));
        }

        // Ensure input is contiguous
        let probs_contig = if probs.is_contiguous() {
            probs.clone()
        } else {
            probs.contiguous()
        };

        // Output dtype: I32 for WebGPU (no I64 support in WGSL)
        let out_dtype = DType::I32;
        let out_shape = if ndim == 1 {
            vec![num_samples]
        } else {
            vec![num_distributions, num_samples]
        };

        // Allocate output
        let out = alloc_output(self, &out_shape, out_dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let probs_buf = get_tensor_buffer(&probs_contig)?;

        // Generate random seed
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(54321u32);
        let seed = time_seed.wrapping_add(counter);

        if replacement {
            // With replacement: parallel sampling
            let params = MultinomialWithReplacementParams {
                num_distributions: num_distributions as u32,
                num_categories: num_categories as u32,
                num_samples: num_samples as u32,
                seed,
            };
            let params_buf = create_params_buffer(self, &params);

            let total_samples = num_distributions * num_samples;
            shape::launch_multinomial_with_replacement(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &probs_buf,
                &out_buf,
                &params_buf,
                total_samples,
                dtype,
            )?;
        } else {
            // Without replacement: sequential sampling with shared memory
            let params = MultinomialWithoutReplacementParams {
                num_distributions: num_distributions as u32,
                num_categories: num_categories as u32,
                num_samples: num_samples as u32,
                seed,
            };
            let params_buf = create_params_buffer(self, &params);

            shape::launch_multinomial_without_replacement(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &probs_buf,
                &out_buf,
                &params_buf,
                num_distributions,
                dtype,
            )?;
        }

        Ok(out)
    }

    // --- Distribution Sampling Operations ---

    fn bernoulli(&self, p: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bernoulli (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = BernoulliParams {
            numel: numel as u32,
            seed,
            p: p as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_bernoulli(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn beta(
        &self,
        alpha: f64,
        beta: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "beta (WebGPU only supports F32)",
            });
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = BetaDistParams {
            numel: numel as u32,
            seed,
            alpha: alpha as f32,
            beta: beta as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_beta_dist(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "gamma (WebGPU only supports F32)",
            });
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = GammaDistParams {
            numel: numel as u32,
            seed,
            shape: shape_param as f32,
            scale: scale as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_gamma_dist(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn exponential(&self, rate: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "exponential (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = ExponentialParams {
            numel: numel as u32,
            seed,
            rate: rate as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_exponential(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn poisson(&self, lambda: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "poisson (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = PoissonParams {
            numel: numel as u32,
            seed,
            lambda: lambda as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_poisson(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn binomial(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "binomial (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = BinomialParams {
            numel: numel as u32,
            seed,
            n_trials: n as u32, // WebGPU doesn't support u64, truncate to u32
            p: p as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_binomial(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn laplace(
        &self,
        loc: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "laplace (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = LaplaceParams {
            numel: numel as u32,
            seed,
            loc: loc as f32,
            scale: scale as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_laplace(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn chi_squared(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "chi_squared (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = ChiSquaredParams {
            numel: numel as u32,
            seed,
            df: df as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_chi_squared(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn student_t(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "student_t (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = StudentTParams {
            numel: numel as u32,
            seed,
            df: df as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_student_t(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn f_distribution(
        &self,
        df1: f64,
        df2: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "f_distribution (WebGPU only supports F32)",
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
            return Ok(Tensor::empty(shape, dtype, self.device()));
        }

        let out = alloc_output(self, shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = FDistributionParams {
            numel: numel as u32,
            seed,
            df1: df1 as f32,
            df2: df2 as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_f_distribution(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
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

    fn embedding_lookup(
        &self,
        embeddings: &Tensor<WgpuRuntime>,
        indices: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_embedding_lookup(self, embeddings, indices)
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

    fn repeat(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        repeats: &[usize],
    ) -> Result<Tensor<WgpuRuntime>> {
        let params = shape_ops::validate_repeat(tensor, repeats)?;

        // No-op if all repeats are 1
        if repeats.iter().all(|&r| r == 1) {
            return Ok(tensor.contiguous());
        }

        // Check dtype is supported by WebGPU
        if !matches!(tensor.dtype(), DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype: tensor.dtype(),
                op: "repeat",
            });
        }

        // Check ndim doesn't exceed shader limit
        if params.out_shape.len() > MAX_DIMS {
            return Err(Error::backend_limitation(
                "WebGPU",
                "repeat",
                format!(
                    "max {} dimensions, got {}",
                    MAX_DIMS,
                    params.out_shape.len()
                ),
            ));
        }

        // Ensure contiguous input
        let tensor_contig = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()
        };

        let total_elements: usize = params.out_shape.iter().product();

        // Allocate output
        let out = alloc_output(self, &params.out_shape, tensor.dtype());
        let out_buf = get_tensor_buffer(&out)?;
        let src_buf = get_tensor_buffer(&tensor_contig)?;

        // Build flat shape arrays, then pack for WGSL uniform buffer alignment
        let ndim = params.out_shape.len();
        let mut src_shape_flat = [0u32; 8];
        let mut out_shape_flat = [0u32; 8];
        for i in 0..ndim {
            src_shape_flat[i] = tensor.shape()[i] as u32;
            out_shape_flat[i] = params.out_shape[i] as u32;
        }

        let shader_params = RepeatParams {
            ndim: ndim as u32,
            total_elements: total_elements as u32,
            _pad0: 0,
            _pad1: 0,
            src_shape: pack_u32_array(&src_shape_flat),
            out_shape: pack_u32_array(&out_shape_flat),
        };
        let params_buf = create_params_buffer(self, &shader_params);

        shape::launch_repeat(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &src_buf,
            &out_buf,
            &params_buf,
            total_elements,
            tensor.dtype(),
        )?;

        Ok(out)
    }

    fn pad(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        padding: &[usize],
        value: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        let params = shape_ops::validate_pad(tensor, padding)?;

        // No-op if all padding is zero
        if padding.iter().all(|&p| p == 0) {
            return Ok(tensor.contiguous());
        }

        let dtype = tensor.dtype();

        // Check dtype is supported by WebGPU
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "pad" });
        }

        // Check ndim doesn't exceed shader limit
        if params.out_shape.len() > MAX_DIMS {
            return Err(Error::backend_limitation(
                "WebGPU",
                "pad",
                format!(
                    "max {} dimensions, got {}",
                    MAX_DIMS,
                    params.out_shape.len()
                ),
            ));
        }

        // Ensure contiguous input
        let tensor_contig = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()
        };

        let total_elements: usize = params.out_shape.iter().product();

        // Allocate output
        let out = alloc_output(self, &params.out_shape, dtype);
        let out_buf = get_tensor_buffer(&out)?;
        let src_buf = get_tensor_buffer(&tensor_contig)?;

        // Build flat shape arrays, then pack for WGSL uniform buffer alignment
        let ndim = params.out_shape.len();
        let mut src_shape_flat = [0u32; 8];
        let mut out_shape_flat = [0u32; 8];
        let mut pad_before_flat = [0u32; 8];
        for i in 0..ndim {
            src_shape_flat[i] = tensor.shape()[i] as u32;
            out_shape_flat[i] = params.out_shape[i] as u32;
            pad_before_flat[i] = params.pad_per_dim[i].0 as u32;
        }

        // Pack arrays for WGSL uniform buffer 16-byte alignment
        let src_shape = pack_u32_array(&src_shape_flat);
        let out_shape = pack_u32_array(&out_shape_flat);
        let pad_before = pack_u32_array(&pad_before_flat);

        // Create dtype-specific params buffer
        let params_buf = match dtype {
            DType::F32 => {
                let shader_params = PadParamsF32 {
                    ndim: ndim as u32,
                    total_elements: total_elements as u32,
                    fill_value: value as f32,
                    _pad0: 0,
                    src_shape,
                    out_shape,
                    pad_before,
                };
                create_params_buffer(self, &shader_params)
            }
            DType::I32 => {
                let shader_params = PadParamsI32 {
                    ndim: ndim as u32,
                    total_elements: total_elements as u32,
                    fill_value: value as i32,
                    _pad0: 0,
                    src_shape,
                    out_shape,
                    pad_before,
                };
                create_params_buffer(self, &shader_params)
            }
            DType::U32 => {
                let shader_params = PadParamsU32 {
                    ndim: ndim as u32,
                    total_elements: total_elements as u32,
                    fill_value: value as u32,
                    _pad0: 0,
                    src_shape,
                    out_shape,
                    pad_before,
                };
                create_params_buffer(self, &shader_params)
            }
            _ => unreachable!("dtype validated above"),
        };

        shape::launch_pad(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &src_buf,
            &out_buf,
            &params_buf,
            total_elements,
            dtype,
        )?;

        Ok(out)
    }

    fn roll(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        shift: isize,
        dim: isize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let params = shape_ops::validate_roll(tensor, shift, dim)?;

        // Zero shift is a no-op
        if params.shift == 0 {
            return Ok(tensor.contiguous());
        }

        // Check dtype is supported by WebGPU
        if !matches!(tensor.dtype(), DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype: tensor.dtype(),
                op: "roll",
            });
        }

        // Ensure contiguous input
        let tensor_contig = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()
        };

        let total_elements = tensor.numel();
        let shape = tensor.shape();

        // Compute outer_size (product of dims before roll dim) and inner_size (product of dims after)
        let outer_size: usize = shape[..params.dim_idx].iter().product();
        let inner_size: usize = shape[params.dim_idx + 1..].iter().product();

        // Allocate output (same shape as input)
        let out = alloc_output(self, shape, tensor.dtype());
        let out_buf = get_tensor_buffer(&out)?;
        let src_buf = get_tensor_buffer(&tensor_contig)?;

        let shader_params = RollParams {
            outer_size: outer_size.max(1) as u32,
            dim_size: params.dim_size as u32,
            inner_size: inner_size.max(1) as u32,
            shift: params.shift as u32,
            total_elements: total_elements as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &shader_params);

        shape::launch_roll(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &src_buf,
            &out_buf,
            &params_buf,
            total_elements,
            tensor.dtype(),
        )?;

        Ok(out)
    }

    // ===== Linear Algebra =====

    fn solve(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::solve(self, a, b)
    }

    fn lstsq(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::lstsq(self, a, b)
    }

    fn pinverse(&self, a: &Tensor<WgpuRuntime>, rcond: Option<f64>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::pinverse(self, a, rcond)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        ord: crate::algorithm::linalg::MatrixNormOrder,
    ) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_norm(self, a, ord)
    }

    fn inverse(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::inverse(self, a)
    }

    fn det(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::det(self, a)
    }

    fn trace(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::trace(self, a)
    }

    fn diag(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diag(self, a)
    }

    fn diagflat(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diagflat(self, a)
    }

    fn matrix_rank(
        &self,
        a: &Tensor<WgpuRuntime>,
        tol: Option<f64>,
    ) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_rank(self, a, tol)
    }

    // ===== Complex Number Operations =====

    fn conj(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types, conjugate is identity
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "conj" });
        }

        let a_contig = ensure_contiguous(a);
        let numel = a.numel();
        let out = alloc_output(self, a.shape(), dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "conj",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn real(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types, return copy
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "real" });
        }

        let a_contig = ensure_contiguous(a);
        let numel = a.numel();
        let out_dtype = DType::F32; // Complex64 → F32
        let out = alloc_output(self, a.shape(), out_dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "real",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn imag(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types, return zeros with same dtype
        if !dtype.is_complex() {
            return Ok(Tensor::zeros(a.shape(), dtype, self.device()));
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "imag" });
        }

        // For complex types, extract imaginary part
        let out_dtype = DType::F32; // Complex64 → F32
        let a_contig = ensure_contiguous(a);
        let numel = a.numel();
        let out = alloc_output(self, a.shape(), out_dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "imag",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn angle(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types: angle(x) = 0 if x >= 0, π if x < 0
        if !dtype.is_complex() {
            match dtype {
                DType::F32 => {
                    // Use angle_real shader for F32
                    let a_contig = ensure_contiguous(a);
                    let numel = a.numel();
                    let out = alloc_output(self, a.shape(), dtype);

                    let a_buf = get_tensor_buffer(&a_contig)?;
                    let out_buf = get_tensor_buffer(&out)?;

                    let params = UnaryParams {
                        numel: numel as u32,
                    };
                    let params_buf = create_params_buffer(self, &params);

                    super::super::shaders::launch_angle_real(
                        self.pipeline_cache(),
                        self.wgpu_queue(),
                        &a_buf,
                        &out_buf,
                        &params_buf,
                        numel,
                    )?;

                    return Ok(out);
                }
                _ => {
                    // For other real types (integers, F64 not supported on WebGPU), return zeros
                    return Ok(Tensor::zeros(a.shape(), dtype, self.device()));
                }
            }
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "angle" });
        }

        // For complex types, compute phase angle
        let out_dtype = DType::F32; // Complex64 → F32
        let a_contig = ensure_contiguous(a);
        let numel = a.numel();
        let out = alloc_output(self, a.shape(), out_dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "angle",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    // ===== Sorting & Search Operations =====

    fn sort(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // Check dtype support (WebGPU: F32, I32, U32)
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "sort" });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(a.clone());
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        // Check sort size limit (WebGPU bitonic sort in shared memory)
        if sort_size > super::super::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "sort",
                format!(
                    "max {} elements per dimension, got {}",
                    super::super::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        // Compute strides
        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        // Ensure contiguous
        let a_contig = ensure_contiguous(a);

        // Allocate output
        let out = alloc_output(self, shape, dtype);
        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params buffer
        let params = SortParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            descending: descending as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Create dummy indices buffer
        let dummy_indices_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_sort_indices"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        super::super::shaders::sort::launch_sort_values_only(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        drop(dummy_indices_buf);
        Ok(out)
    }

    fn sort_with_indices(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sort_with_indices",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            let indices = Tensor::zeros(&[], DType::I32, self.device());
            return Ok((a.clone(), indices));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        if sort_size > super::super::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "sort_with_indices",
                format!(
                    "max {} elements per dimension, got {}",
                    super::super::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a);

        let values_out = alloc_output(self, shape, dtype);
        let indices_out = alloc_output(self, shape, DType::I32);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let values_buf = get_tensor_buffer(&values_out)?;
        let indices_buf = get_tensor_buffer(&indices_out)?;

        let params = SortParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            descending: descending as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::sort::launch_sort(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &values_buf,
            &indices_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        Ok((values_out, indices_out))
    }

    fn argsort(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "argsort",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(Tensor::zeros(&[], DType::I32, self.device()));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        if sort_size > super::super::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "argsort",
                format!(
                    "max {} elements per dimension, got {}",
                    super::super::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a);

        let indices_out = alloc_output(self, shape, DType::I32);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let indices_buf = get_tensor_buffer(&indices_out)?;

        let params = SortParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            descending: descending as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::sort::launch_argsort(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &indices_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        Ok(indices_out)
    }

    fn topk(
        &self,
        a: &Tensor<WgpuRuntime>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "topk" });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "tensor",
                reason: "topk requires at least 1-D tensor".to_string(),
            });
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        if k == 0 || k > sort_size {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!("k must be in [1, {}], got {}", sort_size, k),
            });
        }

        if sort_size > super::super::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "topk",
                format!(
                    "max {} elements per dimension, got {}",
                    super::super::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a);

        // Output shape has k instead of sort_size on dim
        let mut out_shape = shape.to_vec();
        out_shape[dim_idx] = k;

        let values_out = alloc_output(self, &out_shape, dtype);
        let indices_out = alloc_output(self, &out_shape, DType::I32);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let values_buf = get_tensor_buffer(&values_out)?;
        let indices_buf = get_tensor_buffer(&indices_out)?;

        let params = TopkParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            k: k as u32,
            largest: largest as u32,
            sorted: sorted as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::sort::launch_topk(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &values_buf,
            &indices_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        Ok((values_out, indices_out))
    }

    fn unique(&self, a: &Tensor<WgpuRuntime>, _sorted: bool) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "unique",
            });
        }

        let numel = a.numel();

        if numel == 0 {
            return Ok(Tensor::empty(&[0], dtype, self.device()));
        }

        // Step 1: Flatten and sort
        let flat = a.reshape(&[numel])?;
        let sorted_tensor = self.sort(&flat, 0, false)?;

        // Step 2: Count unique elements
        let count_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("unique_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero initialize
        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        let sorted_buf = get_tensor_buffer(&sorted_tensor)?;
        let params = CountParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::sort::launch_count_unique(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &sorted_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        // Read count
        let count = read_u32_from_buffer(self, &count_buf)?;

        if count == 0 {
            return Ok(Tensor::empty(&[0], dtype, self.device()));
        }

        // Step 3: Extract unique elements
        let out = alloc_output(self, &[count as usize], dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Reset counter
        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        super::super::shaders::sort::launch_extract_unique(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &sorted_buf,
            &out_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn unique_with_counts(
        &self,
        _a: &Tensor<WgpuRuntime>,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        // WebGPU unique_with_counts requires complex stream compaction not easily done in WGSL
        // Return unsupported for now - can be implemented with multiple passes
        Err(Error::NotImplemented {
            feature: "unique_with_counts for WebGPU",
        })
    }

    fn nonzero(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "nonzero",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();
        let numel = a.numel();

        if numel == 0 {
            return Ok(Tensor::empty(&[0, ndim], DType::I32, self.device()));
        }

        let a_contig = ensure_contiguous(a);
        let a_buf = get_tensor_buffer(&a_contig)?;

        // Phase 1: Count nonzero elements
        let count_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("nonzero_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        let params = CountParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::sort::launch_count_nonzero(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        let nnz = read_u32_from_buffer(self, &count_buf)? as usize;

        if nnz == 0 {
            return Ok(Tensor::empty(&[0, ndim], DType::I32, self.device()));
        }

        // Phase 2: Gather flat indices
        let flat_indices = alloc_output(self, &[nnz], DType::I32);
        let flat_indices_buf = get_tensor_buffer(&flat_indices)?;

        // Reset counter
        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        super::super::shaders::sort::launch_gather_nonzero(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &flat_indices_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        // Phase 3: Convert flat indices to multi-indices
        let multi_indices = alloc_output(self, &[nnz, ndim], DType::I32);
        let multi_indices_buf = get_tensor_buffer(&multi_indices)?;

        // Create shape buffer
        let mut shape_arr = [0u32; 8];
        for (i, &s) in shape.iter().enumerate().take(8) {
            shape_arr[i] = s as u32;
        }

        let flat_to_multi_params = FlatToMultiParams {
            nnz: nnz as u32,
            ndim: ndim as u32,
            _pad0: 0,
            _pad1: 0,
            shape: shape_arr,
        };
        let flat_to_multi_params_buf = create_params_buffer(self, &flat_to_multi_params);

        super::super::shaders::sort::launch_flat_to_multi_index(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &flat_indices_buf,
            &multi_indices_buf,
            &flat_to_multi_params_buf,
            nnz,
        )?;

        Ok(multi_indices)
    }

    fn searchsorted(
        &self,
        sorted_sequence: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        right: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = sorted_sequence.dtype();

        if dtype != values.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: values.dtype(),
            });
        }

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "searchsorted",
            });
        }

        // Sequence must be 1-D
        if sorted_sequence.shape().len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "sorted_sequence",
                reason: "sorted_sequence must be 1-D".to_string(),
            });
        }

        let seq_len = sorted_sequence.numel();
        let num_values = values.numel();

        if num_values == 0 {
            return Ok(Tensor::empty(values.shape(), DType::I32, self.device()));
        }

        let seq_contig = ensure_contiguous(sorted_sequence);
        let values_contig = ensure_contiguous(values);

        let out = alloc_output(self, values.shape(), DType::I32);

        let seq_buf = get_tensor_buffer(&seq_contig)?;
        let values_buf = get_tensor_buffer(&values_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = SearchsortedParams {
            seq_len: seq_len as u32,
            num_values: num_values as u32,
            right: right as u32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        super::super::shaders::sort::launch_searchsorted(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &seq_buf,
            &values_buf,
            &out_buf,
            &params_buf,
            num_values,
            dtype,
        )?;

        Ok(out)
    }
}
