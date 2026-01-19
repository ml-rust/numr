//! TensorOps, ScalarOps, and CompareOps implementations for WebGPU runtime
//!
//! This module implements tensor operations for WebGPU using native WGSL
//! compute shaders. All operations run entirely on GPU with no CPU fallback.
//!
//! # Performance Note
//!
//! All operations use native WGSL compute shaders for maximum performance.
//! Data stays on GPU throughout the computation pipeline.

use wgpu::BufferUsages;

use super::client::get_buffer;
use super::shaders::{elementwise, matmul, norm, reduce};
use super::{WgpuClient, WgpuRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    AccumulationPrecision, CompareOps, ScalarOps, TensorOps, broadcast_shape, matmul_output_shape,
};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a uniform buffer with the given data.
fn create_params_buffer<T: bytemuck::Pod>(client: &WgpuClient, data: &T) -> wgpu::Buffer {
    let buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&buffer, 0, bytemuck::bytes_of(data));
    buffer
}

/// Get the wgpu buffer from a tensor's storage pointer.
fn get_tensor_buffer(tensor: &Tensor<WgpuRuntime>) -> Result<std::sync::Arc<wgpu::Buffer>> {
    let ptr = tensor.storage().ptr();
    get_buffer(ptr).ok_or_else(|| Error::Internal("Buffer not found in registry".to_string()))
}

/// Allocate output tensor with given shape and dtype.
fn alloc_output(client: &WgpuClient, shape: &[usize], dtype: DType) -> Tensor<WgpuRuntime> {
    Tensor::empty(shape, dtype, client.device())
}

/// Ensure tensor is contiguous, returning a new tensor if needed.
fn ensure_contiguous(tensor: &Tensor<WgpuRuntime>) -> Tensor<WgpuRuntime> {
    if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()
    }
}

// ============================================================================
// Params Structs (must match WGSL shader structs)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryParams {
    numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnaryParams {
    numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarParams {
    numel: u32,
    scalar: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ClampParams {
    numel: u32,
    min_val: f32,
    max_val: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WhereParams {
    numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FullReduceParams {
    numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    batch_size: u32,
    dim_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}

// ============================================================================
// TensorOps Implementation
// ============================================================================

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
        // For now, only F32 -> F32 is supported natively
        // Other casts use CPU fallback
        if a.dtype() == DType::F32 && dtype == DType::F32 {
            // No-op
            return Ok(a.clone());
        }

        use crate::dispatch_dtype;
        let src_dtype = a.dtype();
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
}

// ============================================================================
// ScalarOps Implementation
// ============================================================================

impl ScalarOps<WgpuRuntime> for WgpuClient {
    fn add_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "add_scalar", a, scalar)
    }

    fn sub_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "sub_scalar", a, scalar)
    }

    fn mul_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "mul_scalar", a, scalar)
    }

    fn div_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "div_scalar", a, scalar)
    }

    fn pow_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "pow_scalar", a, scalar)
    }
}

// ============================================================================
// CompareOps Implementation
// ============================================================================

impl CompareOps<WgpuRuntime> for WgpuClient {
    fn eq(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "eq", a, b)
    }

    fn ne(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "ne", a, b)
    }

    fn lt(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "lt", a, b)
    }

    fn le(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "le", a, b)
    }

    fn gt(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "gt", a, b)
    }

    fn ge(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "ge", a, b)
    }
}

// ============================================================================
// Native GPU Operation Implementations
// ============================================================================

fn native_binary_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // Compute broadcast shape
    let out_shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;

    // Broadcasting not yet implemented natively - fall back for different shapes
    if a.shape() != b.shape() {
        return crate::runtime::fallback::binary_op_fallback(
            a,
            b,
            match op {
                "add" => crate::ops::BinaryOp::Add,
                "sub" => crate::ops::BinaryOp::Sub,
                "mul" => crate::ops::BinaryOp::Mul,
                "div" => crate::ops::BinaryOp::Div,
                "pow" => crate::ops::BinaryOp::Pow,
                "maximum" | "max" => crate::ops::BinaryOp::Max,
                "minimum" | "min" => crate::ops::BinaryOp::Min,
                _ => return Err(Error::Internal(format!("Unknown binary op: {}", op))),
            },
            &client.device_id,
            op,
        );
    }

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);

    let numel = out_shape.iter().product();
    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = BinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_binary_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &b_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

fn native_unary_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = UnaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_unary_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

fn native_scalar_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    scalar: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ScalarParams {
        numel: numel as u32,
        scalar: scalar as f32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_scalar_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

fn native_compare_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // Broadcasting not yet implemented natively
    if a.shape() != b.shape() {
        return crate::runtime::fallback::compare_op_fallback(
            a,
            b,
            match op {
                "eq" => crate::ops::CompareOp::Eq,
                "ne" => crate::ops::CompareOp::Ne,
                "lt" => crate::ops::CompareOp::Lt,
                "le" => crate::ops::CompareOp::Le,
                "gt" => crate::ops::CompareOp::Gt,
                "ge" => crate::ops::CompareOp::Ge,
                _ => return Err(Error::Internal(format!("Unknown compare op: {}", op))),
            },
            &client.device_id,
            op,
        );
    }

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let numel = a.numel();

    // Output is same dtype (F32 for now, TODO: U8 for proper bool)
    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = BinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_compare_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &b_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

fn native_reduce_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let _dtype = a.dtype();
    let shape = a.shape();

    if dims.is_empty() {
        // Full reduction
        return native_full_reduce(client, op, a);
    }

    // For multi-dim reduction, reduce one dimension at a time
    if dims.len() > 1 {
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a)); // Sort in descending order

        let mut result = a.clone();
        for &dim in &sorted_dims {
            result = native_single_dim_reduce(client, op, &result, dim, true)?;
        }

        // Remove dims if !keepdim
        if !keepdim {
            let mut out_shape: Vec<usize> = shape.to_vec();
            for &dim in &sorted_dims {
                out_shape.remove(dim);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            result = result.reshape(&out_shape)?;
        }

        return Ok(result);
    }

    // Single dimension reduction
    let dim = dims[0];
    native_single_dim_reduce(client, op, a, dim, keepdim)
}

fn native_single_dim_reduce(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a);

    // Compute parameters
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel_out = outer_size * inner_size;

    // Output shape
    let out_shape: Vec<usize> = if keepdim {
        let mut s = shape.to_vec();
        s[dim] = 1;
        s
    } else {
        let mut s: Vec<usize> = shape[..dim].to_vec();
        s.extend_from_slice(&shape[dim + 1..]);
        if s.is_empty() {
            s.push(1);
        }
        s
    };

    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ReduceParams {
        reduce_size: reduce_size as u32,
        outer_size: outer_size.max(1) as u32,
        inner_size: inner_size.max(1) as u32,
        numel_out: numel_out.max(1) as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel_out.max(1),
        dtype,
    )?;

    Ok(out)
}

fn native_full_reduce(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    // For mean, we need to divide by numel at the end
    let is_mean = op == "mean";
    let reduce_op = if is_mean { "sum" } else { op };

    // Two-pass reduction for large arrays
    let workgroup_size = 256;
    let num_workgroups = (numel + workgroup_size - 1) / workgroup_size;

    if num_workgroups <= 1 {
        // Single pass
        let out = alloc_output(client, &[1], dtype);
        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = FullReduceParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        reduce::launch_full_reduce_op(
            client.pipeline_cache(),
            client.wgpu_queue(),
            reduce_op,
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        if is_mean {
            return client.div_scalar(&out, numel as f64);
        }
        return Ok(out);
    }

    // Multi-pass: first reduce to num_workgroups values, then reduce again
    let partial = alloc_output(client, &[num_workgroups], dtype);
    let a_buf = get_tensor_buffer(&a_contig)?;
    let partial_buf = get_tensor_buffer(&partial)?;

    let params = FullReduceParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_full_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        reduce_op,
        &a_buf,
        &partial_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    // Second pass
    let out = alloc_output(client, &[1], dtype);
    let out_buf = get_tensor_buffer(&out)?;

    let params2 = FullReduceParams {
        numel: num_workgroups as u32,
    };
    let params_buf2 = create_params_buffer(client, &params2);

    reduce::launch_full_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        reduce_op,
        &partial_buf,
        &out_buf,
        &params_buf2,
        num_workgroups,
        dtype,
    )?;

    if is_mean {
        return client.div_scalar(&out, numel as f64);
    }
    Ok(out)
}

fn native_softmax(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Normalize dim
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

    // Softmax is only efficient on last dimension in our implementation
    // For other dimensions, use CPU fallback
    if dim != ndim - 1 {
        return crate::runtime::fallback::softmax_fallback(
            a,
            dim as isize,
            &client.device_id,
            "softmax",
        );
    }

    let a_contig = ensure_contiguous(a);
    let batch_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];

    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = SoftmaxParams {
        batch_size: batch_size.max(1) as u32,
        dim_size: dim_size as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_softmax_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok(out)
}

fn native_argreduce_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a);

    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel_out = outer_size * inner_size;

    // Output shape
    let out_shape: Vec<usize> = if keepdim {
        let mut s = shape.to_vec();
        s[dim] = 1;
        s
    } else {
        let mut s: Vec<usize> = shape[..dim].to_vec();
        s.extend_from_slice(&shape[dim + 1..]);
        if s.is_empty() {
            s.push(1);
        }
        s
    };

    // Output is I64 for indices (but we use F32 storage for now)
    // TODO: proper I64 output
    let out = alloc_output(client, &out_shape, DType::F32);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ArgReduceParams {
        reduce_size: reduce_size as u32,
        outer_size: outer_size.max(1) as u32,
        inner_size: inner_size.max(1) as u32,
        numel_out: numel_out.max(1) as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_argreduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel_out.max(1),
        dtype,
    )?;

    Ok(out)
}

fn native_matmul(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or_else(|| {
        Error::Internal(format!(
            "matmul shape mismatch: {:?} @ {:?}",
            a.shape(),
            b.shape()
        ))
    })?;

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 2D case
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: 1,
        };
        let params_buf = create_params_buffer(client, &params);

        // Use tiled for larger matrices, simple for small ones
        if m * n > 256 * 256 {
            matmul::launch_matmul(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                dtype,
            )?;
        } else {
            matmul::launch_matmul_simple(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                dtype,
            )?;
        }

        return Ok(out);
    }

    // Batched matmul - fall back to CPU for now
    // TODO: implement batched matmul natively
    crate::runtime::fallback::matmul_fallback(a, b, &out_shape, &client.device_id, "matmul")
}

fn native_clamp(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    min_val: f64,
    max_val: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ClampParams {
        numel: numel as u32,
        min_val: min_val as f32,
        max_val: max_val as f32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_clamp_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

fn native_where_cond(
    client: &WgpuClient,
    cond: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    y: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = x.dtype();

    // All must have same shape for native implementation
    if cond.shape() != x.shape() || x.shape() != y.shape() {
        return crate::runtime::fallback::where_cond_fallback(
            cond,
            x,
            y,
            &client.device_id,
            "where_cond",
        );
    }

    let cond_contig = ensure_contiguous(cond);
    let x_contig = ensure_contiguous(x);
    let y_contig = ensure_contiguous(y);
    let numel = x.numel();

    let out = alloc_output(client, x.shape(), dtype);

    let cond_buf = get_tensor_buffer(&cond_contig)?;
    let x_buf = get_tensor_buffer(&x_contig)?;
    let y_buf = get_tensor_buffer(&y_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = WhereParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_where_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &cond_buf,
        &x_buf,
        &y_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

fn native_rms_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "rms_norm requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let a_contig = ensure_contiguous(a);
    let weight_contig = ensure_contiguous(weight);

    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = RmsNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    norm::launch_rms_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &weight_buf,
        &out_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok(out)
}

fn native_layer_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "layer_norm requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let a_contig = ensure_contiguous(a);
    let weight_contig = ensure_contiguous(weight);
    let bias_contig = ensure_contiguous(bias);

    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let bias_buf = get_tensor_buffer(&bias_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = LayerNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    norm::launch_layer_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &weight_buf,
        &bias_buf,
        &out_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok(out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{CompareOps, ScalarOps, TensorOps};
    use crate::runtime::Runtime;
    use crate::runtime::wgpu::is_wgpu_available;

    fn create_test_tensor(data: &[f32], shape: &[usize]) -> Tensor<WgpuRuntime> {
        let device = super::super::WgpuDevice::new(0);
        Tensor::<WgpuRuntime>::from_slice(data, shape, &device)
    }

    #[test]
    fn test_wgpu_add() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = super::super::WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = create_test_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let result = client.add(&a, &b).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_wgpu_matmul() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = super::super::WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        // 2x3 @ 3x2 = 2x2
        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = create_test_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        let result = client.matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_wgpu_relu() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = super::super::WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[-1.0, 0.0, 1.0, 2.0], &[4]);
        let result = client.relu(&a).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_wgpu_sum() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = super::super::WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = client.sum(&a, &[0], false).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_wgpu_mul_scalar() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = super::super::WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let result = client.mul_scalar(&a, 2.0).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_wgpu_eq() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = super::super::WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = create_test_tensor(&[1.0, 0.0, 3.0, 0.0], &[4]);

        let result = client.eq(&a, &b).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![1.0, 0.0, 1.0, 0.0]);
    }
}
