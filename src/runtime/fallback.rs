//! Generic CPU fallback utilities for GPU backends
//!
#![allow(unreachable_code)] // dispatch_dtype! macro uses early returns in all branches
//!
//! This module provides shared CPU fallback implementations for operations
//! that are not yet implemented natively on GPU backends (CUDA, WGPU).
//!
//! # Design
//!
//! The fallback pattern:
//! 1. Copy tensor data from GPU to CPU
//! 2. Execute operation using CPU backend
//! 3. Copy result back to GPU
//!
//! This is intentionally simple and correct - performance optimization
//! via native GPU kernels is a separate phase.
//!
//! # Usage
//!
//! GPU backends use these helpers to implement `TensorOps`:
//!
//! ```ignore
//! use numr::runtime::fallback::{CpuFallbackContext, binary_op_fallback};
//!
//! impl TensorOps<CudaRuntime> for CudaClient {
//!     fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
//!         binary_op_fallback(
//!             a, b,
//!             |cpu_a, cpu_b, cpu_client| cpu_client.add(cpu_a, cpu_b),
//!             &self.device,
//!             "add",
//!         )
//!     }
//! }
//! ```

use crate::dispatch_dtype;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOp, CompareOp, CompareOps, ReduceOp, ScalarOps, TensorOps, UnaryOp, broadcast_shape,
    reduce_output_shape,
};
use crate::runtime::{Device, Runtime, cpu};
use crate::tensor::Tensor;

// ============================================================================
// CPU Fallback Context
// ============================================================================

/// CPU fallback context for operations not yet implemented natively on GPU.
///
/// This struct holds the CPU device and client needed for fallback operations.
/// Using a struct avoids repeated boilerplate in every fallback function.
///
/// # Thread Safety
///
/// Each call creates a new context. The CPU runtime is stateless and
/// thread-safe, so this is fine for concurrent use.
pub struct CpuFallbackContext {
    /// CPU device (lightweight, just an ID)
    pub device: cpu::CpuDevice,
    /// CPU client (contains allocator reference)
    pub client: cpu::CpuClient,
}

impl CpuFallbackContext {
    /// Create a new CPU fallback context.
    #[inline]
    pub fn new() -> Self {
        let device = cpu::CpuDevice::new();
        let client = cpu::CpuRuntime::default_client(&device);
        Self { device, client }
    }

    /// Create a CPU tensor from GPU tensor data.
    ///
    /// This copies the tensor data from GPU memory to CPU memory.
    #[inline]
    pub fn tensor_from_gpu<T: Element, R: Runtime>(
        &self,
        tensor: &Tensor<R>,
    ) -> Tensor<cpu::CpuRuntime> {
        let data: Vec<T> = tensor.to_vec();
        Tensor::<cpu::CpuRuntime>::from_slice(&data, tensor.shape(), &self.device)
    }
}

impl Default for CpuFallbackContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that two tensors have matching dtypes for binary operations.
#[inline]
pub fn validate_binary_dtypes<R: Runtime>(a: &Tensor<R>, b: &Tensor<R>) -> Result<DType> {
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(a.dtype())
}

/// Compute broadcast shape for binary operations.
#[inline]
pub fn compute_broadcast_shape<R: Runtime>(a: &Tensor<R>, b: &Tensor<R>) -> Result<Vec<usize>> {
    broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

// ============================================================================
// Generic Fallback Operations
// ============================================================================

/// Perform a binary operation using CPU fallback.
///
/// # Type Parameters
///
/// * `R` - The GPU runtime (CudaRuntime, WgpuRuntime)
/// * `D` - The device type for the GPU runtime
///
/// # Arguments
///
/// * `a` - First input tensor (on GPU)
/// * `b` - Second input tensor (on GPU)
/// * `op` - Binary operation to perform
/// * `device` - GPU device to create output tensor on
/// * `op_name` - Operation name for error messages
pub fn binary_op_fallback<R, D>(
    a: &Tensor<R>,
    b: &Tensor<R>,
    op: BinaryOp,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b);

        let result_cpu = match op {
            BinaryOp::Add => cpu.client.add(&a_cpu, &b_cpu)?,
            BinaryOp::Sub => cpu.client.sub(&a_cpu, &b_cpu)?,
            BinaryOp::Mul => cpu.client.mul(&a_cpu, &b_cpu)?,
            BinaryOp::Div => cpu.client.div(&a_cpu, &b_cpu)?,
            BinaryOp::Pow => cpu.client.pow(&a_cpu, &b_cpu)?,
            BinaryOp::Max => cpu.client.maximum(&a_cpu, &b_cpu)?,
            BinaryOp::Min => cpu.client.minimum(&a_cpu, &b_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
    }, op_name);

    unreachable!()
}

/// Perform a unary operation using CPU fallback.
pub fn unary_op_fallback<R, D>(
    a: &Tensor<R>,
    op: UnaryOp,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);

        let result_cpu = match op {
            UnaryOp::Neg => cpu.client.neg(&a_cpu)?,
            UnaryOp::Abs => cpu.client.abs(&a_cpu)?,
            UnaryOp::Sqrt => cpu.client.sqrt(&a_cpu)?,
            UnaryOp::Exp => cpu.client.exp(&a_cpu)?,
            UnaryOp::Log => cpu.client.log(&a_cpu)?,
            UnaryOp::Sin => cpu.client.sin(&a_cpu)?,
            UnaryOp::Cos => cpu.client.cos(&a_cpu)?,
            UnaryOp::Tan => cpu.client.tan(&a_cpu)?,
            UnaryOp::Tanh => cpu.client.tanh(&a_cpu)?,
            UnaryOp::Recip => cpu.client.recip(&a_cpu)?,
            UnaryOp::Square => cpu.client.square(&a_cpu)?,
            UnaryOp::Floor => cpu.client.floor(&a_cpu)?,
            UnaryOp::Ceil => cpu.client.ceil(&a_cpu)?,
            UnaryOp::Round => cpu.client.round(&a_cpu)?,
            UnaryOp::Sign => cpu.client.sign(&a_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform a scalar operation using CPU fallback.
pub fn scalar_op_fallback<R, D>(
    a: &Tensor<R>,
    op: BinaryOp,
    scalar: f64,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);

        let result_cpu = match op {
            BinaryOp::Add => cpu.client.add_scalar(&a_cpu, scalar)?,
            BinaryOp::Sub => cpu.client.sub_scalar(&a_cpu, scalar)?,
            BinaryOp::Mul => cpu.client.mul_scalar(&a_cpu, scalar)?,
            BinaryOp::Div => cpu.client.div_scalar(&a_cpu, scalar)?,
            BinaryOp::Pow => cpu.client.pow_scalar(&a_cpu, scalar)?,
            _ => return Err(Error::UnsupportedDType { dtype, op: op_name }),
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform a reduce operation using CPU fallback.
pub fn reduce_op_fallback<R, D>(
    a: &Tensor<R>,
    op: ReduceOp,
    dims: &[usize],
    keepdim: bool,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let out_shape = reduce_output_shape(a.shape(), dims, keepdim);
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);

        let result_cpu = match op {
            ReduceOp::Sum => cpu.client.sum(&a_cpu, dims, keepdim)?,
            ReduceOp::Mean => cpu.client.mean(&a_cpu, dims, keepdim)?,
            ReduceOp::Max => cpu.client.max(&a_cpu, dims, keepdim)?,
            ReduceOp::Min => cpu.client.min(&a_cpu, dims, keepdim)?,
            _ => return Err(Error::UnsupportedDType { dtype, op: op_name }),
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
    }, op_name);

    unreachable!()
}

/// Perform an activation operation using CPU fallback.
///
/// This is a generic helper for activation functions (relu, sigmoid)
/// that share the same pattern: copy to CPU, apply function, copy back.
pub fn activation_fallback<R, D, F>(
    a: &Tensor<R>,
    device: &D,
    op_name: &'static str,
    op_fn: F,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
    F: Fn(&cpu::CpuClient, &Tensor<cpu::CpuRuntime>) -> Result<Tensor<cpu::CpuRuntime>>,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);
        let result_cpu = op_fn(&cpu.client, &a_cpu)?;
        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform softmax operation using CPU fallback.
pub fn softmax_fallback<R, D>(
    a: &Tensor<R>,
    dim: isize,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);
        let result_cpu = cpu.client.softmax(&a_cpu, dim)?;
        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform matmul operation using CPU fallback.
pub fn matmul_fallback<R, D>(
    a: &Tensor<R>,
    b: &Tensor<R>,
    out_shape: &[usize],
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    let dtype = validate_binary_dtypes(a, b)?;
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b);

        let result_cpu = cpu.client.matmul(&a_cpu, &b_cpu)?;
        let result_data: Vec<T> = result_cpu.to_vec();

        return Ok(Tensor::<R>::from_slice(&result_data, out_shape, device));
    }, op_name);

    unreachable!()
}

/// Perform a compare operation using CPU fallback.
pub fn compare_op_fallback<R, D>(
    a: &Tensor<R>,
    b: &Tensor<R>,
    op: CompareOp,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a);
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b);

        let result_cpu = match op {
            CompareOp::Eq => cpu.client.eq(&a_cpu, &b_cpu)?,
            CompareOp::Ne => cpu.client.ne(&a_cpu, &b_cpu)?,
            CompareOp::Lt => cpu.client.lt(&a_cpu, &b_cpu)?,
            CompareOp::Le => cpu.client.le(&a_cpu, &b_cpu)?,
            CompareOp::Gt => cpu.client.gt(&a_cpu, &b_cpu)?,
            CompareOp::Ge => cpu.client.ge(&a_cpu, &b_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
    }, op_name);

    unreachable!()
}

/// Compute broadcast shape for ternary operations (where_cond).
///
/// Returns the broadcasted shape of all three tensors.
#[inline]
pub fn compute_ternary_broadcast_shape<R: Runtime>(
    cond: &Tensor<R>,
    x: &Tensor<R>,
    y: &Tensor<R>,
) -> Result<Vec<usize>> {
    // First compute broadcast of x and y
    let xy_shape = broadcast_shape(x.shape(), y.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: x.shape().to_vec(),
        rhs: y.shape().to_vec(),
    })?;

    // Then broadcast cond with the x-y broadcast result
    broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| Error::BroadcastError {
        lhs: cond.shape().to_vec(),
        rhs: xy_shape,
    })
}

/// Perform a where_cond (ternary conditional select) operation using CPU fallback.
///
/// This supports full broadcasting across cond, x, and y tensors.
///
/// # Arguments
///
/// * `cond` - Condition tensor (U8/boolean) on GPU
/// * `x` - "True" values tensor on GPU
/// * `y` - "False" values tensor on GPU
/// * `device` - GPU device to create output tensor on
/// * `op_name` - Operation name for error messages
pub fn where_cond_fallback<R, D>(
    cond: &Tensor<R>,
    x: &Tensor<R>,
    y: &Tensor<R>,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D>,
    D: Device + Clone,
{
    // Validate dtypes
    let dtype = validate_binary_dtypes(x, y)?;
    if cond.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: cond.dtype(),
        });
    }

    let out_shape = compute_ternary_broadcast_shape(cond, x, y)?;
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        // Copy all three tensors to CPU
        let cond_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<u8, R>(cond);
        let x_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(x);
        let y_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(y);

        // Execute where_cond on CPU
        let result_cpu = cpu.client.where_cond(&cond_cpu, &x_cpu, &y_cpu)?;

        // Copy result back to GPU
        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
    }, op_name);

    unreachable!()
}
