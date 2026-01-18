//! TensorOps, ScalarOps, and CompareOps implementations for WebGPU runtime
//!
//! This module implements tensor operations for WebGPU using:
//! - CPU fallback for all operations (compute shaders will be added in future phases)
//!
//! # Performance Note
//!
//! All operations currently use CPU fallback, which involves Host-GPU memory
//! transfers. This is acceptable for Phase 3 foundation; compute shaders will
//! be implemented in future phases for better performance.

use super::{WgpuClient, WgpuRuntime};
use crate::error::Result;
use crate::ops::{
    AccumulationPrecision, BinaryOp, CompareOp, CompareOps, ReduceOp, ScalarOps, TensorOps,
    UnaryOp, matmul_output_shape,
};
use crate::runtime::fallback::{
    activation_fallback, binary_op_fallback, compare_op_fallback, matmul_fallback,
    reduce_op_fallback, scalar_op_fallback, softmax_fallback, unary_op_fallback,
};
use crate::tensor::Tensor;

// ============================================================================
// TensorOps Implementation
// ============================================================================

impl TensorOps<WgpuRuntime> for WgpuClient {
    // --- Binary Operations ---

    fn add(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Add, &self.device_id, "add")
    }

    fn sub(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Sub, &self.device_id, "sub")
    }

    fn mul(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Mul, &self.device_id, "mul")
    }

    fn div(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Div, &self.device_id, "div")
    }

    fn pow(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Pow, &self.device_id, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Max, &self.device_id, "maximum")
    }

    fn minimum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Min, &self.device_id, "minimum")
    }

    // --- Unary Operations ---

    fn neg(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Neg, &self.device_id, "neg")
    }

    fn abs(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Abs, &self.device_id, "abs")
    }

    fn sqrt(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Sqrt, &self.device_id, "sqrt")
    }

    fn exp(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Exp, &self.device_id, "exp")
    }

    fn log(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Log, &self.device_id, "log")
    }

    fn sin(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Sin, &self.device_id, "sin")
    }

    fn cos(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Cos, &self.device_id, "cos")
    }

    fn tan(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Tan, &self.device_id, "tan")
    }

    fn tanh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Tanh, &self.device_id, "tanh")
    }

    fn recip(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Recip, &self.device_id, "recip")
    }

    fn square(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Square, &self.device_id, "square")
    }

    fn floor(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Floor, &self.device_id, "floor")
    }

    fn ceil(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Ceil, &self.device_id, "ceil")
    }

    fn round(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Round, &self.device_id, "round")
    }

    // --- Matrix Multiplication ---

    fn matmul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or_else(|| {
            crate::error::Error::Internal(format!(
                "matmul shape mismatch: {:?} @ {:?} (expected [..., M, K] @ [..., K, N])",
                a.shape(),
                b.shape()
            ))
        })?;

        matmul_fallback(a, b, &out_shape, &self.device_id, "matmul")
    }

    // --- Reduction Operations ---

    fn sum(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        reduce_op_fallback(a, ReduceOp::Sum, dims, keepdim, &self.device_id, "sum")
    }

    fn mean(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        reduce_op_fallback(a, ReduceOp::Mean, dims, keepdim, &self.device_id, "mean")
    }

    fn max(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        reduce_op_fallback(a, ReduceOp::Max, dims, keepdim, &self.device_id, "max")
    }

    fn min(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        reduce_op_fallback(a, ReduceOp::Min, dims, keepdim, &self.device_id, "min")
    }

    // --- Activation Functions ---

    fn relu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "relu", |client, a_cpu| {
            client.relu(a_cpu)
        })
    }

    fn sigmoid(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "sigmoid", |client, a_cpu| {
            client.sigmoid(a_cpu)
        })
    }

    fn softmax(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        softmax_fallback(a, dim, &self.device_id, "softmax")
    }

    fn silu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "silu", |client, a_cpu| {
            client.silu(a_cpu)
        })
    }

    fn gelu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "gelu", |client, a_cpu| {
            client.gelu(a_cpu)
        })
    }

    // --- Additional Unary Operations ---

    fn sign(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        unary_op_fallback(a, UnaryOp::Sign, &self.device_id, "sign")
    }

    fn isnan(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "isnan", |client, a_cpu| {
            client.isnan(a_cpu)
        })
    }

    fn isinf(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "isinf", |client, a_cpu| {
            client.isinf(a_cpu)
        })
    }

    // --- Precision-Aware Reductions ---

    fn sum_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Fall back to regular sum (precision handling happens on CPU)
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
        activation_fallback(a, &self.device_id, "rms_norm", |client, a_cpu| {
            let weight_data: Vec<f32> = weight.to_vec();
            let cpu_device = crate::runtime::cpu::CpuDevice::new();
            let weight_cpu = crate::tensor::Tensor::<crate::runtime::cpu::CpuRuntime>::from_slice(
                &weight_data,
                weight.shape(),
                &cpu_device,
            );
            client.rms_norm(a_cpu, &weight_cpu, eps)
        })
    }

    fn layer_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "layer_norm", |client, a_cpu| {
            let weight_data: Vec<f32> = weight.to_vec();
            let cpu_device = crate::runtime::cpu::CpuDevice::new();
            let weight_cpu = crate::tensor::Tensor::<crate::runtime::cpu::CpuRuntime>::from_slice(
                &weight_data,
                weight.shape(),
                &cpu_device,
            );
            let bias_data: Vec<f32> = bias.to_vec();
            let bias_cpu = crate::tensor::Tensor::<crate::runtime::cpu::CpuRuntime>::from_slice(
                &bias_data,
                bias.shape(),
                &cpu_device,
            );
            client.layer_norm(a_cpu, &weight_cpu, &bias_cpu, eps)
        })
    }

    // --- Argmax/Argmin ---

    fn argmax(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "argmax", |client, a_cpu| {
            client.argmax(a_cpu, dim, keepdim)
        })
    }

    fn argmin(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        activation_fallback(a, &self.device_id, "argmin", |client, a_cpu| {
            client.argmin(a_cpu, dim, keepdim)
        })
    }

    // --- Cast ---

    fn cast(
        &self,
        a: &Tensor<WgpuRuntime>,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        use crate::dispatch_dtype;

        let src_dtype = a.dtype();
        let cpu = crate::runtime::fallback::CpuFallbackContext::new();

        dispatch_dtype!(src_dtype, T => {
            let a_cpu: crate::tensor::Tensor<crate::runtime::cpu::CpuRuntime> =
                cpu.tensor_from_gpu::<T, WgpuRuntime>(a);
            let result_cpu = cpu.client.cast(&a_cpu, dtype)?;

            // Copy back based on target dtype
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
        crate::runtime::fallback::where_cond_fallback(cond, x, y, &self.device_id, "where_cond")
    }
}

// ============================================================================
// ScalarOps Implementation
// ============================================================================

impl ScalarOps<WgpuRuntime> for WgpuClient {
    fn add_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        scalar_op_fallback(a, BinaryOp::Add, scalar, &self.device_id, "add_scalar")
    }

    fn sub_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        scalar_op_fallback(a, BinaryOp::Sub, scalar, &self.device_id, "sub_scalar")
    }

    fn mul_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        scalar_op_fallback(a, BinaryOp::Mul, scalar, &self.device_id, "mul_scalar")
    }

    fn div_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        scalar_op_fallback(a, BinaryOp::Div, scalar, &self.device_id, "div_scalar")
    }

    fn pow_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        scalar_op_fallback(a, BinaryOp::Pow, scalar, &self.device_id, "pow_scalar")
    }
}

// ============================================================================
// CompareOps Implementation
// ============================================================================

impl CompareOps<WgpuRuntime> for WgpuClient {
    fn eq(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        compare_op_fallback(a, b, CompareOp::Eq, &self.device_id, "eq")
    }

    fn ne(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        compare_op_fallback(a, b, CompareOp::Ne, &self.device_id, "ne")
    }

    fn lt(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        compare_op_fallback(a, b, CompareOp::Lt, &self.device_id, "lt")
    }

    fn le(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        compare_op_fallback(a, b, CompareOp::Le, &self.device_id, "le")
    }

    fn gt(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        compare_op_fallback(a, b, CompareOp::Gt, &self.device_id, "gt")
    }

    fn ge(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        compare_op_fallback(a, b, CompareOp::Ge, &self.device_id, "ge")
    }
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
        // [1,2,3] @ [[1,2],[3,4],[5,6]] = [22, 28]
        // [4,5,6] @ [[1,2],[3,4],[5,6]] = [49, 64]
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

        // 1.0 for equal, 0.0 for not equal
        assert_eq!(data, vec![1.0, 0.0, 1.0, 0.0]);
    }
}
