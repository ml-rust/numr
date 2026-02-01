//! CUDA tensor operations implementation
//!
//! This module implements TensorOps, ScalarOps, CompareOps, and LogicalOps
//! for the CUDA runtime using native CUDA kernels:
//! - Element-wise, unary, scalar, reduction, and activation ops
//! - Native tiled matrix multiplication (shared memory optimization)
//!
//! Kernels are compiled from .cu files by build.rs and loaded at runtime.
//!
//! # Performance Characteristics
//!
//! ## Native GPU Operations (Fast Path)
//!
//! Operations on tensors with matching shapes run entirely on GPU:
//! - Binary ops (add, sub, mul, div, pow, max, min)
//! - Unary ops (neg, abs, sqrt, exp, log, sin, cos, tan, tanh, etc.)
//! - Scalar ops (add_scalar, mul_scalar, etc.)
//! - Reductions (sum, max, min) - single dimension
//! - Activations (relu, sigmoid, softmax)
//! - Matrix multiplication (native tiled GEMM with shared memory)
//!
//! ## CPU Fallback (Slow Path)
//!
//! The following operations trigger GPU→CPU→GPU transfers, causing significant overhead:
//!
//! 1. **Broadcasting binary operations**: When tensor shapes don't match (e.g., `[3, 4] + [4]`),
//!    the operation falls back to CPU. This involves:
//!    - Copying both tensors from GPU to CPU
//!    - Computing the result on CPU
//!    - Copying the result back to GPU
//!
//! 2. **Multi-dimension reductions**: Reducing over multiple dimensions at once
//!    (e.g., `sum(&[0, 1])`) falls back to CPU.
//!
//! 3. **Unsupported dtypes for scalar ops**: Non-F32/F64 scalar operations use CPU.
//!
//! ## Recommendations
//!
//! - Pre-broadcast tensors to matching shapes before binary operations
//! - Use single-dimension reductions and chain them if needed
//! - Use F32 or F64 for best GPU performance

mod compare;
mod helpers;
mod logical;
mod scalar;
mod statistics;
mod tensor;

#[cfg(test)]
mod tests {
    use crate::ops::{
        ActivationOps, BinaryOps, IndexingOps, MatmulOps, NormalizationOps, ReduceOps, TensorOps,
    };
    use crate::runtime::Runtime;
    use crate::runtime::cuda::{CudaDevice, CudaRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_cuda_tensor_add() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CudaRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_cuda_tensor_matmul_2x2() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CudaRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

        let c = client.matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cuda_tensor_matmul_3x2_2x4() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
        let b = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        );

        let c = client.matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[3, 4]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(
            result,
            [
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_cuda_tensor_relu() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-1.0f32, 0.0, 1.0, -2.0], &[4], &device);
        let b = client.relu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_cuda_tensor_sum() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

        let b = client.sum(&a, &[1], false).unwrap();

        assert_eq!(b.shape(), &[2]);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [6.0, 15.0]);
    }

    #[test]
    fn test_cuda_tensor_silu() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device);
        let b = client.silu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // SiLU(x) = x / (1 + exp(-x))
        // SiLU(0) = 0
        // SiLU(1) ≈ 0.731
        // SiLU(-1) ≈ -0.269
        assert!((result[2] - 0.0).abs() < 1e-5); // SiLU(0) = 0
        assert!((result[3] - 0.7310586).abs() < 1e-4); // SiLU(1) ≈ 0.731
        assert!((result[1] - (-0.2689414)).abs() < 1e-4); // SiLU(-1) ≈ -0.269
    }

    #[test]
    fn test_cuda_tensor_gelu() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device);
        let b = client.gelu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // GELU(0) = 0
        // GELU is approximately x for large positive x
        // GELU is approximately 0 for large negative x
        assert!((result[2] - 0.0).abs() < 1e-5); // GELU(0) = 0
        assert!((result[3] - 0.8413).abs() < 0.01); // GELU(1) ≈ 0.841
        assert!((result[4] - 1.9545).abs() < 0.01); // GELU(2) ≈ 1.955
    }

    #[test]
    fn test_cuda_tensor_rms_norm() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        );
        let weight = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);

        let out = client.rms_norm(&input, &weight, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.739
        let rms1 = (30.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[0] - 1.0 / rms1).abs() < 1e-3); // Wider tolerance for GPU
        assert!((result[1] - 2.0 / rms1).abs() < 1e-3);
        assert!((result[2] - 3.0 / rms1).abs() < 1e-3);
        assert!((result[3] - 4.0 / rms1).abs() < 1e-3);

        // Row 2: [2, 4, 6, 8]
        let rms2 = (120.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[4] - 2.0 / rms2).abs() < 1e-3);
    }

    #[test]
    fn test_cuda_tensor_layer_norm() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        );
        let weight = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);
        let bias = Tensor::<CudaRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

        let out = client.layer_norm(&input, &weight, &bias, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], mean = 2.5, var = 1.25, std = 1.118
        let mean1 = 2.5f32;
        let var1 = ((1.0 - mean1).powi(2)
            + (2.0 - mean1).powi(2)
            + (3.0 - mean1).powi(2)
            + (4.0 - mean1).powi(2))
            / 4.0;
        let std1 = (var1 + 1e-5).sqrt();
        assert!((result[0] - (1.0 - mean1) / std1).abs() < 1e-3); // Wider tolerance for GPU
        assert!((result[1] - (2.0 - mean1) / std1).abs() < 1e-3);
        assert!((result[2] - (3.0 - mean1) / std1).abs() < 1e-3);
        assert!((result[3] - (4.0 - mean1) / std1).abs() < 1e-3);

        // Verify normalized outputs sum to approximately 0 (zero-centered)
        let row1_sum: f32 = result[0..4].iter().sum();
        assert!(row1_sum.abs() < 1e-3);
    }

    #[test]
    fn test_cuda_tensor_argmax() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device);

        // argmax along dim=1 (find max index in each row)
        let out = client.argmax(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [1, 2]); // Row 0: max at index 1 (5.0), Row 1: max at index 2 (6.0)

        // argmax along dim=0 (find max index in each column)
        let out = client.argmax(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [1, 0, 1]); // Col 0: max at 1 (4.0), Col 1: max at 0 (5.0), Col 2: max at 1 (6.0)

        // Test keepdim=true
        let out = client.argmax(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [1, 2]);
    }

    #[test]
    fn test_cuda_tensor_argmin() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device);

        // argmin along dim=1 (find min index in each row)
        let out = client.argmin(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [0, 1]); // Row 0: min at index 0 (1.0), Row 1: min at index 1 (2.0)

        // argmin along dim=0 (find min index in each column)
        let out = client.argmin(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [0, 1, 0]); // Col 0: min at 0 (1.0), Col 1: min at 1 (2.0), Col 2: min at 0 (3.0)

        // Test keepdim=true
        let out = client.argmin(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [0, 1]);
    }
}
