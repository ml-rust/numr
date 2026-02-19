//! CPU implementation of activation operations.

use crate::error::{Error, Result};
use crate::ops::impl_generic::activation::{dropout_impl, log_softmax_impl, softplus_impl};
use crate::ops::{ActivationOps, activation::normalize_softmax_dim};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{
        ActivationOp, activation_op_impl, dispatch_dtype, elu_impl, ensure_contiguous,
        leaky_relu_impl,
    },
    kernels,
};
use crate::tensor::Tensor;

/// ActivationOps implementation for CPU runtime.
impl ActivationOps<CpuRuntime> for CpuClient {
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

    fn leaky_relu(
        &self,
        a: &Tensor<CpuRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        leaky_relu_impl(self, a, negative_slope)
    }

    fn elu(&self, a: &Tensor<CpuRuntime>, alpha: f64) -> Result<Tensor<CpuRuntime>> {
        elu_impl(self, a, alpha)
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
            let a_ptr = a_contig.ptr();
            let out_ptr = out.ptr();

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
            let a_ptr = a_contig.ptr();
            let out_ptr = out.ptr();

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

    fn softplus(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        softplus_impl(self, a)
    }

    fn log_softmax(&self, a: &Tensor<CpuRuntime>, dim: isize) -> Result<Tensor<CpuRuntime>> {
        log_softmax_impl(self, a, dim)
    }

    fn dropout(
        &self,
        a: &Tensor<CpuRuntime>,
        p: f64,
        training: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        dropout_impl(self, a, p, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::ActivationOps;
    use crate::runtime::cpu::CpuDevice;

    #[test]
    fn test_log_softmax_basic() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let result = client.log_softmax(&input, -1).unwrap();
        let data: Vec<f32> = result.to_vec();

        // log_softmax should sum to something reasonable
        // exp(log_softmax) should sum to 1
        let exp_sum: f32 = data.iter().map(|x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);

        // Values should be negative (log of probability)
        for &v in &data {
            assert!(v < 0.0);
        }
    }

    #[test]
    fn test_log_softmax_2d() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let result = client.log_softmax(&input, -1).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Each row should independently sum (in exp space) to 1
        let row1_sum: f32 = data[0..3].iter().map(|x| x.exp()).sum();
        let row2_sum: f32 = data[3..6].iter().map(|x| x.exp()).sum();
        assert!((row1_sum - 1.0).abs() < 1e-5);
        assert!((row2_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dropout_training() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input = Tensor::<CpuRuntime>::ones(&[1000], crate::dtype::DType::F32, &device);
        let result = client.dropout(&input, 0.5, true).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Some values should be 0 (dropped), others should be 2.0 (scaled by 1/(1-0.5))
        let zeros = data.iter().filter(|&&v| v == 0.0).count();
        let scaled = data.iter().filter(|&&v| (v - 2.0).abs() < 1e-5).count();

        // With p=0.5, roughly half should be dropped (allow wide margin for randomness)
        assert!(zeros > 200, "too few zeros: {zeros}");
        assert!(zeros < 800, "too many zeros: {zeros}");
        assert_eq!(zeros + scaled, 1000);
    }

    #[test]
    fn test_dropout_inference() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let result = client.dropout(&input, 0.5, false).unwrap();
        let data: Vec<f32> = result.to_vec();

        // During inference, dropout is identity
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dropout_p_zero() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let result = client.dropout(&input, 0.0, true).unwrap();
        let data: Vec<f32> = result.to_vec();

        // p=0 means no dropout
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dropout_p_one() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let result = client.dropout(&input, 1.0, true).unwrap();
        let data: Vec<f32> = result.to_vec();

        // p=1 means all dropped
        for &v in &data {
            assert!((v).abs() < 1e-6);
        }
    }
}

unsafe fn softmax_non_last_dim<T: crate::dtype::Element>(
    a_ptr: *const T,
    out_ptr: *mut T,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) {
    unsafe {
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
}
