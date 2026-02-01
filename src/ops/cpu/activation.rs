//! CPU implementation of activation operations.

use crate::dtype::Element;
use crate::error::{Error, Result};
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
