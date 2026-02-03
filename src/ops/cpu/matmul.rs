//! CPU implementation of matrix multiplication operations.

use crate::error::{Error, Result};
use crate::ops::{Kernel, MatmulOps};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
};
use crate::tensor::Tensor;

/// MatmulOps implementation for CPU runtime.
impl MatmulOps<CpuRuntime> for CpuClient {
    fn matmul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::matmul_output_shape;

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        let dtype = a.dtype();

        // Compute output shape
        let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        })?;

        // Get matrix dimensions (last two dims)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Require row-major contiguous tensors for SIMD-optimized packing
        // Non-contiguous tensors (transposed, views) are copied to contiguous layout
        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);

        // Calculate batch size
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Create output tensor
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let b_ptr = b_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        // Leading dimensions for contiguous row-major matrices
        let lda = k;
        let ldb = n;
        let ldc = n;

        // Dispatch based on dtype
        dispatch_dtype!(dtype, T => {
            unsafe {
                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let out_offset = batch * m * n;

                    <Self as Kernel<CpuRuntime>>::matmul::<T>(
                        self,
                        (a_ptr as *const T).add(a_offset),
                        (b_ptr as *const T).add(b_offset),
                        (out_ptr as *mut T).add(out_offset),
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                }
            }
        }, "matmul");

        Ok(out)
    }

    fn matmul_bias(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::ops::{matmul_bias_output_shape, validate_matmul_bias_dtypes};
        use crate::runtime::cpu::kernels::matmul_bias_kernel;

        // Validate dtypes using unified helper (ensures consistent error handling across backends)
        let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

        // Compute output shape (also validates bias shape)
        let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            },
        )?;

        // Get matrix dimensions (last two dims)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Require row-major contiguous tensors for SIMD-optimized packing
        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let bias_contig = ensure_contiguous(bias);

        // Calculate batch size
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Create output tensor
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let b_ptr = b_contig.storage().ptr();
        let bias_ptr = bias_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        // Leading dimensions for contiguous row-major matrices
        let lda = k;
        let ldb = n;
        let ldc = n;

        // Dispatch based on dtype
        dispatch_dtype!(dtype, T => {
            unsafe {
                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let out_offset = batch * m * n;

                    matmul_bias_kernel::<T>(
                        (a_ptr as *const T).add(a_offset),
                        (b_ptr as *const T).add(b_offset),
                        bias_ptr as *const T, // bias is 1D, same for all batches
                        (out_ptr as *mut T).add(out_offset),
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                }
            }
        }, "matmul_bias");

        Ok(out)
    }
}
