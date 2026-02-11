//! Matrix multiplication operations for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    MatmulOps, matmul_bias_output_shape, matmul_output_shape, validate_matmul_bias_dtypes,
};
use crate::runtime::cuda::ops::helpers::{
    matmul_batched_native, matmul_bias_batched_native, matmul_bias_native, matmul_native,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::fallback::{matmul_fallback, validate_binary_dtypes};
use crate::tensor::Tensor;

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
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
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
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
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
