//! Matrix multiplication operations for CUDA runtime.
//!
//! `matmul_wide` delegates to `super::matmul_wide`, which carries its body.
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::matmul::matmul_mkn;
use crate::ops::{
    BinaryOps, MatmulOps, ShapeOps, matmul_bias_output_shape, matmul_output_dtype,
    matmul_output_shape, validate_matmul_bias_dtypes,
};
use crate::runtime::cuda::kernels::int_matmul_has_kernel;
use crate::runtime::cuda::ops::helpers::{
    matmul_batched_native, matmul_bias_batched_native, matmul_bias_native, matmul_native,
};
use crate::runtime::cuda::ops::wmma_pad::pad_ab_for_wmma;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Device, validate_binary_dtypes};
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
        let (m, k, n) = matmul_mkn(a_shape, b_shape);

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

        // A zero-element output has nothing to compute, and the matmul launchers
        // derive their grid extents from `m`, `n` and the batch count without
        // flooring them. A grid extent of 0 is a launch error, so the empty
        // result is returned before any launch.
        if out_shape.iter().product::<usize>() == 0 {
            // I8 widens to I32, so the empty result carries the OUTPUT dtype.
            let out_dtype = matmul_output_dtype(dtype);
            return Tensor::<CudaRuntime>::empty(&out_shape, out_dtype, &self.device);
        }

        // A zero-length contraction leaves a NON-empty output whose every element
        // is the sum of nothing. CPU answers zeros; the kernels would launch over
        // empty operand buffers and read off the end, so answer it here.
        if k == 0 {
            return Tensor::<CudaRuntime>::zeros(
                &out_shape,
                matmul_output_dtype(dtype),
                &self.device,
            );
        }

        // Native tiled CUDA kernel. The integer dtypes are gated by
        // `int_matmul_has_kernel`, which is the same predicate the launcher
        // uses, so this match and `matmul_int.cu`'s instantiation list cannot
        // drift apart. FP8 has its own kernels in `kernels/matmul_fp8.cu` and
        // accumulates in F32, matching CPU.
        match dtype {
            DType::F32
            | DType::F64
            | DType::F16
            | DType::BF16
            | DType::FP8E4M3
            | DType::FP8E5M2 => {
                // The WMMA kernel handles any M, N, K: it zero-fills past every
                // edge and masks its store. A row stride (K for A, N for B)
                // that is not a multiple of `WMMA_STAGE_HALVES` makes the kernel
                // stage that operand one element at a time instead of through
                // its 128-bit path. That costs a fixed fraction of the GEMM's
                // work; a pad costs one pass over each copied operand plus the
                // output narrow. So the op pads only when the GEMM is heavy
                // enough per copied element for the copy to pay
                // (`WMMA_PAD_MIN_WORK_PER_COPIED_ELEMENT`); every other shape,
                // ragged M included, launches as it is. Zero-padding is exact:
                // the extra K contributes 0, and the extra N columns are sliced
                // off. The rule is per slice, so the batched form applies it
                // too: `pad` takes the last two dims and leaves the batch dims
                // as they are, and the batched launcher derives its batch
                // strides from the padded shapes.
                //
                // `use_wmma_after_padding` is the same predicate the launcher uses
                // to pick the WMMA kernel (src/runtime/cuda/kernels/loader/matmul_wmma_policy.rs),
                // gated on this device's real capabilities. Padding a BF16 operand
                // on a device without native bf16 (caps.bf16) would allocate and
                // copy for a WMMA path that never fires.
                //
                // The pad spec below addresses two dims of each operand, so a
                // rank-1 operand (a `[k]` row or column) never takes this branch.
                // `pad_ab_for_wmma` (runtime/cuda/ops/wmma_pad.rs) carries the
                // decision and the pad itself, shared with matmul_bias below and
                // with matmul_bias_activation/residual and matmul_wide.
                let caps = self.device.profile().caps;
                if let Some((a_pad, b_pad, n_pad, k_pad)) =
                    pad_ab_for_wmma(self, a, b, dtype, caps, m, n, k)?
                {
                    let out_pad = if batch_size > 1 {
                        matmul_batched_native(
                            self, &a_pad, &b_pad, dtype, batch_size, m, k_pad, n_pad,
                        )?
                    } else {
                        matmul_native(self, &a_pad, &b_pad, dtype, m, k_pad, n_pad)?
                    };
                    // Slice N (last dim) back via negative indexing, NOT dim 1:
                    // the output can carry leading batch dims (e.g. a 3D
                    // [1, m, n] from the padded encoder forward).
                    out_pad.narrow(-1, 0, n)?.contiguous()
                } else if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    matmul_native(self, a, b, dtype, m, k, n)
                }
            }
            // Integers never take the WMMA padding branch above, so they only
            // need the two native entry points. I8 is included and returns an
            // I32 tensor: the helpers allocate through `int_matmul_output_dtype`,
            // which mirrors CPU's quantized-accumulation branch.
            d if int_matmul_has_kernel(d) => {
                if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    matmul_native(self, a, b, dtype, m, k, n)
                }
            }
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "matmul",
            }),
        }
    }

    fn matmul_wide(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        self.matmul_wide_impl(a, b)
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

        let (m, k, n) = matmul_mkn(a_shape, b_shape);

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

        // A zero-element output has nothing to compute, and the matmul launchers
        // derive their grid extents from `m`, `n` and the batch count without
        // flooring them. A grid extent of 0 is a launch error, so the empty
        // result is returned before any launch.
        if out_shape.iter().product::<usize>() == 0 {
            // I8 widens to I32, so the empty result carries the OUTPUT dtype.
            let out_dtype = matmul_output_dtype(dtype);
            return Tensor::<CudaRuntime>::empty(&out_shape, out_dtype, &self.device);
        }

        // A zero-length contraction leaves a NON-empty output whose every element
        // is just the bias. CPU seeds its accumulator with the bias and adds
        // nothing; the kernels would read off the end of the empty operands.
        if k == 0 {
            let zeros =
                Tensor::<CudaRuntime>::zeros(&out_shape, matmul_output_dtype(dtype), &self.device)?;
            return self.add(&zeros, bias);
        }

        // Native tiled CUDA kernel with fused bias. FP8 and the integers are
        // included: CPU seeds its wide accumulator with the bias, so composing
        // matmul with a separate add would narrow twice and report a different
        // number.
        match dtype {
            DType::F32
            | DType::F64
            | DType::F16
            | DType::BF16
            | DType::FP8E4M3
            | DType::FP8E5M2 => {
                // Same padding rule as matmul(), batched included: only N and
                // K, and only when the pad pass pays against scalar staging.
                // The bias is [n], so it pads to [n_pad]. Rank-1 operands never
                // pad: the spec addresses two dims of each.
                let caps = self.device.profile().caps;
                if let Some((a_pad, b_pad, n_pad, k_pad)) =
                    pad_ab_for_wmma(self, a, b, dtype, caps, m, n, k)?
                {
                    let bias_pad = self.pad(bias, &[0, n_pad - n], 0.0)?;
                    let out_pad = if batch_size > 1 {
                        matmul_bias_batched_native(
                            self, &a_pad, &b_pad, &bias_pad, dtype, batch_size, m, k_pad, n_pad,
                        )?
                    } else {
                        matmul_bias_native(self, &a_pad, &b_pad, &bias_pad, dtype, m, k_pad, n_pad)?
                    };
                    // Slice N (last dim) back via negative indexing, see matmul().
                    out_pad.narrow(-1, 0, n)?.contiguous()
                } else if batch_size > 1 {
                    matmul_bias_batched_native(self, a, b, bias, dtype, batch_size, m, k, n)
                } else {
                    matmul_bias_native(self, a, b, bias, dtype, m, k, n)
                }
            }
            // Integers have their own fused-bias kernels in `matmul_int.cu`, and
            // fused is the only correct form: the bias seeds the 128-bit
            // accumulator, so composing a matmul with an elementwise add would
            // saturate the product and then wrap the bias into the element type.
            // I8 widens here as well: the bias is I32 and so is the result,
            // because the bias seeds the accumulator the widened output exists
            // to carry (see `ops/matmul_dtype.rs`).
            d if int_matmul_has_kernel(d) => {
                if batch_size > 1 {
                    matmul_bias_batched_native(self, a, b, bias, dtype, batch_size, m, k, n)
                } else {
                    matmul_bias_native(self, a, b, bias, dtype, m, k, n)
                }
            }
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "matmul_bias",
            }),
        }
    }
}
