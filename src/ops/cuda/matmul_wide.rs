//! CUDA `matmul_wide`: a half matmul written as its F32 accumulator.
//!
//! The WMMA kernels already accumulate in F32 and narrow only in their store,
//! so the `f32out` instantiations (`matmul_wmma_*_f32out_*`) give the same
//! product at tensor-core speed with no operand casts. Where the WMMA policy
//! does not admit a shape as it stands, the op pads the inputs exactly as
//! `matmul` does; where the device has no WMMA path for the dtype at all, it
//! casts both operands to F32 and runs the F32 tiled GEMM, which is the same
//! precision class (exact F16×F16 products, F32 sums) at the slower rate.
//! Every other dtype delegates to `matmul`, which already widens I8 to I32.
//!
//! `#[path]`-included into `runtime::cuda::ops::tensor`, so `super` here is
//! that module.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::matmul::matmul_mkn;
use crate::ops::{MatmulOps, TypeConversionOps, matmul_output_shape};
use crate::runtime::cuda::kernels::{
    launch_matmul_wmma_f32out_batched_kernel, launch_matmul_wmma_f32out_kernel, use_wmma,
};
use crate::runtime::cuda::ops::matmul_broadcast::resolve_batched_operands;
use crate::runtime::cuda::ops::wmma_pad::pad_ab_for_wmma;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Device, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

impl CudaClient {
    /// Carries the body of [`crate::ops::MatmulOps::matmul_wide`] for the
    /// CUDA runtime; the trait method delegates here.
    pub(super) fn matmul_wide_impl(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_binary_dtypes(a, b)?;

        // Only the half floats widen here; `matmul` already returns I32 for I8
        // and the element dtype for the rest.
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return self.matmul(a, b);
        }

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

        // Same empty-shape rules as `matmul`: a grid extent of 0 is a launch
        // error, and a zero-length contraction is a non-empty output of zeros.
        if out_shape.iter().product::<usize>() == 0 {
            return Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, &self.device);
        }
        if k == 0 {
            return Tensor::<CudaRuntime>::zeros(&out_shape, DType::F32, &self.device);
        }

        let caps = self.device.profile().caps;
        if use_wmma(dtype, caps, m, n, k) {
            return self.matmul_wide_wmma(a, b, dtype, &out_shape, m, n, k);
        }
        // Rank-1 operands never pad: the spec addresses two dims of each. Zero-
        // padding is exact, and the extra N columns are sliced off the F32
        // result. `pad_ab_for_wmma` (runtime/cuda/ops/wmma_pad.rs) carries the
        // decision and the pad itself, shared with matmul/matmul_bias and
        // matmul_bias_activation/residual.
        if let Some((a_pad, b_pad, n_pad, k_pad)) =
            pad_ab_for_wmma(self, a, b, dtype, caps, m, n, k)?
        {
            let mut pad_shape = out_shape.clone();
            let last = pad_shape.len() - 1;
            pad_shape[last] = n_pad;
            let out_pad =
                self.matmul_wide_wmma(&a_pad, &b_pad, dtype, &pad_shape, m, n_pad, k_pad)?;
            return out_pad.narrow(-1, 0, n)?.contiguous();
        }

        // No tensor-core path for this dtype on this device: the F32 tiled
        // GEMM over cast operands is the same arithmetic without the speed.
        let a32 = self.cast(a, DType::F32)?;
        let b32 = self.cast(b, DType::F32)?;
        self.matmul(&a32, &b32)
    }

    /// Launch the `f32out` WMMA kernel, 2-D or batched by the output rank.
    ///
    /// `out_shape` is the shape the kernel writes, which is the padded one on
    /// the padding path.
    #[allow(clippy::too_many_arguments)]
    fn matmul_wide_wmma(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        dtype: DType,
        out_shape: &[usize],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let batch: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product::<usize>()
            .max(1);
        let out = Tensor::<CudaRuntime>::empty(out_shape, DType::F32, &self.device)?;

        // A single batch launches the 2-D kernel whatever the rank, as
        // `matmul` does: the operands are flat `M×K` and `K×N` buffers.
        if batch > 1 {
            // Pointers and batch counts must come from the same tensors, so
            // both come from one resolver, as in `matmul_batched_native`.
            let operands = resolve_batched_operands(a, b, out_shape)?;
            let a_contig = ensure_contiguous(&operands.a)?;
            let b_contig = ensure_contiguous(&operands.b)?;
            unsafe {
                launch_matmul_wmma_f32out_batched_kernel(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    b_contig.ptr(),
                    out.ptr(),
                    batch,
                    m,
                    n,
                    k,
                    operands.a_batch,
                    operands.b_batch,
                )?;
            }
            return Ok(out);
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        unsafe {
            launch_matmul_wmma_f32out_kernel(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                m,
                n,
                k,
            )?;
        }
        Ok(out)
    }
}
