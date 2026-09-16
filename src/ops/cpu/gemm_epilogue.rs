//! CPU implementation of GEMM epilogue operations.

use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::matmul::matmul_dims_and_batches;
use crate::ops::{GemmActivation, GemmEpilogueOps};
use crate::ops::{matmul_bias_output_shape, validate_gemm_epilogue_dtypes};
use crate::runtime::cpu::helpers::{dispatch_dtype, ensure_contiguous};
use crate::runtime::cpu::kernels::{
    matmul_bias_activation_bwd_kernel, matmul_bias_activation_kernel, matmul_bias_residual_kernel,
};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl GemmEpilogueOps<CpuRuntime> for CpuClient {
    fn matmul_bias_activation(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        activation: GemmActivation,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_activation",
        )?;

        let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            },
        )?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        // Batch dims broadcast per dimension, so each output batch reads its own
        // source batch per operand; a wrapping batch count would read past a
        // `[k, n]` operand paired with a batched `a`.
        let (m, k, n, batch_size, a_idx, b_idx) =
            matmul_dims_and_batches(a_shape, b_shape, &out_shape);

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;

        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;

        // A zero-element output has nothing to compute. The dispatch below takes the
        // single-matmul branch for any `batch_size <= 1`, and a batch of 0 lands
        // there too — it would then write one full m*n tile into an empty
        // allocation. Return before that.
        if out.numel() == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let out_ptr = out.ptr();

        let lda = k;
        let ldb = n;
        let ldc = n;

        dispatch_dtype!(dtype, T => {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;

                if batch_size > 1 {
                    let min_len = self.rayon_min_len();
                    self.install_parallelism(|| {
                        (0..batch_size)
                            .into_par_iter()
                            .with_min_len(min_len)
                            .for_each(|batch| unsafe {
                            matmul_bias_activation_kernel::<T>(
                                (a_ptr as *const T).add(a_idx[batch] * m * k),
                                (b_ptr as *const T).add(b_idx[batch] * k * n),
                                bias_ptr as *const T,
                                (out_ptr as *mut T).add(batch * m * n),
                                m, n, k, lda, ldb, ldc,
                                activation,
                            );
                        });
                    });
                } else {
                    unsafe {
                        matmul_bias_activation_kernel::<T>(
                            a_ptr as *const T,
                            b_ptr as *const T,
                            bias_ptr as *const T,
                            out_ptr as *mut T,
                            m, n, k, lda, ldb, ldc,
                            activation,
                        );
                    }
                }
            }

            #[cfg(not(feature = "rayon"))]
            unsafe {
                for batch in 0..batch_size {
                    matmul_bias_activation_kernel::<T>(
                        (a_ptr as *const T).add(a_idx[batch] * m * k),
                        (b_ptr as *const T).add(b_idx[batch] * k * n),
                        bias_ptr as *const T,
                        (out_ptr as *mut T).add(batch * m * n),
                        m, n, k, lda, ldb, ldc,
                        activation,
                    );
                }
            }
        }, "matmul_bias_activation");

        Ok(out)
    }

    fn matmul_bias_residual(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        residual: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_residual",
        )?;
        if residual.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: residual.dtype(),
            });
        }

        let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            },
        )?;

        // Validate residual shape matches output shape
        if residual.shape() != out_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: out_shape.clone(),
                got: residual.shape().to_vec(),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        // Batch dims broadcast per dimension, so each output batch reads its own
        // source batch per operand; a wrapping batch count would read past a
        // `[k, n]` operand paired with a batched `a`.
        let (m, k, n, batch_size, a_idx, b_idx) =
            matmul_dims_and_batches(a_shape, b_shape, &out_shape);

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;
        let residual_contig = ensure_contiguous(residual)?;

        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device)?;

        // A zero-element output has nothing to compute. The dispatch below takes the
        // single-matmul branch for any `batch_size <= 1`, and a batch of 0 lands
        // there too — it would then write one full m*n tile into an empty
        // allocation. Return before that.
        if out.numel() == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let res_ptr = residual_contig.ptr();
        let out_ptr = out.ptr();

        let lda = k;
        let ldb = n;
        let ldc = n;

        dispatch_dtype!(dtype, T => {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;

                if batch_size > 1 {
                    let min_len = self.rayon_min_len();
                    self.install_parallelism(|| {
                        (0..batch_size)
                            .into_par_iter()
                            .with_min_len(min_len)
                            .for_each(|batch| unsafe {
                            matmul_bias_residual_kernel::<T>(
                                (a_ptr as *const T).add(a_idx[batch] * m * k),
                                (b_ptr as *const T).add(b_idx[batch] * k * n),
                                bias_ptr as *const T,
                                (res_ptr as *const T).add(batch * m * n),
                                (out_ptr as *mut T).add(batch * m * n),
                                m, n, k, lda, ldb, ldc,
                            );
                        });
                    });
                } else {
                    unsafe {
                        matmul_bias_residual_kernel::<T>(
                            a_ptr as *const T,
                            b_ptr as *const T,
                            bias_ptr as *const T,
                            res_ptr as *const T,
                            out_ptr as *mut T,
                            m, n, k, lda, ldb, ldc,
                        );
                    }
                }
            }

            #[cfg(not(feature = "rayon"))]
            unsafe {
                for batch in 0..batch_size {
                    matmul_bias_residual_kernel::<T>(
                        (a_ptr as *const T).add(a_idx[batch] * m * k),
                        (b_ptr as *const T).add(b_idx[batch] * k * n),
                        bias_ptr as *const T,
                        (res_ptr as *const T).add(batch * m * n),
                        (out_ptr as *mut T).add(batch * m * n),
                        m, n, k, lda, ldb, ldc,
                    );
                }
            }
        }, "matmul_bias_residual");

        Ok(out)
    }

    fn matmul_bias_activation_bwd(
        &self,
        grad: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        activation: GemmActivation,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let dtype = validate_gemm_epilogue_dtypes(
            a.dtype(),
            b.dtype(),
            bias.dtype(),
            "matmul_bias_activation_bwd",
        )?;
        if grad.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: grad.dtype(),
            });
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let out_shape = matmul_bias_output_shape(a_shape, b_shape, bias.shape()).ok_or(
            Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            },
        )?;
        if grad.shape() != out_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: out_shape,
                got: grad.shape().to_vec(),
            });
        }
        let (m, k, n, batch_size, a_idx, b_idx) =
            matmul_dims_and_batches(a_shape, b_shape, &out_shape);
        // An operand whose batch count equals the output's is read once per
        // batch, so its gradient slice is written in place. A broadcast operand
        // is read by several batches, so its slice accumulates their gradients.
        let a_owns_batch = batch_count(a_shape) == batch_size;
        let b_owns_batch = batch_count(b_shape) == batch_size;

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let bias_contig = ensure_contiguous(bias)?;
        let grad_contig = ensure_contiguous(grad)?;

        // Output gradients. `d_b` and `d_bias` accumulate, so they start at zero;
        // `d_a` starts at zero only when it accumulates.
        let d_a = if a_owns_batch {
            Tensor::<CpuRuntime>::empty(a_shape, dtype, &self.device)?
        } else {
            Tensor::<CpuRuntime>::zeros(a_shape, dtype, &self.device)?
        };
        let d_b = Tensor::<CpuRuntime>::zeros(b_shape, dtype, &self.device)?;
        let d_bias_full = Tensor::<CpuRuntime>::zeros(&[n], dtype, &self.device)?;

        // No batch contributes: every gradient sums over nothing and stays at the
        // additive identity it was seeded with.
        if batch_size == 0 {
            return Ok((d_a, d_b, d_bias_full));
        }

        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let bias_ptr = bias_contig.ptr();
        let grad_ptr = grad_contig.ptr();
        let d_a_ptr = d_a.ptr();
        let d_b_ptr = d_b.ptr();
        let d_bias_ptr = d_bias_full.ptr();

        let lda = k;
        let ldb = n;
        let ld_grad = n;

        dispatch_dtype!(dtype, T => {
            // Per-batch scratch for whatever accumulates: `d_bias` always, `d_a`
            // and `d_b` when the operand is broadcast.
            let mut temp_d_a = vec![T::zero(); if a_owns_batch { 0 } else { m * k }];
            let mut temp_d_b = vec![T::zero(); if b_owns_batch { 0 } else { k * n }];
            let mut temp_d_bias = vec![T::zero(); n];

            for batch in 0..batch_size {
                let a_off = a_idx[batch] * m * k;
                let b_off = b_idx[batch] * k * n;
                unsafe {
                    let d_a_dst = if a_owns_batch {
                        (d_a_ptr as *mut T).add(a_off)
                    } else {
                        temp_d_a.as_mut_ptr()
                    };
                    let d_b_dst = if b_owns_batch {
                        (d_b_ptr as *mut T).add(b_off)
                    } else {
                        temp_d_b.as_mut_ptr()
                    };
                    matmul_bias_activation_bwd_kernel::<T>(
                        (grad_ptr as *const T).add(batch * m * n),
                        (a_ptr as *const T).add(a_off),
                        (b_ptr as *const T).add(b_off),
                        bias_ptr as *const T,
                        d_a_dst,
                        d_b_dst,
                        temp_d_bias.as_mut_ptr(),
                        m, n, k, lda, ldb, ld_grad,
                        activation,
                    );

                    if !a_owns_batch {
                        for (i, v) in temp_d_a.iter().enumerate() {
                            *(d_a_ptr as *mut T).add(a_off + i) += *v;
                        }
                    }
                    if !b_owns_batch {
                        for (i, v) in temp_d_b.iter().enumerate() {
                            *(d_b_ptr as *mut T).add(b_off + i) += *v;
                        }
                    }
                    for (j, v) in temp_d_bias.iter().enumerate() {
                        *(d_bias_ptr as *mut T).add(j) += *v;
                    }
                }
            }
        }, "matmul_bias_activation_bwd");

        Ok((d_a, d_b, d_bias_full))
    }
}

/// Product of an operand's batch dims; 1 for a plain matrix.
fn batch_count(shape: &[usize]) -> usize {
    shape.iter().take(shape.len().saturating_sub(2)).product()
}
