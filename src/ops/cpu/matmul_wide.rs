//! CPU `matmul_wide`: a half matmul written as its F32 accumulator.
//!
//! F16 and BF16 already run their product in F32 (`matmul_wide_kernel`,
//! `gemv_bt_wide_kernel`); this op keeps that F32 result instead of narrowing
//! it. Every other dtype delegates to `matmul`, which already widens I8 to
//! I32 and writes its own dtype otherwise.
//!
//! The dispatch mirrors `matmul` on the paths a half operand can take there:
//! the small-M transposed-B GEMV and the general tiled path, each batched
//! and broadcast the same way. The transposed-B tiled path is F32/F64 only in
//! `matmul` and so has no wide form.
//!
//! `#[path]`-included into `runtime::cpu::ops`, so `super` here is that module.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{MatmulOps, matmul_output_shape};
use crate::runtime::cpu::kernels::{gemv_bt_wide_kernel, matmul_wide_kernel};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
};
use crate::tensor::Tensor;

/// Column chunk width of the GEMV-BT path, shared with `matmul`.
///
/// Fixed for the same reason as there: the chunk boundaries move float
/// rounding, so they must be a function of the shape and never of the pool.
const GEMV_COLUMN_CHUNK_WIDTH: usize = 64;

impl CpuClient {
    /// Carries the body of [`crate::ops::MatmulOps::matmul_wide`] for the CPU
    /// runtime; the trait method delegates here.
    pub(super) fn matmul_wide_impl(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        let dtype = a.dtype();

        // Only the half floats widen here. I8 already comes back as I32 from
        // `matmul`, and every other dtype writes itself.
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return self.matmul(a, b);
        }

        let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        })?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let (m, k, n, batch_size, a_batch_idx, b_batch_idx) =
            crate::ops::matmul::matmul_dims_and_batches(a_shape, b_shape, &out_shape);

        let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::F32, &self.device)?;

        // A zero-element output has nothing to compute; see `matmul` for why
        // the single-matmul branch below must not see it.
        if out_shape.iter().product::<usize>() == 0 {
            return Ok(out);
        }

        // Addresses, not pointers: the rayon closures below need `Sync`
        // captures, and every use casts at the point of the call.
        let out_addr = out.ptr();
        let ldc = n;

        // Small M against a transposed weight: dot A rows against B's own
        // `[N, K]` rows, as `matmul` does, without materializing the view.
        if m <= 16 && crate::ops::matmul::is_transposed_b(b_shape, b.strides(), k, n) {
            let a_contig = ensure_contiguous(a)?;
            let a_ptr = a_contig.ptr();
            let b_ptr = b.ptr();

            dispatch_dtype!(dtype, T => {
                // SAFETY: `a_contig` is contiguous `[.., M, K]`, `b` is the
                // transposed view of a contiguous `[.., N, K]` buffer, and each
                // batch writes a disjoint `m * n` block of the F32 output.
                for batch in 0..batch_size {
                    unsafe {
                        self.gemv_bt_wide_columns::<T>(
                            (a_ptr as *const T).add(a_batch_idx[batch] * m * k),
                            (b_ptr as *const T).add(b_batch_idx[batch] * n * k),
                            (out_addr as *mut f32).add(batch * m * n),
                            m, n, k, ldc,
                        );
                    }
                }
            }, "matmul_wide_gemv_bt");

            return Ok(out);
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let a_ptr = a_contig.ptr();
        let b_ptr = b_contig.ptr();
        let (lda, ldb) = (k, n);

        dispatch_dtype!(dtype, T => {
            // One batch element, shared by the serial and parallel loops so
            // the offset arithmetic cannot drift between them.
            //
            // SAFETY: every pointer derives from a contiguous tensor of the
            // validated shape, and each batch writes a disjoint m*n block.
            let run_batch = |batch: usize| unsafe {
                matmul_wide_kernel::<T>(
                    (a_ptr as *const T).add(a_batch_idx[batch] * m * k),
                    (b_ptr as *const T).add(b_batch_idx[batch] * k * n),
                    (out_addr as *mut f32).add(batch * m * n),
                    m, n, k, lda, ldb, ldc,
                );
            };

            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                use super::matmul_columns::{column_chunk_count, matmul_wide_columns};

                // Same axis rule as `matmul`: columns when they offer more
                // units than the batch axis, batches otherwise, never both.
                if let Some(chunks) = column_chunk_count(batch_size, m, n, k) {
                    for batch in 0..batch_size {
                        unsafe {
                            matmul_wide_columns::<T>(
                                self,
                                (a_ptr as *const T).add(a_batch_idx[batch] * m * k),
                                (b_ptr as *const T).add(b_batch_idx[batch] * k * n),
                                (out_addr as *mut f32).add(batch * m * n),
                                m, n, k, lda, ldb, ldc, chunks,
                            );
                        }
                    }
                } else if batch_size > 1 {
                    let min_len = self.rayon_min_len();
                    self.install_parallelism(|| {
                        (0..batch_size)
                            .into_par_iter()
                            .with_min_len(min_len)
                            .for_each(run_batch);
                    });
                } else {
                    run_batch(0);
                }
            }

            #[cfg(not(feature = "rayon"))]
            for batch in 0..batch_size {
                run_batch(batch);
            }
        }, "matmul_wide");

        Ok(out)
    }

    /// One batch slice of the GEMV-BT path, split over fixed-width column
    /// chunks when the pool can use them. The chunk list depends on `n` alone,
    /// so the arithmetic is the same on every machine; only the scheduling of
    /// the chunks varies.
    ///
    /// Pointers arrive already offset to the batch slice.
    ///
    /// # Safety
    /// Same as [`gemv_bt_wide_kernel`]: `a` valid for `m * k` reads, `b_nk`
    /// for `n * k`, `out` for `m * ldc` f32 writes, `out` aliasing neither.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemv_bt_wide_columns<T: crate::dtype::Element>(
        &self,
        a: *const T,
        b_nk: *const T,
        out: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
    ) {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let chunk = GEMV_COLUMN_CHUNK_WIDTH;
            if n > chunk {
                // Addresses, not pointers: the closure must be `Sync`, and the
                // chunks write disjoint column ranges of `out`.
                let elem = std::mem::size_of::<T>();
                let out_elem = std::mem::size_of::<f32>();
                let (a_addr, b_addr, out_addr) = (a as usize, b_nk as usize, out as usize);

                self.install_parallelism(|| {
                    (0..n).into_par_iter().step_by(chunk).for_each(|col_start| {
                        let cols = (col_start + chunk).min(n) - col_start;
                        unsafe {
                            gemv_bt_wide_kernel::<T>(
                                a_addr as *const T,
                                (b_addr + col_start * k * elem) as *const T,
                                (out_addr + col_start * out_elem) as *mut f32,
                                m,
                                cols,
                                k,
                                ldc,
                            );
                        }
                    });
                });
                return;
            }
        }

        unsafe {
            gemv_bt_wide_kernel::<T>(a, b_nk, out, m, n, k, ldc);
        }
    }
}

#[cfg(all(test, feature = "f16"))]
mod tests {
    use crate::dtype::DType;
    use crate::ops::{MatmulOps, TypeConversionOps};
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    fn values(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + seed * 11) % 251) as f32) * 0.004 - 0.5)
            .collect()
    }

    /// Reference: the same operands cast to F32 and multiplied there.
    fn reference(c: &CpuClient, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Vec<f32> {
        let a32 = c.cast(a, DType::F32).expect("cast a");
        let b32 = c.cast(b, DType::F32).expect("cast b");
        c.matmul(&a32, &b32).expect("matmul").to_vec::<f32>()
    }

    fn assert_close(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-4 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "{label}: element {i}: {g} vs {w}");
        }
    }

    fn half(
        c: &CpuClient,
        dev: &CpuDevice,
        data: &[f32],
        shape: &[usize],
        dtype: DType,
    ) -> Tensor<CpuRuntime> {
        let t = Tensor::<CpuRuntime>::from_slice(data, shape, dev).expect("tensor");
        c.cast(&t, dtype).expect("cast")
    }

    #[test]
    fn wide_output_is_f32_and_matches_f32_route() {
        let (c, dev) = client();
        for dtype in [DType::F16, DType::BF16] {
            let a = half(&c, &dev, &values(37 * 24, 1), &[37, 24], dtype);
            let b = half(&c, &dev, &values(24 * 40, 2), &[24, 40], dtype);
            let out = c.matmul_wide(&a, &b).expect("matmul_wide");
            assert_eq!(out.dtype(), DType::F32);
            assert_eq!(out.shape(), &[37, 40]);
            assert_close(&out.to_vec::<f32>(), &reference(&c, &a, &b), "2-D");
        }
    }

    #[test]
    fn wide_batched_broadcast_matches_f32_route() {
        let (c, dev) = client();
        let a = half(
            &c,
            &dev,
            &values(2 * 3 * 5 * 8, 1),
            &[2, 3, 5, 8],
            DType::F16,
        );
        let b = half(&c, &dev, &values(3 * 8 * 7, 2), &[1, 3, 8, 7], DType::F16);
        let out = c.matmul_wide(&a, &b).expect("matmul_wide");
        assert_eq!(out.shape(), &[2, 3, 5, 7]);
        assert_close(&out.to_vec::<f32>(), &reference(&c, &a, &b), "batched");
    }

    /// The transposed-weight decode shape takes the GEMV-BT path, with enough
    /// columns to split into chunks.
    #[test]
    fn wide_transposed_b_small_m_matches_f32_route() {
        let (c, dev) = client();
        let a = half(&c, &dev, &values(4 * 96, 1), &[4, 96], DType::BF16);
        let w = half(&c, &dev, &values(200 * 96, 2), &[200, 96], DType::BF16);
        let b = w.transpose(0, 1).expect("transpose");
        let out = c.matmul_wide(&a, &b).expect("matmul_wide");
        assert_eq!(out.dtype(), DType::F32);
        assert_close(&out.to_vec::<f32>(), &reference(&c, &a, &b), "gemv_bt");
    }

    #[test]
    fn wide_non_half_delegates_to_matmul() {
        let (c, dev) = client();
        let a = Tensor::<CpuRuntime>::from_slice(&values(6, 1), &[2, 3], &dev).expect("a");
        let b = Tensor::<CpuRuntime>::from_slice(&values(12, 2), &[3, 4], &dev).expect("b");
        let wide = c.matmul_wide(&a, &b).expect("wide").to_vec::<f32>();
        let plain = c.matmul(&a, &b).expect("matmul").to_vec::<f32>();
        assert_eq!(wide, plain);

        let ai =
            Tensor::<CpuRuntime>::from_slice(&[100i8, 100, 100, 100], &[1, 4], &dev).expect("ai");
        let bi =
            Tensor::<CpuRuntime>::from_slice(&[100i8, 100, 100, 100], &[4, 1], &dev).expect("bi");
        let out = c.matmul_wide(&ai, &bi).expect("wide i8");
        assert_eq!(out.dtype(), DType::I32);
        assert_eq!(out.to_vec::<i32>(), vec![40_000]);
    }

    #[test]
    fn wide_empty_output_is_f32() {
        let (c, dev) = client();
        let a = half(&c, &dev, &[], &[0, 4], DType::F16);
        let b = half(&c, &dev, &values(12, 2), &[4, 3], DType::F16);
        let out = c.matmul_wide(&a, &b).expect("empty");
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.shape(), &[0, 3]);
    }
}
