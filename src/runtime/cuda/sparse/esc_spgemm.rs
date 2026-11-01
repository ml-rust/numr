//! SpGEMM `C = A * B` for CSR operands on CUDA.
//!
//! The same result as the CPU backend: every output row sorted by column,
//! and entries with `|value| <= zero_tolerance` dropped.
//!
//! 1. Bound each output row by its product count, and scan the bounds into
//!    scratch slices.
//! 2. Hash-accumulate each row into its slice, sort it and count the kept
//!    entries.
//! 3. Scan the kept counts into the output pointers and gather the entries.
//!
//! Each scan reads one `i32` total to the host to size the next allocation.

use super::super::{CudaClient, CudaRuntime};
use super::indices::{check_float_dtype, indices_to_i64};
use crate::algorithm::sparse::{validate_dtype_match, validate_spgemm_shapes};
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::common::sparse_utils::zero_tolerance;
use crate::runtime::cuda::kernels::{
    SpgemmCsr, exclusive_scan_i32_gpu, spgemm_accumulate, spgemm_gather, spgemm_row_bound,
};
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Called from `SparseAlgorithms::esc_spgemm_csr`.
pub(super) fn esc_spgemm_csr(
    client: &CudaClient,
    a_csr: &CsrData<CudaRuntime>,
    b_csr: &CsrData<CudaRuntime>,
) -> Result<CsrData<CudaRuntime>> {
    // Check shapes and dtypes. The dtype check reads metadata only, so no data moves.
    let ([m, n], _k) = validate_spgemm_shapes(a_csr.shape, b_csr.shape)?;
    let dtype = a_csr.values.dtype();
    validate_dtype_match(dtype, b_csr.values.dtype())?;
    check_float_dtype(dtype, "sparse_matmul (CUDA)")?;
    let threshold = crate::dispatch_dtype!(dtype, T => { zero_tolerance::<T>() }, "sparse_matmul");
    client.spgemm(a_csr, b_csr, [m, n], threshold)
}

impl CudaClient {
    fn spgemm(
        &self,
        a: &CsrData<CudaRuntime>,
        b: &CsrData<CudaRuntime>,
        [m, n]: [usize; 2],
        threshold: f64,
    ) -> Result<CsrData<CudaRuntime>> {
        let device = a.values.device();
        let dtype = a.values.dtype();
        let empty = || -> Result<CsrData<CudaRuntime>> {
            CsrData::new(
                Tensor::zeros(&[m + 1], DType::I64, device)?,
                Tensor::zeros(&[0], DType::I64, device)?,
                Tensor::zeros(&[0], dtype, device)?,
                [m, n],
            )
        };
        if m == 0 || a.values.numel() == 0 || b.values.numel() == 0 {
            return empty();
        }

        let a_ptrs = indices_to_i64(self, &a.row_ptrs.contiguous()?)?;
        let a_cols = indices_to_i64(self, &a.col_indices.contiguous()?)?;
        let a_vals = a.values.contiguous()?;
        let b_ptrs = indices_to_i64(self, &b.row_ptrs.contiguous()?)?;
        let b_cols = indices_to_i64(self, &b.col_indices.contiguous()?)?;
        let b_vals = b.values.contiguous()?;
        let a_dev = SpgemmCsr {
            row_ptrs: a_ptrs.ptr(),
            col_indices: a_cols.ptr(),
            values: a_vals.ptr(),
        };
        let b_dev = SpgemmCsr {
            row_ptrs: b_ptrs.ptr(),
            col_indices: b_cols.ptr(),
            values: b_vals.ptr(),
        };
        let (ctx, stream, index) = (&self.context, &self.stream, self.device.index);
        let scan = |counts: &Tensor<CudaRuntime>| {
            // SAFETY: `counts` is a live I32 device tensor of m entries.
            unsafe { exclusive_scan_i32_gpu(ctx, stream, index, device, counts) }
        };

        // 1. Product bound per row, scanned into scratch slices.
        let bound = Tensor::<CudaRuntime>::zeros(&[m], DType::I32, device)?;
        // SAFETY: both operands are live CSR device buffers; `bound` holds m I32.
        unsafe { spgemm_row_bound(ctx, stream, index, &a_dev, &b_dev, bound.ptr(), m)? };
        let (slot_ptrs, slots) = scan(&bound)?;
        if slots == 0 {
            return empty();
        }

        // 2. Accumulate, sort and filter each row in its slice.
        let acc_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let slot_cols = Tensor::<CudaRuntime>::zeros(&[slots], DType::I64, device)?;
        let slot_vals = Tensor::<CudaRuntime>::zeros(&[slots], acc_dtype, device)?;
        let kept = Tensor::<CudaRuntime>::zeros(&[m], DType::I32, device)?;
        // SAFETY: scratch holds `slots` entries, `kept` holds m I32.
        unsafe {
            spgemm_accumulate(
                ctx,
                stream,
                index,
                dtype,
                &a_dev,
                &b_dev,
                slot_ptrs.ptr(),
                slot_cols.ptr(),
                slot_vals.ptr(),
                kept.ptr(),
                m,
                threshold,
            )?;
        }

        // 3. Output pointers from the kept counts, then the gather.
        let (out_ptrs, total) = scan(&kept)?;
        let out_cols = Tensor::<CudaRuntime>::zeros(&[total], DType::I64, device)?;
        let out_vals = Tensor::<CudaRuntime>::zeros(&[total], dtype, device)?;
        if total > 0 {
            // SAFETY: the outputs hold `total` = out_ptrs[m] entries.
            unsafe {
                spgemm_gather(
                    ctx,
                    stream,
                    index,
                    dtype,
                    slot_ptrs.ptr(),
                    slot_cols.ptr(),
                    slot_vals.ptr(),
                    out_ptrs.ptr(),
                    out_cols.ptr(),
                    out_vals.ptr(),
                    m,
                )?;
            }
        }
        CsrData::new(indices_to_i64(self, &out_ptrs)?, out_cols, out_vals, [m, n])
    }
}
