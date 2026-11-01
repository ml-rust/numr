//! Launchers for the CSR and CSC merge kernels in `sparse_merge.cu`.
//!
//! Every index buffer holds `int`. Value buffers hold F32, F64, F16 or BF16.

use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::sparse_strategy::{SparseFormat, SparseMergeOp};
use super::helpers::{launch_count_kernel, launch_csc_compute_kernel, launch_csr_compute_kernel};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Device pointers of one compressed sparse operand.
#[derive(Debug, Clone, Copy)]
pub struct CompressedPtrs {
    /// Row pointers (CSR) or column pointers (CSC), `int`.
    pub ptrs: u64,
    /// Column indices (CSR) or row indices (CSC), `int`.
    pub indices: u64,
    /// Stored values.
    pub values: u64,
}

fn count_kernel(format: SparseFormat, op: SparseMergeOp) -> Result<&'static str> {
    match (format, op.is_union()) {
        (SparseFormat::Csr, true) => Ok("csr_merge_count"),
        (SparseFormat::Csr, false) => Ok("csr_mul_count"),
        (SparseFormat::Csc, true) => Ok("csc_merge_count"),
        (SparseFormat::Csc, false) => Ok("csc_intersect_count"),
        (SparseFormat::Coo, _) => Err(Error::Internal(
            "sparse merge kernels take CSR or CSC operands, not COO".to_string(),
        )),
    }
}

/// Count the output entries of each of the `n` rows (CSR) or columns (CSC):
/// the union of both patterns for add and sub, the intersection for mul and
/// div. `counts` holds `n` `int`s.
///
/// # Safety
///
/// Every pointer must be a live device buffer on `context`: `n + 1` pointers
/// and the matching indices for both operands, and `n` counts.
pub unsafe fn launch_compressed_merge_count(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    format: SparseFormat,
    op: SparseMergeOp,
    a: &CompressedPtrs,
    b: &CompressedPtrs,
    counts: u64,
    n: usize,
) -> Result<()> {
    let kernel = count_kernel(format, op)?;
    unsafe {
        launch_count_kernel(
            context,
            stream,
            device_index,
            kernel,
            a.ptrs,
            a.indices,
            b.ptrs,
            b.indices,
            counts,
            n,
            kernel,
        )
    }
}

/// Merge both operands into `out_indices` and `out_values` at the offsets in
/// `out_ptrs` (the exclusive scan of the counts).
///
/// # Safety
///
/// Every pointer must be a live device buffer on `context`, and the output
/// buffers must hold `out_ptrs[n]` entries.
pub unsafe fn launch_compressed_merge_compute(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    format: SparseFormat,
    op: SparseMergeOp,
    dtype: DType,
    a: &CompressedPtrs,
    b: &CompressedPtrs,
    out_ptrs: u64,
    out_indices: u64,
    out_values: u64,
    n: usize,
) -> Result<()> {
    let base = match format {
        SparseFormat::Csr => format!("csr_{}_compute", op.kernel_prefix()),
        SparseFormat::Csc => format!("csc_{}_compute", op.kernel_prefix()),
        SparseFormat::Coo => {
            return Err(Error::Internal(
                "sparse merge kernels take CSR or CSC operands, not COO".to_string(),
            ));
        }
    };
    let launch = if format == SparseFormat::Csr {
        launch_csr_compute_kernel
    } else {
        launch_csc_compute_kernel
    };
    unsafe {
        launch(
            context,
            stream,
            device_index,
            &base,
            dtype,
            a.ptrs,
            a.indices,
            a.values,
            b.ptrs,
            b.indices,
            b.values,
            out_ptrs,
            out_indices,
            out_values,
            n,
            &base,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_kernels_match_the_cu_names() {
        assert_eq!(
            count_kernel(SparseFormat::Csr, SparseMergeOp::Sub).unwrap(),
            "csr_merge_count"
        );
        assert_eq!(
            count_kernel(SparseFormat::Csr, SparseMergeOp::Div).unwrap(),
            "csr_mul_count"
        );
        assert_eq!(
            count_kernel(SparseFormat::Csc, SparseMergeOp::Add).unwrap(),
            "csc_merge_count"
        );
        assert_eq!(
            count_kernel(SparseFormat::Csc, SparseMergeOp::Mul).unwrap(),
            "csc_intersect_count"
        );
        assert!(count_kernel(SparseFormat::Coo, SparseMergeOp::Add).is_err());
    }
}
