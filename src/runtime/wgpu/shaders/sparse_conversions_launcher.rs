//! WGSL kernel launchers for sparse format conversions.
//!
//! Index buffers hold `i32`. Value buffers hold any 32-bit dtype (F32, I32,
//! U32): the kernels move raw words and never read a value as a number,
//! except the dense keep predicate, which takes the dtype as a mode.
//!
//! Each `params` buffer is a 16-byte uniform. [`launch_dense_row_count`] and
//! [`launch_dense_row_scatter`] take a 32-byte uniform instead.

use wgpu::{Buffer, Queue};

use super::pipeline::PipelineCache;
use super::sparse_dispatch::{SparseKernel, run_sparse_kernel};
use crate::error::Result;

const SPARSE_FORMAT_INDICES: &str = include_str!("sparse_format_indices.wgsl");
const SPARSE_DENSE_CONVERT: &str = include_str!("sparse_dense_convert.wgsl");

const fn format_kernel(entry: &'static str, num_readonly: u32) -> SparseKernel {
    SparseKernel {
        module: "sparse_format_indices",
        source: SPARSE_FORMAT_INDICES,
        entry,
        num_readonly,
    }
}

const fn dense_kernel(entry: &'static str, num_readonly: u32) -> SparseKernel {
    SparseKernel {
        module: "sparse_dense_convert",
        source: SPARSE_DENSE_CONVERT,
        entry,
        num_readonly,
    }
}

/// Expand `n + 1` compressed pointers into one bucket index per entry:
/// `out[i] = b` for every `i` in `ptrs[b]..ptrs[b + 1]`. `params` holds `{ n }`.
pub fn launch_expand_ptrs(
    cache: &PipelineCache,
    queue: &Queue,
    ptrs: &Buffer,
    out: &Buffer,
    params: &Buffer,
    n: usize,
) -> Result<()> {
    let kernel = format_kernel("expand_ptrs", 1);
    run_sparse_kernel(cache, queue, &kernel, &[ptrs, out], params, n)
}

/// Count entries per bucket: `counts[keys[i]] += 1`. `counts` must start at
/// zero. `params` holds `{ nnz }`.
pub fn launch_histogram(
    cache: &PipelineCache,
    queue: &Queue,
    keys: &Buffer,
    counts: &Buffer,
    params: &Buffer,
    nnz: usize,
) -> Result<()> {
    let kernel = format_kernel("histogram", 1);
    run_sparse_kernel(cache, queue, &kernel, &[keys, counts], params, nnz)
}

/// Group entry ids by bucket: entry `i` lands in `order[ptrs[keys[i]]..]`.
/// `fill` must start at zero. The order inside a bucket is arbitrary;
/// [`launch_rank_gather`] fixes it. `params` holds `{ nnz }`.
pub fn launch_scatter_order(
    cache: &PipelineCache,
    queue: &Queue,
    keys: &Buffer,
    ptrs: &Buffer,
    fill: &Buffer,
    order: &Buffer,
    params: &Buffer,
    nnz: usize,
) -> Result<()> {
    let kernel = format_kernel("scatter_order", 2);
    run_sparse_kernel(
        cache,
        queue,
        &kernel,
        &[keys, ptrs, fill, order],
        params,
        nnz,
    )
}

/// Sort every bucket by `(secondary, entry id)` and gather `secondary` and
/// `values` into the sorted slots. `params` holds `{ nnz }`.
pub fn launch_rank_gather(
    cache: &PipelineCache,
    queue: &Queue,
    primary: &Buffer,
    secondary: &Buffer,
    values: &Buffer,
    ptrs: &Buffer,
    order: &Buffer,
    out_secondary: &Buffer,
    out_values: &Buffer,
    params: &Buffer,
    nnz: usize,
) -> Result<()> {
    let kernel = format_kernel("rank_gather", 5);
    run_sparse_kernel(
        cache,
        queue,
        &kernel,
        &[
            primary,
            secondary,
            values,
            ptrs,
            order,
            out_secondary,
            out_values,
        ],
        params,
        nnz,
    )
}

/// Write CSR entries into a zeroed row-major dense buffer. A repeated
/// position keeps its last entry. `params` holds `{ nrows, ncols }`.
pub fn launch_csr_to_dense(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    dense: &Buffer,
    params: &Buffer,
    nrows: usize,
) -> Result<()> {
    let kernel = dense_kernel("csr_to_dense", 3);
    run_sparse_kernel(
        cache,
        queue,
        &kernel,
        &[row_ptrs, col_indices, values, dense],
        params,
        nrows,
    )
}

/// Count the kept entries of every dense row into `counts`.
pub fn launch_dense_row_count(
    cache: &PipelineCache,
    queue: &Queue,
    dense: &Buffer,
    counts: &Buffer,
    params: &Buffer,
    nrows: usize,
) -> Result<()> {
    let kernel = dense_kernel("dense_row_count", 1);
    run_sparse_kernel(cache, queue, &kernel, &[dense, counts], params, nrows)
}

/// Write the kept entries of every dense row, in column order, starting at
/// `ptrs[row]`.
pub fn launch_dense_row_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    dense: &Buffer,
    ptrs: &Buffer,
    rows: &Buffer,
    cols: &Buffer,
    values: &Buffer,
    params: &Buffer,
    nrows: usize,
) -> Result<()> {
    let kernel = dense_kernel("dense_row_scatter", 2);
    run_sparse_kernel(
        cache,
        queue,
        &kernel,
        &[dense, ptrs, rows, cols, values],
        params,
        nrows,
    )
}
