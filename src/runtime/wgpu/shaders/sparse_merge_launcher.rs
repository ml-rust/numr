//! WGSL kernel launchers for sparse element-wise merges (add, sub, mul, div)
//! over CSR and CSC operands.
//!
//! A merge runs in three passes:
//! - count: output nonzeros per row (CSR) or column (CSC)
//! - exclusive scan: output pointers from the counts
//! - compute: merge the sorted index lists and apply the operation
//!
//! Every index buffer holds `i32`. Value buffers hold `f32`, `i32` or `u32`.

use wgpu::{Buffer, Queue};

use super::pipeline::PipelineCache;
use super::sparse_dispatch::{SparseKernel, run_sparse_kernel};
use crate::dtype::DType;
use crate::error::{Error, Result};

const SPARSE_MERGE_COUNT: &str = include_str!("sparse_merge_count.wgsl");
const SPARSE_MERGE_F32: &str = include_str!("sparse_merge_f32.wgsl");
const SPARSE_MERGE_I32: &str = include_str!("sparse_merge_i32.wgsl");
const SPARSE_MERGE_U32: &str = include_str!("sparse_merge_u32.wgsl");

/// Compressed sparse layout of both merge operands and the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressedFormat {
    /// Row pointers + column indices.
    Csr,
    /// Column pointers + row indices.
    Csc,
}

/// Element-wise operation of a sparse merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMergeOp {
    /// `a + b` over the union of both patterns.
    Add,
    /// `a - b` over the union of both patterns.
    Sub,
    /// `a * b` over the intersection of both patterns.
    Mul,
    /// `a / b` over the intersection of both patterns.
    Div,
}

impl SparseMergeOp {
    /// Whether the output pattern is the intersection of both patterns.
    pub fn is_intersection(self) -> bool {
        matches!(self, Self::Mul | Self::Div)
    }
}

/// Operand buffers of one compressed sparse matrix.
pub struct CompressedBuffers<'a> {
    /// Row pointers (CSR) or column pointers (CSC), `i32`.
    pub ptrs: &'a Buffer,
    /// Column indices (CSR) or row indices (CSC), `i32`.
    pub indices: &'a Buffer,
    /// Stored values.
    pub values: &'a Buffer,
}

fn count_entry(format: CompressedFormat, op: SparseMergeOp) -> &'static str {
    match (format, op.is_intersection()) {
        (CompressedFormat::Csr, false) => "csr_merge_count",
        (CompressedFormat::Csr, true) => "csr_mul_count",
        (CompressedFormat::Csc, false) => "csc_merge_count",
        (CompressedFormat::Csc, true) => "csc_mul_count",
    }
}

fn compute_entry(
    format: CompressedFormat,
    op: SparseMergeOp,
    dtype: DType,
) -> Result<SparseKernel> {
    use CompressedFormat::{Csc, Csr};
    use SparseMergeOp::{Add, Div, Mul, Sub};

    let (module, source) = match dtype {
        DType::F32 => ("sparse_merge_f32", SPARSE_MERGE_F32),
        DType::I32 => ("sparse_merge_i32", SPARSE_MERGE_I32),
        DType::U32 => ("sparse_merge_u32", SPARSE_MERGE_U32),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse merge (WebGPU supports F32, I32, U32)",
            });
        }
    };
    let entry = match (format, op, dtype) {
        (Csr, Add, DType::F32) => "csr_add_compute_f32",
        (Csr, Sub, DType::F32) => "csr_sub_compute_f32",
        (Csr, Mul, DType::F32) => "csr_mul_compute_f32",
        (Csr, Div, DType::F32) => "csr_div_compute_f32",
        (Csc, Add, DType::F32) => "csc_add_compute_f32",
        (Csc, Sub, DType::F32) => "csc_sub_compute_f32",
        (Csc, Mul, DType::F32) => "csc_mul_compute_f32",
        (Csc, Div, DType::F32) => "csc_div_compute_f32",
        (Csr, Add, DType::I32) => "csr_add_compute_i32",
        (Csr, Sub, DType::I32) => "csr_sub_compute_i32",
        (Csr, Mul, DType::I32) => "csr_mul_compute_i32",
        (Csr, Div, DType::I32) => "csr_div_compute_i32",
        (Csc, Add, DType::I32) => "csc_add_compute_i32",
        (Csc, Sub, DType::I32) => "csc_sub_compute_i32",
        (Csc, Mul, DType::I32) => "csc_mul_compute_i32",
        (Csc, Div, DType::I32) => "csc_div_compute_i32",
        (Csr, Add, _) => "csr_add_compute_u32",
        (Csr, Sub, _) => "csr_sub_compute_u32",
        (Csr, Mul, _) => "csr_mul_compute_u32",
        (Csr, Div, _) => "csr_div_compute_u32",
        (Csc, Add, _) => "csc_add_compute_u32",
        (Csc, Sub, _) => "csc_sub_compute_u32",
        (Csc, Mul, _) => "csc_mul_compute_u32",
        (Csc, Div, _) => "csc_div_compute_u32",
    };
    // Bindings 0-6 (both operands and the output pointers) are read-only.
    Ok(SparseKernel {
        module,
        source,
        entry,
        num_readonly: 7,
    })
}

/// Launch the count pass: `counts[i]` = output nonzeros of row (CSR) or
/// column (CSC) `i`, over the union (add, sub) or intersection (mul, div) of
/// both patterns. `params` holds `{ n }`, the row or column count.
pub fn launch_sparse_merge_count(
    cache: &PipelineCache,
    queue: &Queue,
    format: CompressedFormat,
    op: SparseMergeOp,
    a_ptrs: &Buffer,
    a_indices: &Buffer,
    b_ptrs: &Buffer,
    b_indices: &Buffer,
    counts: &Buffer,
    params: &Buffer,
    n: usize,
) -> Result<()> {
    // Bindings 0-3 (both operands' pointers and indices) are read-only.
    let kernel = SparseKernel {
        module: "sparse_merge_count",
        source: SPARSE_MERGE_COUNT,
        entry: count_entry(format, op),
        num_readonly: 4,
    };
    run_sparse_kernel(
        cache,
        queue,
        &kernel,
        &[a_ptrs, a_indices, b_ptrs, b_indices, counts],
        params,
        n,
    )
}

/// Launch the compute pass: merge both operands into `out_indices` and
/// `out_values` at the offsets in `out_ptrs`. `params` holds `{ n }`, the
/// row or column count.
pub fn launch_sparse_merge_compute(
    cache: &PipelineCache,
    queue: &Queue,
    format: CompressedFormat,
    op: SparseMergeOp,
    a: &CompressedBuffers<'_>,
    b: &CompressedBuffers<'_>,
    out_ptrs: &Buffer,
    out_indices: &Buffer,
    out_values: &Buffer,
    params: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let kernel = compute_entry(format, op, dtype)?;
    run_sparse_kernel(
        cache,
        queue,
        &kernel,
        &[
            a.ptrs,
            a.indices,
            a.values,
            b.ptrs,
            b.indices,
            b.values,
            out_ptrs,
            out_indices,
            out_values,
        ],
        params,
        n,
    )
}

/// Launch the sequential exclusive scan of `n` `i32` counts.
///
/// Writes `output[i] = sum(input[0..i])` for `i` in `0..=n`, so `output[n]`
/// holds the total. `params` holds `{ n }`. One invocation does the work.
pub fn launch_exclusive_scan_i32(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
) -> Result<()> {
    let kernel = SparseKernel {
        module: "sparse_merge_count",
        source: SPARSE_MERGE_COUNT,
        entry: "exclusive_scan_i32",
        num_readonly: 1,
    };
    run_sparse_kernel(cache, queue, &kernel, &[input, output], params, 1)
}
