//! CSR (Compressed Sparse Row) merge kernel launchers
//!
//! Low-level count and compute launchers plus high-level public merge operations
//! for CSR format sparse matrices.

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

use super::helpers::{launch_count_kernel, launch_csr_compute_kernel};

// ============================================================================
// Count Kernels
// ============================================================================

/// Launch CSR merge count kernel (for add/sub operations)
///
/// Counts output size per row using union semantics
///
/// # Safety
///
/// - `row_ptrs_a`, `col_indices_a`, `row_ptrs_b`, `col_indices_b`, and `row_counts` must be
///   valid device memory pointers on the device associated with `context`.
/// - `nrows` must match the number of rows in both input CSR matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csr_merge_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    row_counts: u64,
    nrows: usize,
) -> Result<()> {
    launch_count_kernel(
        context,
        stream,
        device_index,
        "csr_merge_count",
        row_ptrs_a,
        col_indices_a,
        row_ptrs_b,
        col_indices_b,
        row_counts,
        nrows,
        "CUDA sparse merge count",
    )
}

/// Launch CSR mul count kernel (intersection semantics)
///
/// # Safety
///
/// - `row_ptrs_a`, `col_indices_a`, `row_ptrs_b`, `col_indices_b`, and `row_counts` must be
///   valid device memory pointers on the device associated with `context`.
/// - `nrows` must match the number of rows in both input CSR matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csr_mul_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    row_counts: u64,
    nrows: usize,
) -> Result<()> {
    launch_count_kernel(
        context,
        stream,
        device_index,
        "csr_mul_count",
        row_ptrs_a,
        col_indices_a,
        row_ptrs_b,
        col_indices_b,
        row_counts,
        nrows,
        "CUDA sparse mul count",
    )
}

// ============================================================================
// Compute Kernels
// ============================================================================

/// Launch CSR add compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `nrows` must match the number of rows in both input CSR matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csr_add_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_add_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse add compute",
    )
}

/// Launch CSR sub compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `nrows` must match the number of rows in both input CSR matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csr_sub_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_sub_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse sub compute",
    )
}

/// Launch CSR mul compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `nrows` must match the number of rows in both input CSR matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csr_mul_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_mul_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse mul compute",
    )
}

/// Launch CSR div compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `nrows` must match the number of rows in both input CSR matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csr_div_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_div_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse div compute",
    )
}

// ============================================================================
// High-level CSR Merge Operations
// ============================================================================

/// Two-pass CSR addition: C = A + B (union semantics)
///
/// Now uses generic_csr_merge with AddMerge strategy to eliminate duplication.
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSR format. `nrows` must match the sparse matrix dimensions.
pub unsafe fn csr_add_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::AddMerge;
    super::generic::generic_csr_merge::<T, AddMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}

/// Two-pass CSR subtraction: C = A - B (union semantics)
///
/// Now uses generic_csr_merge with SubMerge strategy to eliminate duplication.
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSR format. `nrows` must match the sparse matrix dimensions.
pub unsafe fn csr_sub_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::SubMerge;
    super::generic::generic_csr_merge::<T, SubMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}

/// Two-pass CSR element-wise multiplication: C = A .* B (intersection semantics)
///
/// Now uses generic_csr_merge with MulMerge strategy to eliminate duplication.
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSR format. `nrows` must match the sparse matrix dimensions.
pub unsafe fn csr_mul_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::MulMerge;
    super::generic::generic_csr_merge::<T, MulMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}

/// Two-pass CSR element-wise division: C = A ./ B (intersection semantics)
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSR format. `nrows` must match the sparse matrix dimensions.
pub unsafe fn csr_div_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::DivMerge;
    super::generic::generic_csr_merge::<T, DivMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}
