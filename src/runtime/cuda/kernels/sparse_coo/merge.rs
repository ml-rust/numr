//! COO Sparse Matrix High-Level Merge Operations
//!
//! GPU-native implementations of COO sparse matrix element-wise operations.
//! Uses sort-merge algorithm with Thrust sorting for high performance.

#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use super::kernels::*;
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

/// Perform COO add merge (A + B) on GPU
///
/// Uses the following algorithm:
/// 1. Compute composite keys for both matrices
/// 2. Concatenate keys, values, and source flags
/// 3. Sort by keys
/// 4. Mark unique positions
/// 5. Exclusive scan to get output positions
/// 6. Merge duplicates with addition
/// 7. Filter out zeros
/// 8. Extract row/col indices from keys
pub unsafe fn coo_add_merge<T: CudaTypeName + Element>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_indices_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    shape: [usize; 2],
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let [_nrows, ncols] = shape;
    let nnz_a = values_a.numel();
    let nnz_b = values_b.numel();
    let total = nnz_a + nnz_b;

    if total == 0 {
        // Both matrices are empty
        return Ok((
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], dtype, device),
        ));
    }

    // Step 1: Compute keys for both matrices
    let keys_a = Tensor::<CudaRuntime>::zeros(&[nnz_a], DType::I64, device);
    let keys_b = Tensor::<CudaRuntime>::zeros(&[nnz_b], DType::I64, device);

    if nnz_a > 0 {
        launch_coo_compute_keys(
            context,
            stream,
            device_index,
            row_indices_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            keys_a.storage().ptr(),
            ncols as i64,
            nnz_a,
        )?;
    }

    if nnz_b > 0 {
        launch_coo_compute_keys(
            context,
            stream,
            device_index,
            row_indices_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            keys_b.storage().ptr(),
            ncols as i64,
            nnz_b,
        )?;
    }

    // Step 2: Concatenate keys, values, and source flags
    let concat_keys = Tensor::<CudaRuntime>::zeros(&[total], DType::I64, device);
    let concat_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let concat_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_concat_keys(
        context,
        stream,
        device_index,
        keys_a.storage().ptr(),
        keys_b.storage().ptr(),
        concat_keys.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    launch_coo_concat_values_with_source::<T>(
        context,
        stream,
        device_index,
        values_a.storage().ptr(),
        values_b.storage().ptr(),
        concat_values.storage().ptr(),
        concat_sources.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    // Step 3: Initialize indices array [0, 1, 2, ..., total-1] on GPU
    let indices = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_init_indices(
        context,
        stream,
        device_index,
        indices.storage().ptr(),
        total,
    )?;

    // Step 4: Sort (keys, indices) using Thrust stable_sort_by_key - FULLY ON GPU
    // Thrust sorts IN-PLACE, so we sort concat_keys and indices directly
    unsafe {
        launch_thrust_sort_pairs_i64_i32(
            context,
            stream,
            device_index,
            concat_keys.storage().ptr(),
            indices.storage().ptr(),
            total as u32,
        )?;
    }

    // After sorting, concat_keys and indices are now sorted

    // Step 5: Gather values and sources using sorted indices - ALL ON GPU
    let sorted_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let sorted_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_gather::<T>(
        context,
        stream,
        device_index,
        concat_values.storage().ptr(),
        indices.storage().ptr(), // indices is now sorted
        sorted_values.storage().ptr(),
        total,
    )?;

    launch_coo_gather_i32(
        context,
        stream,
        device_index,
        concat_sources.storage().ptr(),
        indices.storage().ptr(), // indices is now sorted
        sorted_sources.storage().ptr(),
        total,
    )?;

    // Step 6: Mark unique positions - ALL ON GPU
    let unique_flags = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_mark_unique(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(), // concat_keys is now sorted
        unique_flags.storage().ptr(),
        total,
    )?;

    // Step 7: Exclusive scan to get output positions - ALL ON GPU (using CUB)
    let (output_positions, num_unique) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &unique_flags,
        )?
    };

    // Step 8: Merge duplicates (add operation) - ALL ON GPU
    let merged_keys = Tensor::<CudaRuntime>::zeros(&[num_unique], DType::I64, device);
    let merged_values = Tensor::<CudaRuntime>::zeros(&[num_unique], dtype, device);

    launch_coo_merge_duplicates_add::<T>(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(), // concat_keys is sorted
        sorted_values.storage().ptr(),
        sorted_sources.storage().ptr(),
        unique_flags.storage().ptr(),
        output_positions.storage().ptr(),
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        total,
    )?;

    // Step 9: Filter out zeros - ALL ON GPU (using CUB)
    let threshold = crate::runtime::sparse_utils::zero_tolerance::<T>();
    let nonzero_flags = Tensor::<CudaRuntime>::zeros(&[num_unique], DType::I32, device);

    launch_coo_mark_nonzero::<T>(
        context,
        stream,
        device_index,
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        threshold,
        num_unique,
    )?;

    // Exclusive scan to get compaction positions
    let (compact_positions, nnz_out) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &nonzero_flags,
        )?
    };

    // Compact arrays to remove zeros
    let final_keys = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_values = Tensor::<CudaRuntime>::zeros(&[nnz_out], dtype, device);

    launch_coo_compact::<T>(
        context,
        stream,
        device_index,
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        compact_positions.storage().ptr(),
        final_keys.storage().ptr(),
        final_values.storage().ptr(),
        num_unique,
    )?;

    // Step 10: Extract row/col indices from keys - ALL ON GPU
    let final_row_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_col_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);

    launch_coo_extract_indices(
        context,
        stream,
        device_index,
        final_keys.storage().ptr(),
        final_row_indices.storage().ptr(),
        final_col_indices.storage().ptr(),
        ncols as i64,
        nnz_out,
    )?;

    // All operations completed on GPU - no CPU transfers!
    Ok((final_row_indices, final_col_indices, final_values))
}

/// Perform COO sub merge (A - B) on GPU
///
/// Uses the following algorithm:
/// 1. Compute composite keys for both matrices
/// 2. Concatenate keys, values, and source flags
/// 3. Sort by keys
/// 4. Mark unique positions
/// 5. Exclusive scan to get output positions
/// 6. Merge duplicates with subtraction (union semantics)
/// 7. Filter out zeros
/// 8. Extract row/col indices from keys
pub unsafe fn coo_sub_merge<T: CudaTypeName + Element>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_indices_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    shape: [usize; 2],
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let [_nrows, ncols] = shape;
    let nnz_a = values_a.numel();
    let nnz_b = values_b.numel();
    let total = nnz_a + nnz_b;

    if total == 0 {
        // Both matrices are empty
        return Ok((
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], dtype, device),
        ));
    }

    // Step 1: Compute keys for both matrices
    let keys_a = Tensor::<CudaRuntime>::zeros(&[nnz_a], DType::I64, device);
    let keys_b = Tensor::<CudaRuntime>::zeros(&[nnz_b], DType::I64, device);

    if nnz_a > 0 {
        launch_coo_compute_keys(
            context,
            stream,
            device_index,
            row_indices_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            keys_a.storage().ptr(),
            ncols as i64,
            nnz_a,
        )?;
    }

    if nnz_b > 0 {
        launch_coo_compute_keys(
            context,
            stream,
            device_index,
            row_indices_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            keys_b.storage().ptr(),
            ncols as i64,
            nnz_b,
        )?;
    }

    // Step 2: Concatenate keys, values, and source flags - ALL ON GPU
    let concat_keys = Tensor::<CudaRuntime>::zeros(&[total], DType::I64, device);
    let concat_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let concat_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_concat_keys(
        context,
        stream,
        device_index,
        keys_a.storage().ptr(),
        keys_b.storage().ptr(),
        concat_keys.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    launch_coo_concat_values_with_source::<T>(
        context,
        stream,
        device_index,
        values_a.storage().ptr(),
        values_b.storage().ptr(),
        concat_values.storage().ptr(),
        concat_sources.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    // Step 3: Initialize indices array [0, 1, 2, ..., total-1] on GPU
    let indices = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_init_indices(
        context,
        stream,
        device_index,
        indices.storage().ptr(),
        total,
    )?;

    // Step 4: Sort (keys, indices) using Thrust stable_sort_by_key - FULLY ON GPU
    unsafe {
        launch_thrust_sort_pairs_i64_i32(
            context,
            stream,
            device_index,
            concat_keys.storage().ptr(),
            indices.storage().ptr(),
            total as u32,
        )?;
    }

    // Step 5: Gather values and sources using sorted indices - ALL ON GPU
    let sorted_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let sorted_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_gather::<T>(
        context,
        stream,
        device_index,
        concat_values.storage().ptr(),
        indices.storage().ptr(),
        sorted_values.storage().ptr(),
        total,
    )?;

    launch_coo_gather_i32(
        context,
        stream,
        device_index,
        concat_sources.storage().ptr(),
        indices.storage().ptr(),
        sorted_sources.storage().ptr(),
        total,
    )?;

    // Step 6: Mark unique positions - ALL ON GPU
    let unique_flags = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_mark_unique(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(),
        unique_flags.storage().ptr(),
        total,
    )?;

    // Step 7: Exclusive scan to get output positions - ALL ON GPU (using CUB)
    let (output_positions, num_unique) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &unique_flags,
        )?
    };

    // Step 8: Merge duplicates (subtract operation) - ALL ON GPU
    let merged_keys = Tensor::<CudaRuntime>::zeros(&[num_unique], DType::I64, device);
    let merged_values = Tensor::<CudaRuntime>::zeros(&[num_unique], dtype, device);

    launch_coo_merge_sub::<T>(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(),
        sorted_values.storage().ptr(),
        sorted_sources.storage().ptr(),
        output_positions.storage().ptr(),
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        total,
        num_unique,
    )?;

    // Step 9: Filter out zeros - ALL ON GPU (using CUB)
    let threshold = crate::runtime::sparse_utils::zero_tolerance::<T>();
    let nonzero_flags = Tensor::<CudaRuntime>::zeros(&[num_unique], DType::I32, device);

    launch_coo_mark_nonzero::<T>(
        context,
        stream,
        device_index,
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        threshold,
        num_unique,
    )?;

    // Exclusive scan to get compaction positions
    let (compact_positions, nnz_out) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &nonzero_flags,
        )?
    };

    // Compact arrays to remove zeros
    let final_keys = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_values = Tensor::<CudaRuntime>::zeros(&[nnz_out], dtype, device);

    launch_coo_compact::<T>(
        context,
        stream,
        device_index,
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        compact_positions.storage().ptr(),
        final_keys.storage().ptr(),
        final_values.storage().ptr(),
        num_unique,
    )?;

    // Step 10: Extract row/col indices from keys - ALL ON GPU
    let final_row_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_col_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);

    launch_coo_extract_indices(
        context,
        stream,
        device_index,
        final_keys.storage().ptr(),
        final_row_indices.storage().ptr(),
        final_col_indices.storage().ptr(),
        ncols as i64,
        nnz_out,
    )?;

    // All operations completed on GPU - no CPU transfers!
    Ok((final_row_indices, final_col_indices, final_values))
}

/// Perform COO mul merge (A * B) on GPU (intersection semantics)
///
/// Uses the following algorithm:
/// 1. Compute composite keys for both matrices
/// 2. Concatenate keys, values, and source flags
/// 3. Sort by keys
/// 4. Count intersection positions (where consecutive keys match with different sources)
/// 5. Exclusive scan to get output positions
/// 6. Merge intersections with multiplication
/// 7. Filter out zeros
/// 8. Extract row/col indices from keys
pub unsafe fn coo_mul_merge<T: CudaTypeName + Element>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_indices_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    shape: [usize; 2],
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let [_nrows, ncols] = shape;
    let nnz_a = values_a.numel();
    let nnz_b = values_b.numel();
    let total = nnz_a + nnz_b;

    if nnz_a == 0 || nnz_b == 0 {
        // One matrix is empty, intersection is empty
        return Ok((
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], dtype, device),
        ));
    }

    // Step 1: Compute keys for both matrices
    let keys_a = Tensor::<CudaRuntime>::zeros(&[nnz_a], DType::I64, device);
    let keys_b = Tensor::<CudaRuntime>::zeros(&[nnz_b], DType::I64, device);

    launch_coo_compute_keys(
        context,
        stream,
        device_index,
        row_indices_a.storage().ptr(),
        col_indices_a.storage().ptr(),
        keys_a.storage().ptr(),
        ncols as i64,
        nnz_a,
    )?;

    launch_coo_compute_keys(
        context,
        stream,
        device_index,
        row_indices_b.storage().ptr(),
        col_indices_b.storage().ptr(),
        keys_b.storage().ptr(),
        ncols as i64,
        nnz_b,
    )?;

    // Step 2: Concatenate keys, values, and source flags - ALL ON GPU
    let concat_keys = Tensor::<CudaRuntime>::zeros(&[total], DType::I64, device);
    let concat_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let concat_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_concat_keys(
        context,
        stream,
        device_index,
        keys_a.storage().ptr(),
        keys_b.storage().ptr(),
        concat_keys.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    launch_coo_concat_values_with_source::<T>(
        context,
        stream,
        device_index,
        values_a.storage().ptr(),
        values_b.storage().ptr(),
        concat_values.storage().ptr(),
        concat_sources.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    // Step 3: Initialize indices array [0, 1, 2, ..., total-1] on GPU
    let indices = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_init_indices(
        context,
        stream,
        device_index,
        indices.storage().ptr(),
        total,
    )?;

    // Step 4: Sort (keys, indices) using Thrust stable_sort_by_key - FULLY ON GPU
    unsafe {
        launch_thrust_sort_pairs_i64_i32(
            context,
            stream,
            device_index,
            concat_keys.storage().ptr(),
            indices.storage().ptr(),
            total as u32,
        )?;
    }

    // Step 5: Gather values and sources using sorted indices - ALL ON GPU
    let sorted_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let sorted_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_gather::<T>(
        context,
        stream,
        device_index,
        concat_values.storage().ptr(),
        indices.storage().ptr(),
        sorted_values.storage().ptr(),
        total,
    )?;

    launch_coo_gather_i32(
        context,
        stream,
        device_index,
        concat_sources.storage().ptr(),
        indices.storage().ptr(),
        sorted_sources.storage().ptr(),
        total,
    )?;

    // Step 6: Count intersections - ALL ON GPU
    let intersection_flags = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_count_intersections(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(),
        sorted_sources.storage().ptr(),
        intersection_flags.storage().ptr(),
        total,
    )?;

    // Step 7: Exclusive scan to get output positions - ALL ON GPU (using CUB)
    let (output_positions, num_intersections) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &intersection_flags,
        )?
    };

    // Step 8: Merge intersections (multiply operation) - ALL ON GPU
    let merged_keys = Tensor::<CudaRuntime>::zeros(&[num_intersections], DType::I64, device);
    let merged_values = Tensor::<CudaRuntime>::zeros(&[num_intersections], dtype, device);

    launch_coo_merge_mul::<T>(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(),
        sorted_values.storage().ptr(),
        sorted_sources.storage().ptr(),
        intersection_flags.storage().ptr(),
        output_positions.storage().ptr(),
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        total,
    )?;

    // Step 9: Filter out zeros - ALL ON GPU (using CUB)
    let threshold = crate::runtime::sparse_utils::zero_tolerance::<T>();
    let nonzero_flags = Tensor::<CudaRuntime>::zeros(&[num_intersections], DType::I32, device);

    launch_coo_mark_nonzero::<T>(
        context,
        stream,
        device_index,
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        threshold,
        num_intersections,
    )?;

    // Exclusive scan to get compaction positions
    let (compact_positions, nnz_out) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &nonzero_flags,
        )?
    };

    // Compact arrays to remove zeros
    let final_keys = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_values = Tensor::<CudaRuntime>::zeros(&[nnz_out], dtype, device);

    launch_coo_compact::<T>(
        context,
        stream,
        device_index,
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        compact_positions.storage().ptr(),
        final_keys.storage().ptr(),
        final_values.storage().ptr(),
        num_intersections,
    )?;

    // Step 10: Extract row/col indices from keys - ALL ON GPU
    let final_row_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_col_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);

    launch_coo_extract_indices(
        context,
        stream,
        device_index,
        final_keys.storage().ptr(),
        final_row_indices.storage().ptr(),
        final_col_indices.storage().ptr(),
        ncols as i64,
        nnz_out,
    )?;

    // All operations completed on GPU - no CPU transfers!
    Ok((final_row_indices, final_col_indices, final_values))
}

/// Perform COO div merge (A / B) on GPU (intersection semantics)
///
/// Uses the following algorithm:
/// 1. Compute composite keys for both matrices
/// 2. Concatenate keys, values, and source flags
/// 3. Sort by keys
/// 4. Count intersection positions (where consecutive keys match with different sources)
/// 5. Exclusive scan to get output positions
/// 6. Merge intersections with division
/// 7. Filter out zeros and non-finite values
/// 8. Extract row/col indices from keys
pub unsafe fn coo_div_merge<T: CudaTypeName + Element>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_indices_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    shape: [usize; 2],
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let [_nrows, ncols] = shape;
    let nnz_a = values_a.numel();
    let nnz_b = values_b.numel();
    let total = nnz_a + nnz_b;

    if nnz_a == 0 || nnz_b == 0 {
        // One matrix is empty, intersection is empty
        return Ok((
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], dtype, device),
        ));
    }

    // Step 1: Compute keys for both matrices
    let keys_a = Tensor::<CudaRuntime>::zeros(&[nnz_a], DType::I64, device);
    let keys_b = Tensor::<CudaRuntime>::zeros(&[nnz_b], DType::I64, device);

    launch_coo_compute_keys(
        context,
        stream,
        device_index,
        row_indices_a.storage().ptr(),
        col_indices_a.storage().ptr(),
        keys_a.storage().ptr(),
        ncols as i64,
        nnz_a,
    )?;

    launch_coo_compute_keys(
        context,
        stream,
        device_index,
        row_indices_b.storage().ptr(),
        col_indices_b.storage().ptr(),
        keys_b.storage().ptr(),
        ncols as i64,
        nnz_b,
    )?;

    // Step 2: Concatenate keys, values, and source flags - ALL ON GPU
    let concat_keys = Tensor::<CudaRuntime>::zeros(&[total], DType::I64, device);
    let concat_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let concat_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_concat_keys(
        context,
        stream,
        device_index,
        keys_a.storage().ptr(),
        keys_b.storage().ptr(),
        concat_keys.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    launch_coo_concat_values_with_source::<T>(
        context,
        stream,
        device_index,
        values_a.storage().ptr(),
        values_b.storage().ptr(),
        concat_values.storage().ptr(),
        concat_sources.storage().ptr(),
        nnz_a,
        nnz_b,
    )?;

    // Step 3: Initialize indices array [0, 1, 2, ..., total-1] on GPU
    let indices = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_init_indices(
        context,
        stream,
        device_index,
        indices.storage().ptr(),
        total,
    )?;

    // Step 4: Sort (keys, indices) using Thrust stable_sort_by_key - FULLY ON GPU
    unsafe {
        launch_thrust_sort_pairs_i64_i32(
            context,
            stream,
            device_index,
            concat_keys.storage().ptr(),
            indices.storage().ptr(),
            total as u32,
        )?;
    }

    // Step 5: Gather values and sources using sorted indices - ALL ON GPU
    let sorted_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device);
    let sorted_sources = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);

    launch_coo_gather::<T>(
        context,
        stream,
        device_index,
        concat_values.storage().ptr(),
        indices.storage().ptr(),
        sorted_values.storage().ptr(),
        total,
    )?;

    launch_coo_gather_i32(
        context,
        stream,
        device_index,
        concat_sources.storage().ptr(),
        indices.storage().ptr(),
        sorted_sources.storage().ptr(),
        total,
    )?;

    // Step 6: Count intersections - ALL ON GPU
    let intersection_flags = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device);
    launch_coo_count_intersections(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(),
        sorted_sources.storage().ptr(),
        intersection_flags.storage().ptr(),
        total,
    )?;

    // Step 7: Exclusive scan to get output positions - ALL ON GPU (using CUB)
    let (output_positions, num_intersections) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &intersection_flags,
        )?
    };

    // Step 8: Merge intersections (divide operation) - ALL ON GPU
    let merged_keys = Tensor::<CudaRuntime>::zeros(&[num_intersections], DType::I64, device);
    let merged_values = Tensor::<CudaRuntime>::zeros(&[num_intersections], dtype, device);

    launch_coo_merge_div::<T>(
        context,
        stream,
        device_index,
        concat_keys.storage().ptr(),
        sorted_values.storage().ptr(),
        sorted_sources.storage().ptr(),
        intersection_flags.storage().ptr(),
        output_positions.storage().ptr(),
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        total,
    )?;

    // Step 9: Filter out zeros and non-finite values - ALL ON GPU (using CUB)
    let threshold = crate::runtime::sparse_utils::zero_tolerance::<T>();
    let nonzero_flags = Tensor::<CudaRuntime>::zeros(&[num_intersections], DType::I32, device);

    launch_coo_mark_nonzero::<T>(
        context,
        stream,
        device_index,
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        threshold,
        num_intersections,
    )?;

    // Exclusive scan to get compaction positions
    let (compact_positions, nnz_out) = unsafe {
        crate::runtime::cuda::kernels::scan::exclusive_scan_i32_gpu(
            context,
            stream,
            device_index,
            device,
            &nonzero_flags,
        )?
    };

    // Compact arrays to remove zeros
    let final_keys = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_values = Tensor::<CudaRuntime>::zeros(&[nnz_out], dtype, device);

    launch_coo_compact::<T>(
        context,
        stream,
        device_index,
        merged_keys.storage().ptr(),
        merged_values.storage().ptr(),
        nonzero_flags.storage().ptr(),
        compact_positions.storage().ptr(),
        final_keys.storage().ptr(),
        final_values.storage().ptr(),
        num_intersections,
    )?;

    // Step 10: Extract row/col indices from keys - ALL ON GPU
    let final_row_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);
    let final_col_indices = Tensor::<CudaRuntime>::zeros(&[nnz_out], DType::I64, device);

    launch_coo_extract_indices(
        context,
        stream,
        device_index,
        final_keys.storage().ptr(),
        final_row_indices.storage().ptr(),
        final_col_indices.storage().ptr(),
        ncols as i64,
        nnz_out,
    )?;

    // All operations completed on GPU - no CPU transfers!
    Ok((final_row_indices, final_col_indices, final_values))
}
