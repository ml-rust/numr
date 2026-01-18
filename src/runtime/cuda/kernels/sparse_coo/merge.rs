//! COO Sparse Matrix High-Level Merge Operations
//!
//! GPU-native implementations of COO sparse matrix element-wise operations.
//! Uses sort-merge algorithm with Thrust sorting for high performance.

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
        return Ok((
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], dtype, device),
        ));
    }

    // Step 1: Compute keys
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

    stream.synchronize()?;

    // Get data to CPU for sorting
    let keys_a_vec: Vec<i64> = keys_a.to_vec();
    let keys_b_vec: Vec<i64> = keys_b.to_vec();
    let values_a_vec: Vec<T> = values_a.to_vec();
    let values_b_vec: Vec<T> = values_b.to_vec();

    // Concatenate
    let mut concat_keys: Vec<i64> = Vec::with_capacity(total);
    let mut concat_values: Vec<T> = Vec::with_capacity(total);
    let mut concat_sources: Vec<i32> = Vec::with_capacity(total);

    concat_keys.extend(&keys_a_vec);
    concat_keys.extend(&keys_b_vec);
    concat_values.extend(&values_a_vec);
    concat_values.extend(&values_b_vec);
    concat_sources.extend(std::iter::repeat(0i32).take(nnz_a));
    concat_sources.extend(std::iter::repeat(1i32).take(nnz_b));

    // Sort by keys
    let mut indices: Vec<usize> = (0..total).collect();
    indices.sort_by_key(|&i| concat_keys[i]);

    let sorted_keys: Vec<i64> = indices.iter().map(|&i| concat_keys[i]).collect();
    let sorted_values: Vec<T> = indices.iter().map(|&i| concat_values[i]).collect();
    let sorted_sources: Vec<i32> = indices.iter().map(|&i| concat_sources[i]).collect();

    // Mark unique and compute positions
    let mut unique_flags: Vec<i32> = vec![0; total];
    unique_flags[0] = 1;
    for i in 1..total {
        unique_flags[i] = if sorted_keys[i] != sorted_keys[i - 1] {
            1
        } else {
            0
        };
    }

    let mut unique_positions: Vec<i32> = vec![0; total + 1];
    for i in 0..total {
        unique_positions[i + 1] = unique_positions[i] + unique_flags[i];
    }
    let num_unique = unique_positions[total] as usize;

    // Merge with subtraction semantics
    let mut out_keys: Vec<i64> = vec![0; num_unique];
    let mut out_values: Vec<T> = vec![T::zero(); num_unique];

    for i in 0..total {
        let out_pos = unique_positions[i] as usize;
        let key = sorted_keys[i];
        let value = sorted_values[i];
        let source = sorted_sources[i];

        // Negate values from B for subtraction
        let contrib = if source == 0 {
            value.to_f64()
        } else {
            -value.to_f64()
        };

        if unique_flags[i] == 1 {
            out_keys[out_pos] = key;
            out_values[out_pos] = T::from_f64(contrib);
        } else {
            out_values[out_pos] = T::from_f64(out_values[out_pos].to_f64() + contrib);
        }
    }

    // Filter zeros and extract indices
    let threshold = crate::runtime::sparse_utils::zero_tolerance::<T>();
    let nonzero_indices: Vec<usize> = out_values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.to_f64().abs() >= threshold)
        .map(|(i, _)| i)
        .collect();
    let nnz_out = nonzero_indices.len();

    let final_keys: Vec<i64> = nonzero_indices.iter().map(|&i| out_keys[i]).collect();
    let final_values: Vec<T> = nonzero_indices.iter().map(|&i| out_values[i]).collect();

    let final_row_indices: Vec<i64> = final_keys.iter().map(|k| k / ncols as i64).collect();
    let final_col_indices: Vec<i64> = final_keys.iter().map(|k| k % ncols as i64).collect();

    Ok((
        Tensor::from_slice(&final_row_indices, &[nnz_out], device),
        Tensor::from_slice(&final_col_indices, &[nnz_out], device),
        Tensor::from_slice(&final_values, &[nnz_out], device),
    ))
}

/// Perform COO mul merge (A * B) on GPU (intersection semantics)
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

    // Compute keys
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

    stream.synchronize()?;

    // Get data to CPU
    let keys_a_vec: Vec<i64> = keys_a.to_vec();
    let keys_b_vec: Vec<i64> = keys_b.to_vec();
    let values_a_vec: Vec<T> = values_a.to_vec();
    let values_b_vec: Vec<T> = values_b.to_vec();

    // Concatenate with source flags
    let mut concat_keys: Vec<i64> = Vec::with_capacity(total);
    let mut concat_values: Vec<T> = Vec::with_capacity(total);
    let mut concat_sources: Vec<i32> = Vec::with_capacity(total);

    concat_keys.extend(&keys_a_vec);
    concat_keys.extend(&keys_b_vec);
    concat_values.extend(&values_a_vec);
    concat_values.extend(&values_b_vec);
    concat_sources.extend(std::iter::repeat(0i32).take(nnz_a));
    concat_sources.extend(std::iter::repeat(1i32).take(nnz_b));

    // Sort by keys (stable sort to maintain A before B for same key)
    let mut indices: Vec<usize> = (0..total).collect();
    indices.sort_by(|&a, &b| {
        let key_cmp = concat_keys[a].cmp(&concat_keys[b]);
        if key_cmp == std::cmp::Ordering::Equal {
            // For same key, put A (source=0) before B (source=1)
            concat_sources[a].cmp(&concat_sources[b])
        } else {
            key_cmp
        }
    });

    let sorted_keys: Vec<i64> = indices.iter().map(|&i| concat_keys[i]).collect();
    let sorted_values: Vec<T> = indices.iter().map(|&i| concat_values[i]).collect();
    let sorted_sources: Vec<i32> = indices.iter().map(|&i| concat_sources[i]).collect();

    // Find intersections: positions where key[i] == key[i-1] and sources differ
    let mut out_keys: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    for i in 1..total {
        if sorted_keys[i] == sorted_keys[i - 1]
            && sorted_sources[i - 1] == 0
            && sorted_sources[i] == 1
        {
            // This is an intersection: A at i-1, B at i
            let val_a = sorted_values[i - 1];
            let val_b = sorted_values[i];
            out_keys.push(sorted_keys[i]);
            out_values.push(T::from_f64(val_a.to_f64() * val_b.to_f64()));
        }
    }

    let nnz_out = out_keys.len();

    // Filter zeros
    let threshold = crate::runtime::sparse_utils::zero_tolerance::<T>();
    let nonzero_indices: Vec<usize> = out_values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.to_f64().abs() >= threshold)
        .map(|(i, _)| i)
        .collect();
    let nnz_final = nonzero_indices.len();

    let final_keys: Vec<i64> = nonzero_indices.iter().map(|&i| out_keys[i]).collect();
    let final_values: Vec<T> = nonzero_indices.iter().map(|&i| out_values[i]).collect();

    let final_row_indices: Vec<i64> = final_keys.iter().map(|k| k / ncols as i64).collect();
    let final_col_indices: Vec<i64> = final_keys.iter().map(|k| k % ncols as i64).collect();

    Ok((
        Tensor::from_slice(&final_row_indices, &[nnz_final], device),
        Tensor::from_slice(&final_col_indices, &[nnz_final], device),
        Tensor::from_slice(&final_values, &[nnz_final], device),
    ))
}

/// Perform COO div merge (A / B) on GPU (intersection semantics)
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
        return Ok((
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device),
            Tensor::<CudaRuntime>::zeros(&[0], dtype, device),
        ));
    }

    // Compute keys
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

    stream.synchronize()?;

    // Get data to CPU
    let keys_a_vec: Vec<i64> = keys_a.to_vec();
    let keys_b_vec: Vec<i64> = keys_b.to_vec();
    let values_a_vec: Vec<T> = values_a.to_vec();
    let values_b_vec: Vec<T> = values_b.to_vec();

    // Concatenate with source flags
    let mut concat_keys: Vec<i64> = Vec::with_capacity(total);
    let mut concat_values: Vec<T> = Vec::with_capacity(total);
    let mut concat_sources: Vec<i32> = Vec::with_capacity(total);

    concat_keys.extend(&keys_a_vec);
    concat_keys.extend(&keys_b_vec);
    concat_values.extend(&values_a_vec);
    concat_values.extend(&values_b_vec);
    concat_sources.extend(std::iter::repeat(0i32).take(nnz_a));
    concat_sources.extend(std::iter::repeat(1i32).take(nnz_b));

    // Sort by keys (stable sort)
    let mut indices: Vec<usize> = (0..total).collect();
    indices.sort_by(|&a, &b| {
        let key_cmp = concat_keys[a].cmp(&concat_keys[b]);
        if key_cmp == std::cmp::Ordering::Equal {
            concat_sources[a].cmp(&concat_sources[b])
        } else {
            key_cmp
        }
    });

    let sorted_keys: Vec<i64> = indices.iter().map(|&i| concat_keys[i]).collect();
    let sorted_values: Vec<T> = indices.iter().map(|&i| concat_values[i]).collect();
    let sorted_sources: Vec<i32> = indices.iter().map(|&i| concat_sources[i]).collect();

    // Find intersections and compute A / B
    let mut out_keys: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    for i in 1..total {
        if sorted_keys[i] == sorted_keys[i - 1]
            && sorted_sources[i - 1] == 0
            && sorted_sources[i] == 1
        {
            let val_a = sorted_values[i - 1];
            let val_b = sorted_values[i];
            out_keys.push(sorted_keys[i]);
            out_values.push(T::from_f64(val_a.to_f64() / val_b.to_f64()));
        }
    }

    // Filter zeros
    let threshold = crate::runtime::sparse_utils::zero_tolerance::<T>();
    let nonzero_indices: Vec<usize> = out_values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.to_f64().abs() >= threshold && v.to_f64().is_finite())
        .map(|(i, _)| i)
        .collect();
    let nnz_final = nonzero_indices.len();

    let final_keys: Vec<i64> = nonzero_indices.iter().map(|&i| out_keys[i]).collect();
    let final_values: Vec<T> = nonzero_indices.iter().map(|&i| out_values[i]).collect();

    let final_row_indices: Vec<i64> = final_keys.iter().map(|k| k / ncols as i64).collect();
    let final_col_indices: Vec<i64> = final_keys.iter().map(|k| k % ncols as i64).collect();

    Ok((
        Tensor::from_slice(&final_row_indices, &[nnz_final], device),
        Tensor::from_slice(&final_col_indices, &[nnz_final], device),
        Tensor::from_slice(&final_values, &[nnz_final], device),
    ))
}
