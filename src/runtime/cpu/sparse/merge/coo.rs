//! Generic and intersection merge implementations for COO element-wise operations

use super::super::super::CpuRuntime;
use super::OperationSemantics;
use super::zero_tolerance;
use crate::dtype::Element;
use crate::error::Result;
use crate::tensor::Tensor;

/// Generic merge implementation for COO element-wise operations
///
/// Merges two COO matrices element-wise with full control over value transformations.
///
/// # Arguments
///
/// * `op` - Operation to apply when both matrices have a value at the same position
/// * `only_a_op` - Transformation for values that only exist in A (identity for add, identity for sub)
/// * `only_b_op` - Transformation for values that only exist in B (identity for add, negate for sub)
///
/// # Algorithm
///
/// 1. Concatenate both matrices' triplets (applying transformations)
/// 2. Sort by (row, col)
/// 3. Merge duplicate positions by applying the operation
pub(crate) fn merge_coo_impl<T: Element, F, FA, FB>(
    a_row_indices: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    _semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let device = a_values.device();

    // Handle empty inputs early to avoid bytemuck alignment errors
    let a_nnz = a_values.numel();
    let b_nnz = b_values.numel();

    if a_nnz == 0 && b_nnz == 0 {
        // Both empty - return empty result (always empty regardless of semantics)
        let empty_rows = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_cols = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device)?;
        return Ok((empty_rows, empty_cols, empty_vals));
    } else if a_nnz == 0 {
        // A is empty, B is not - apply B transformation
        let b_vals: Vec<T> = b_values.to_vec();
        let transformed_vals: Vec<T> = b_vals.iter().map(|&v| only_b_op(v)).collect();
        let out_vals = Tensor::from_slice(&transformed_vals, &[transformed_vals.len()], device)?;
        return Ok((b_row_indices.clone(), b_col_indices.clone(), out_vals));
    } else if b_nnz == 0 {
        // B is empty, A is not - apply A transformation (usually identity)
        let a_vals: Vec<T> = a_values.to_vec();
        let transformed_vals: Vec<T> = a_vals.iter().map(|&v| only_a_op(v)).collect();
        let out_vals = Tensor::from_slice(&transformed_vals, &[transformed_vals.len()], device)?;
        return Ok((a_row_indices.clone(), a_col_indices.clone(), out_vals));
    }

    // Both non-empty - proceed with merge
    let a_rows: Vec<i64> = a_row_indices.to_vec();
    let a_cols: Vec<i64> = a_col_indices.to_vec();
    let a_vals: Vec<T> = a_values.to_vec();
    let b_rows: Vec<i64> = b_row_indices.to_vec();
    let b_cols: Vec<i64> = b_col_indices.to_vec();
    let b_vals: Vec<T> = b_values.to_vec();

    // Concatenate triplets with source tag (true=from A, false=from B)
    let mut triplets: Vec<(i64, i64, T, bool)> = Vec::new();
    for i in 0..a_rows.len() {
        triplets.push((a_rows[i], a_cols[i], a_vals[i], true));
    }
    for i in 0..b_rows.len() {
        triplets.push((b_rows[i], b_cols[i], b_vals[i], false));
    }

    // Sort by (row, col)
    triplets.sort_by_key(|&(r, c, _, _)| (r, c));

    // Merge duplicates
    let mut result_rows: Vec<i64> = Vec::new();
    let mut result_cols: Vec<i64> = Vec::new();
    let mut result_vals: Vec<T> = Vec::new();

    if triplets.is_empty() {
        // Empty result
        let empty_rows = Tensor::from_slice(&result_rows, &[0], device)?;
        let empty_cols = Tensor::from_slice(&result_cols, &[0], device)?;
        let empty_vals = Tensor::from_slice(&result_vals, &[0], device)?;
        return Ok((empty_rows, empty_cols, empty_vals));
    }

    let mut current_row = triplets[0].0;
    let mut current_col = triplets[0].1;
    let mut current_val = triplets[0].2;
    let mut current_from_a = triplets[0].3;
    let mut current_merged = false; // Track if this value resulted from merging

    for i in 1..triplets.len() {
        let (row, col, val, from_a) = triplets[i];

        if row == current_row && col == current_col {
            // Same position - apply operation (merge)
            current_val = op(current_val, val);
            current_merged = true; // This value is now a merged result
        } else {
            // Different position - save current (with transformation if not merged)
            let final_val = if current_merged {
                // Already merged, no transformation needed
                current_val
            } else {
                // Not merged, apply transformation based on source
                if current_from_a {
                    only_a_op(current_val)
                } else {
                    only_b_op(current_val)
                }
            };

            if final_val.to_f64().abs() > zero_tolerance::<T>() {
                result_rows.push(current_row);
                result_cols.push(current_col);
                result_vals.push(final_val);
            }

            // Start new accumulation
            current_row = row;
            current_col = col;
            current_val = val;
            current_from_a = from_a;
            current_merged = false;
        }
    }

    // Don't forget the last triplet
    let final_val = if current_merged {
        // Already merged, no transformation needed
        current_val
    } else {
        // Not merged, apply transformation based on source
        if current_from_a {
            only_a_op(current_val)
        } else {
            only_b_op(current_val)
        }
    };

    if final_val.to_f64().abs() > zero_tolerance::<T>() {
        result_rows.push(current_row);
        result_cols.push(current_col);
        result_vals.push(final_val);
    }

    // Create result tensors
    let out_rows = Tensor::from_slice(&result_rows, &[result_rows.len()], device)?;
    let out_cols = Tensor::from_slice(&result_cols, &[result_cols.len()], device)?;
    let out_vals = Tensor::from_slice(&result_vals, &[result_vals.len()], device)?;

    Ok((out_rows, out_cols, out_vals))
}

/// Intersection-based merge for COO element-wise operations
///
/// Only keeps positions where BOTH matrices have non-zero values.
/// This is the correct semantics for sparse element-wise multiplication and division.
///
/// Uses a merge-join algorithm on sorted triplets for O(nnz_a + nnz_b) complexity.
pub(crate) fn intersect_coo_impl<T: Element, F>(
    a_row_indices: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    op: F,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
{
    let device = a_values.device();

    // Handle empty inputs - multiplication by zero gives zero (empty)
    let a_nnz = a_values.numel();
    let b_nnz = b_values.numel();

    if a_nnz == 0 || b_nnz == 0 {
        // If either is empty, result is empty
        let empty_rows = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_cols = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device)?;
        return Ok((empty_rows, empty_cols, empty_vals));
    }

    // Read COO data
    let a_rows: Vec<i64> = a_row_indices.to_vec();
    let a_cols: Vec<i64> = a_col_indices.to_vec();
    let a_vals: Vec<T> = a_values.to_vec();
    let b_rows: Vec<i64> = b_row_indices.to_vec();
    let b_cols: Vec<i64> = b_col_indices.to_vec();
    let b_vals: Vec<T> = b_values.to_vec();

    // Sort both by (row, col) if not already sorted
    let mut a_triplets: Vec<(i64, i64, T)> = a_rows
        .iter()
        .zip(a_cols.iter())
        .zip(a_vals.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();
    a_triplets.sort_by_key(|&(r, c, _)| (r, c));

    let mut b_triplets: Vec<(i64, i64, T)> = b_rows
        .iter()
        .zip(b_cols.iter())
        .zip(b_vals.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();
    b_triplets.sort_by_key(|&(r, c, _)| (r, c));

    // Merge-join to find intersection
    let mut result_rows: Vec<i64> = Vec::new();
    let mut result_cols: Vec<i64> = Vec::new();
    let mut result_vals: Vec<T> = Vec::new();

    let mut i = 0;
    let mut j = 0;

    while i < a_triplets.len() && j < b_triplets.len() {
        let (a_row, a_col, a_val) = a_triplets[i];
        let (b_row, b_col, b_val) = b_triplets[j];

        match (a_row.cmp(&b_row), a_col.cmp(&b_col)) {
            (std::cmp::Ordering::Less, _)
            | (std::cmp::Ordering::Equal, std::cmp::Ordering::Less) => {
                // A's position is before B's - skip it (not in intersection)
                i += 1;
            }
            (std::cmp::Ordering::Greater, _)
            | (std::cmp::Ordering::Equal, std::cmp::Ordering::Greater) => {
                // B's position is before A's - skip it (not in intersection)
                j += 1;
            }
            (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
                // Same position - apply operation and keep result
                let result = op(a_val, b_val);
                if result.to_f64().abs() > zero_tolerance::<T>() {
                    result_rows.push(a_row);
                    result_cols.push(a_col);
                    result_vals.push(result);
                }
                i += 1;
                j += 1;
            }
        }
    }

    // Create result tensors
    let out_rows = Tensor::from_slice(&result_rows, &[result_rows.len()], device)?;
    let out_cols = Tensor::from_slice(&result_cols, &[result_cols.len()], device)?;
    let out_vals = Tensor::from_slice(&result_vals, &[result_vals.len()], device)?;

    Ok((out_rows, out_cols, out_vals))
}

#[cfg(test)]
mod tests {
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuRuntime};
    use crate::sparse::SparseOps;
    use crate::tensor::Tensor;

    #[test]
    fn test_coo_f32_removes_values_below_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // COO format test: Use multiplication to create predictable small values
        // A has triplets: (0, 0, 1e-4), (1, 1, 2.0)
        let a_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        // B has triplets: (0, 0, 1e-5), (1, 1, 2.0)
        let b_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // (1,1): 2.0 * 2.0 = 4.0 (kept)
        let (_row_indices, col_indices, values) = client
            .mul_coo::<f32>(
                &a_row_indices,
                &a_col_indices,
                &a_values,
                &b_row_indices,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        assert_eq!(
            values_data.len(),
            1,
            "COO zero elimination should remove near-zero values"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![1]);
    }
}
