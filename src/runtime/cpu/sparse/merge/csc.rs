//! Generic merge implementation for CSC element-wise operations

use super::super::super::CpuRuntime;
use super::common::{MergeStrategy, OperationSemantics, handle_empty_compressed};
use super::zero_tolerance;
use crate::dtype::Element;
use crate::error::Result;
use crate::tensor::Tensor;

/// Generic merge implementation for CSC element-wise operations
///
/// Merges two CSC matrices element-wise. The operation is specified by the `op` function.
///
/// # Algorithm
///
/// For each column:
/// 1. Merge the two sorted lists of row indices
/// 2. Apply operation when both matrices have a value at that row
/// 3. Keep values from only one matrix when the other is zero
#[allow(clippy::too_many_arguments)]
pub(crate) fn merge_csc_impl<T: Element, F, FA, FB>(
    a_col_ptrs: &Tensor<CpuRuntime>,
    a_row_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_col_ptrs: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
    strategy: MergeStrategy,
    semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let [_nrows, ncols] = shape;
    let device = a_values.device();

    // Handle empty inputs with centralized logic
    if let Some(result) = handle_empty_compressed::<T>(
        a_values.numel(),
        b_values.numel(),
        shape,
        device,
        a_col_ptrs,
        a_row_indices,
        a_values,
        b_col_ptrs,
        b_row_indices,
        b_values,
        semantics,
        false, // CSC format
    ) {
        return result;
    }

    // Read CSC data
    let a_col_ptrs_data: Vec<i64> = a_col_ptrs.to_vec();
    let a_row_indices_data: Vec<i64> = a_row_indices.to_vec();
    let a_values_data: Vec<T> = a_values.to_vec();
    let b_col_ptrs_data: Vec<i64> = b_col_ptrs.to_vec();
    let b_row_indices_data: Vec<i64> = b_row_indices.to_vec();
    let b_values_data: Vec<T> = b_values.to_vec();

    // Build result CSC
    let mut out_col_ptrs: Vec<i64> = Vec::with_capacity(ncols + 1);
    let mut out_row_indices: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    out_col_ptrs.push(0);

    for col in 0..ncols {
        let a_start = a_col_ptrs_data[col] as usize;
        let a_end = a_col_ptrs_data[col + 1] as usize;
        let b_start = b_col_ptrs_data[col] as usize;
        let b_end = b_col_ptrs_data[col + 1] as usize;

        let mut i = a_start;
        let mut j = b_start;

        // Merge strategy determines the loop condition and handling
        match strategy {
            MergeStrategy::Union => {
                // Union: Keep positions from either matrix (|| semantics)
                while i < a_end || j < b_end {
                    let a_row = if i < a_end {
                        a_row_indices_data[i]
                    } else {
                        i64::MAX
                    };
                    let b_row = if j < b_end {
                        b_row_indices_data[j]
                    } else {
                        i64::MAX
                    };

                    if a_row < b_row {
                        // Only A has value at this row - apply only_a_op
                        let result = only_a_op(a_values_data[i]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                    } else if a_row > b_row {
                        // Only B has value at this row - apply only_b_op
                        let result = only_b_op(b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(b_row);
                            out_values.push(result);
                        }
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
            MergeStrategy::Intersection => {
                // Intersection: Keep only positions where both have values (&& semantics)
                while i < a_end && j < b_end {
                    let a_row = a_row_indices_data[i];
                    let b_row = b_row_indices_data[j];

                    if a_row < b_row {
                        // Only A has value - skip in intersection
                        i += 1;
                    } else if a_row > b_row {
                        // Only B has value - skip in intersection
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }

        out_col_ptrs.push(out_row_indices.len() as i64);
    }

    // Create result tensors
    let result_col_ptrs = Tensor::from_slice(&out_col_ptrs, &[ncols + 1], device)?;
    let result_row_indices =
        Tensor::from_slice(&out_row_indices, &[out_row_indices.len()], device)?;
    let result_values = Tensor::from_slice(&out_values, &[out_values.len()], device)?;

    Ok((result_col_ptrs, result_row_indices, result_values))
}

#[cfg(test)]
mod tests {
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuRuntime};
    use crate::sparse::SparseOps;
    use crate::tensor::Tensor;

    #[test]
    fn test_csc_f32_removes_values_below_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // CSC format test: Use multiplication to create predictable small values
        // A (column-major):
        // Col 0: [1e-4]
        // Col 1: [2.0]
        let a_col_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        // B (column-major):
        // Col 0: [1e-5]
        // Col 1: [2.0]
        let b_col_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // Col 0: 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // Col 1: 2.0 * 2.0 = 4.0 (kept)
        let (_, row_indices, values) = client
            .mul_csc::<f32>(
                &a_col_ptrs,
                &a_row_indices,
                &a_values,
                &b_col_ptrs,
                &b_row_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let row_indices_data: Vec<i64> = row_indices.to_vec();

        assert_eq!(
            values_data.len(),
            1,
            "CSC zero elimination should work like CSR"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(row_indices_data, vec![1]);
    }
}
