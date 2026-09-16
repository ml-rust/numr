//! Generic CSR merge operation for element-wise ops (add, sub, mul, div)

use super::super::super::CpuRuntime;
use super::common::{MergeStrategy, OperationSemantics, handle_empty_compressed};
use super::zero_tolerance;
use crate::dtype::Element;
use crate::error::Result;
use crate::tensor::Tensor;

/// Generic CSR merge operation for element-wise ops (add, sub, mul, div)
///
/// This function implements the sorted-merge algorithm for combining two
/// CSR matrices element-wise. The operation is specified by the `op` function
/// and the merge strategy determines which positions to keep.
///
/// # Arguments
///
/// * `strategy` - Whether to use Union (keep all positions) or Intersection (keep only common positions)
/// * `semantics` - Operation semantics for handling empty matrices
/// * `op` - Operation to apply when both matrices have values at a position
/// * `only_a_op` - Transformation for values that only exist in A
/// * `only_b_op` - Transformation for values that only exist in B
///
/// # Algorithm
///
/// For each row:
/// 1. Merge the two sorted lists of column indices
/// 2. Union strategy: Keep positions from either matrix
/// 3. Intersection strategy: Keep only positions where both have values
/// 4. Apply operation when both matrices have values
#[allow(clippy::too_many_arguments)]
pub(crate) fn merge_csr_impl<T: Element, F, FA, FB>(
    a_row_ptrs: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_ptrs: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
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
    let [nrows, _ncols] = shape;
    let device = a_values.device();

    // Handle empty inputs with centralized logic
    if let Some(result) = handle_empty_compressed::<T>(
        a_values.numel(),
        b_values.numel(),
        shape,
        device,
        a_row_ptrs,
        a_col_indices,
        a_values,
        b_row_ptrs,
        b_col_indices,
        b_values,
        semantics,
        true, // CSR format
    ) {
        return result;
    }

    // Read CSR data
    let a_row_ptrs_data: Vec<i64> = a_row_ptrs.to_vec();
    let a_col_indices_data: Vec<i64> = a_col_indices.to_vec();
    let a_values_data: Vec<T> = a_values.to_vec();
    let b_row_ptrs_data: Vec<i64> = b_row_ptrs.to_vec();
    let b_col_indices_data: Vec<i64> = b_col_indices.to_vec();
    let b_values_data: Vec<T> = b_values.to_vec();

    // Build result CSR
    let mut out_row_ptrs: Vec<i64> = Vec::with_capacity(nrows + 1);
    let mut out_col_indices: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    out_row_ptrs.push(0);

    for row in 0..nrows {
        let a_start = a_row_ptrs_data[row] as usize;
        let a_end = a_row_ptrs_data[row + 1] as usize;
        let b_start = b_row_ptrs_data[row] as usize;
        let b_end = b_row_ptrs_data[row + 1] as usize;

        let mut i = a_start;
        let mut j = b_start;

        // Merge strategy determines the loop condition and handling
        match strategy {
            MergeStrategy::Union => {
                // Union: Keep positions from either matrix (|| semantics)
                while i < a_end || j < b_end {
                    let a_col = if i < a_end {
                        a_col_indices_data[i]
                    } else {
                        i64::MAX
                    };
                    let b_col = if j < b_end {
                        b_col_indices_data[j]
                    } else {
                        i64::MAX
                    };

                    if a_col < b_col {
                        // Only A has value at this column - apply only_a_op
                        let result = only_a_op(a_values_data[i]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                    } else if a_col > b_col {
                        // Only B has value at this column - apply only_b_op
                        let result = only_b_op(b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(b_col);
                            out_values.push(result);
                        }
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
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
                    let a_col = a_col_indices_data[i];
                    let b_col = b_col_indices_data[j];

                    if a_col < b_col {
                        // Only A has value - skip in intersection
                        i += 1;
                    } else if a_col > b_col {
                        // Only B has value - skip in intersection
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }

        out_row_ptrs.push(out_col_indices.len() as i64);
    }

    // Create result tensors
    let result_row_ptrs = Tensor::from_slice(&out_row_ptrs, &[nrows + 1], device)?;
    let result_col_indices =
        Tensor::from_slice(&out_col_indices, &[out_col_indices.len()], device)?;
    let result_values = Tensor::from_slice(&out_values, &[out_values.len()], device)?;

    Ok((result_row_ptrs, result_col_indices, result_values))
}

#[cfg(test)]
mod tests {
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuRuntime};
    use crate::sparse::SparseOps;
    use crate::tensor::Tensor;

    #[test]
    fn test_add_csr_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // A:
        // [1, 0, 2]
        // [0, 3, 0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 2, 3], &[3], &device).unwrap();
        let a_col_indices = Tensor::from_slice(&[0i64, 2, 1], &[3], &device).unwrap();
        let a_values = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        // B:
        // [0, 4, 0]
        // [5, 0, 6]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 3], &[3], &device).unwrap();
        let b_col_indices = Tensor::from_slice(&[1i64, 0, 2], &[3], &device).unwrap();
        let b_values = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device).unwrap();

        // C = A + B:
        // [1, 4, 2]
        // [5, 3, 6]
        let (row_ptrs, col_indices, values) = client
            .add_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 3],
            )
            .unwrap();

        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let values_data: Vec<f32> = values.to_vec();

        assert_eq!(row_ptrs_data, vec![0, 3, 6]);
        assert_eq!(col_indices_data, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(values_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // =========================================================================
    // Zero-Elimination Tests
    // =========================================================================
    //
    // These tests explicitly verify that values below the dtype-specific
    // tolerance threshold are eliminated from sparse results.

    #[test]
    fn test_csr_f32_removes_values_below_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Test via multiplication which reliably produces small values
        // A = [1e-4, 0]
        //     [0, 2.0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        // B = [1e-5, 0]
        //     [0, 2.0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // (1,1): 2.0 * 2.0 = 4.0 (kept)
        let (_, col_indices, values) = client
            .mul_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        // The tiny value (1e-9) should be eliminated, only 4.0 remains
        assert_eq!(
            values_data.len(),
            1,
            "Near-zero values should be eliminated"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![1]);
    }

    #[test]
    fn test_csr_f32_preserves_values_above_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Create matrices that produce values ABOVE the tolerance
        // A = [1.001, 0]
        //     [0, 5.0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::from_slice(&[1.001f32, 5.0], &[2], &device).unwrap();

        // B = [1.0, 0]
        //     [0, 5.0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::from_slice(&[1.0f32, 5.0], &[2], &device).unwrap();

        // C = A - B should keep the first element (1.001 - 1.0 = 0.001)
        // since 0.001 > 1e-7 (F32 tolerance)
        let (_row_ptrs, col_indices, values) = client
            .sub_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        // The value 0.001 should be preserved (above tolerance)
        assert_eq!(
            values_data.len(),
            1,
            "Values above tolerance should be kept"
        );
        assert!((values_data[0] - 0.001).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![0]);
    }

    #[test]
    fn test_csr_f64_higher_precision_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // F64 has much tighter tolerance (1e-15)
        // Create values that would be eliminated in F32 but kept in F64
        // A = [1.0 + 1e-8, 0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_col_indices = Tensor::from_slice(&[0i64], &[1], &device).unwrap();
        let a_values = Tensor::from_slice(&[1.0 + 1e-8], &[1], &device).unwrap();

        // B = [1.0, 0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_col_indices = Tensor::from_slice(&[0i64], &[1], &device).unwrap();
        let b_values = Tensor::from_slice(&[1.0], &[1], &device).unwrap();

        // C = A - B = 1e-8, which is well above F64 tolerance (1e-15)
        let (_, _col_indices, values) = client
            .sub_csr::<f64>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [1, 2],
            )
            .unwrap();

        let values_data: Vec<f64> = values.to_vec();

        // F64 should preserve values down to 1e-15
        assert_eq!(values_data.len(), 1, "F64 preserves higher precision");
        assert!((values_data[0] - 1e-8).abs() < 1e-16);
    }

    #[test]
    fn test_mul_intersection_applies_zero_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Multiplication uses intersection semantics
        // Create small values through multiplication, testing both sides of tolerance
        // A = [1e-3, 0]
        //     [0, 2.0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::from_slice(&[1e-3f32, 2.0], &[2], &device).unwrap();

        // B = [1e-3, 0]
        //     [0, 3.0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::from_slice(&[1e-3f32, 3.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-3 * 1e-3 = 1e-6 (> 1e-7, above tolerance, kept)
        // (1,1): 2.0 * 3.0 = 6.0 (kept)
        let (_, col_indices, values) = client
            .mul_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        assert_eq!(
            values_data.len(),
            2,
            "Both values above F32 tolerance (1e-7)"
        );
        assert!(
            (values_data[0] - 1e-6).abs() < 1e-8,
            "Expected 1e-6, got {}",
            values_data[0]
        );
        assert!((values_data[1] - 6.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![0, 1]);
    }

    #[test]
    fn test_f32_boundary_conditions_at_tolerance_threshold() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Test values right at the boundary of F32 tolerance (1e-7)
        // Use 2x2 matrices to avoid edge cases with single-element matrices

        // Test 1: Below tolerance (1e-5 * 1e-4 = 1e-9 < 1e-7, eliminated)
        // A = [1e-5, 0]    B = [1e-4, 0]
        //     [0, 2.0]         [0, 2.0]
        let a1_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a1_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a1_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        let b1_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b1_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b1_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        let (_, _col_indices1, values1) = client
            .mul_csr::<f32>(
                &a1_row_ptrs,
                &a1_col_indices,
                &a1_values,
                &b1_row_ptrs,
                &b1_col_indices,
                &b1_values,
                [2, 2],
            )
            .unwrap();
        let vals1 = values1.to_vec::<f32>();
        assert_eq!(
            vals1.len(),
            1,
            "1e-9 < 1e-7 should be eliminated, only 4.0 kept"
        );
        assert!((vals1[0] - 4.0).abs() < 1e-6);

        // Test 2: Above tolerance (1e-3 * 1e-3 = 1e-6 > 1e-7, kept)
        // A = [1e-3, 0]    B = [1e-3, 0]
        //     [0, 3.0]         [0, 3.0]
        let a2_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a2_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a2_values = Tensor::from_slice(&[1e-3f32, 3.0], &[2], &device).unwrap();

        let b2_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b2_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b2_values = Tensor::from_slice(&[1e-3f32, 3.0], &[2], &device).unwrap();

        let (_, _col_indices2, values2) = client
            .mul_csr::<f32>(
                &a2_row_ptrs,
                &a2_col_indices,
                &a2_values,
                &b2_row_ptrs,
                &b2_col_indices,
                &b2_values,
                [2, 2],
            )
            .unwrap();
        let vals2 = values2.to_vec::<f32>();
        assert_eq!(vals2.len(), 2, "1e-6 > 1e-7 should be kept along with 9.0");
        assert!(
            (vals2[0] - 1e-6).abs() < 1e-8,
            "Expected 1e-6, got {}",
            vals2[0]
        );
        assert!((vals2[1] - 9.0).abs() < 1e-6);
    }
}
