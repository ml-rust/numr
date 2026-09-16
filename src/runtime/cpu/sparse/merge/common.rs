//! Merge strategy/semantics enums and the empty-matrix short-circuit shared by
//! the CSR and CSC merges.

use super::super::super::CpuRuntime;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Strategy for merging two sparse matrices
///
/// Determines which positions to keep in the result:
/// - Union: Keep positions that exist in EITHER matrix (OR semantics)
/// - Intersection: Keep positions that exist in BOTH matrices (AND semantics)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum MergeStrategy {
    /// Union semantics: Keep all positions from both matrices
    ///
    /// Example: Addition, Subtraction
    /// - Position in A only: keep A's value
    /// - Position in B only: keep B's value (or -B for subtraction)
    /// - Position in both: apply operation
    Union,

    /// Intersection semantics: Keep only positions where both matrices have values
    ///
    /// Example: Multiplication, Division
    /// - Position in A only: skip (0 * B = 0)
    /// - Position in B only: skip (A * 0 = 0)
    /// - Position in both: apply operation
    Intersection,
}

/// Operation semantics for element-wise operations
///
/// Defines how to handle empty matrices and special cases
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum OperationSemantics {
    /// Addition: 0 + x = x, x + 0 = x
    Add,
    /// Subtraction: x - 0 = x, 0 - x = -x
    Subtract,
    /// Multiplication: 0 * x = 0, x * 0 = 0
    Multiply,
    /// Division: 0 / x = 0, x / 0 = error
    Divide,
}

/// Generic centralized empty matrix handling for CSR/CSC operations
///
/// Returns Some(result) if the operation can be short-circuited due to empty inputs,
/// None if both inputs are non-empty and normal processing should continue.
///
/// # Type Parameters
///
/// * `format_is_csr` - If true, uses row-based pointers (CSR), else column-based (CSC)
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_empty_compressed<T: Element>(
    a_nnz: usize,
    b_nnz: usize,
    shape: [usize; 2],
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    a_ptrs: &Tensor<CpuRuntime>,
    a_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_ptrs: &Tensor<CpuRuntime>,
    b_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    semantics: OperationSemantics,
    format_is_csr: bool,
) -> Option<Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>> {
    // CSR uses nrows, CSC uses ncols for pointer dimension
    let ptr_dim = if format_is_csr { shape[0] } else { shape[1] };

    let empty_result =
        || -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
            let empty_ptrs = Tensor::from_slice(&vec![0i64; ptr_dim + 1], &[ptr_dim + 1], device)?;
            let empty_indices = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
            let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device)?;
            Ok((empty_ptrs, empty_indices, empty_vals))
        };

    match (a_nnz, b_nnz, semantics) {
        // Both empty - always return empty
        (0, 0, _) => Some(empty_result()),

        // A empty, B not empty
        (0, _, OperationSemantics::Add) => {
            // 0 + B = B
            Some(Ok((b_ptrs.clone(), b_indices.clone(), b_values.clone())))
        }
        (0, _, OperationSemantics::Subtract) => {
            // 0 - B = -B
            let b_vals: Vec<T> = b_values.to_vec();
            let negated_vals: Vec<T> = b_vals.iter().map(|&v| T::from_f64(-v.to_f64())).collect();
            Some(
                Tensor::from_slice(&negated_vals, &[negated_vals.len()], device)
                    .map(|out_vals| (b_ptrs.clone(), b_indices.clone(), out_vals)),
            )
        }
        (0, _, OperationSemantics::Multiply) => {
            // 0 * B = 0
            Some(empty_result())
        }
        (0, _, OperationSemantics::Divide) => {
            // 0 / B = 0 (mathematically)
            Some(empty_result())
        }

        // A not empty, B empty
        (_, 0, OperationSemantics::Add) => {
            // A + 0 = A
            Some(Ok((a_ptrs.clone(), a_indices.clone(), a_values.clone())))
        }
        (_, 0, OperationSemantics::Subtract) => {
            // A - 0 = A
            Some(Ok((a_ptrs.clone(), a_indices.clone(), a_values.clone())))
        }
        (_, 0, OperationSemantics::Multiply) => {
            // A * 0 = 0
            Some(empty_result())
        }
        (_, 0, OperationSemantics::Divide) => {
            // A / 0 = undefined (division by zero matrix)
            Some(Err(Error::Internal(
                "Division by zero - B matrix is empty".to_string(),
            )))
        }

        // Both non-empty - continue with normal processing
        (_, _, _) => None,
    }
}
