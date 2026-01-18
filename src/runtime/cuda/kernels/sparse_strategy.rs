//! Merge strategy abstraction for sparse matrix element-wise operations
//!
//! This module defines traits and types to eliminate code duplication across
//! CSR/CSC/COO merge operations. All operations follow the same two-pass algorithm:
//!
//! 1. **Pass 1**: Count output size (union or intersection semantics)
//! 2. **Scan**: Exclusive prefix sum to get output positions
//! 3. **Pass 2**: Compute merged values
//!
//! # Architecture
//!
//! ```text
//! MergeStrategy trait
//! ├── UnionMerge (add, sub) - keeps all positions from either matrix
//! └── IntersectionMerge (mul, div) - keeps only positions in both matrices
//!
//! SparseMergeOp enum
//! ├── Add - a + b
//! ├── Sub - a - b
//! ├── Mul - a * b
//! └── Div - a / b
//! ```

use std::marker::PhantomData;

/// Sparse element-wise operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMergeOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl SparseMergeOp {
    /// Get the kernel prefix for this operation
    pub fn kernel_prefix(&self) -> &'static str {
        match self {
            SparseMergeOp::Add => "add",
            SparseMergeOp::Sub => "sub",
            SparseMergeOp::Mul => "mul",
            SparseMergeOp::Div => "div",
        }
    }

    /// Check if this operation uses union semantics (keeps positions from either matrix)
    pub fn is_union(&self) -> bool {
        matches!(self, SparseMergeOp::Add | SparseMergeOp::Sub)
    }

    /// Check if this operation uses intersection semantics (keeps only positions in both)
    pub fn is_intersection(&self) -> bool {
        matches!(self, SparseMergeOp::Mul | SparseMergeOp::Div)
    }
}

/// Merge strategy trait that captures operation semantics
///
/// This trait enables generic implementation of sparse merge operations
/// by abstracting over union vs intersection semantics.
pub trait MergeStrategy: Copy {
    /// The operation this strategy implements
    const OP: SparseMergeOp;

    /// Whether this strategy uses union semantics (true for add/sub, false for mul/div)
    const IS_UNION: bool;

    /// Get the count kernel name for a given sparse format
    fn count_kernel_name(format: SparseFormat) -> &'static str {
        if Self::IS_UNION {
            // Union semantics: generic merge_count kernel
            match format {
                SparseFormat::Csr => "csr_merge_count",
                SparseFormat::Csc => "csc_merge_count",
                SparseFormat::Coo => panic!("COO uses different algorithm"),
            }
        } else {
            // Intersection semantics: operation-specific kernel
            match format {
                SparseFormat::Csr => match Self::OP {
                    SparseMergeOp::Mul => "csr_mul_count",
                    SparseMergeOp::Div => "csr_mul_count", // div uses same count as mul
                    _ => unreachable!(),
                },
                SparseFormat::Csc => match Self::OP {
                    SparseMergeOp::Mul => "csc_mul_count",
                    SparseMergeOp::Div => "csc_mul_count",
                    _ => unreachable!(),
                },
                SparseFormat::Coo => panic!("COO uses different algorithm"),
            }
        }
    }

    /// Get the compute kernel name for a given sparse format and dtype
    fn compute_kernel_name(format: SparseFormat, dtype_suffix: &str) -> String {
        let format_prefix = match format {
            SparseFormat::Csr => "csr",
            SparseFormat::Csc => "csc",
            SparseFormat::Coo => "coo",
        };
        format!(
            "{}_{}_{}",
            format_prefix,
            Self::OP.kernel_prefix(),
            dtype_suffix
        )
    }
}

/// Sparse matrix format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row
    Csr,
    /// Compressed Sparse Column
    Csc,
    /// Coordinate format
    Coo,
}

// ============================================================================
// Strategy Implementations
// ============================================================================

/// Add merge strategy (union semantics)
#[derive(Debug, Clone, Copy)]
pub struct AddMerge;

impl MergeStrategy for AddMerge {
    const OP: SparseMergeOp = SparseMergeOp::Add;
    const IS_UNION: bool = true;
}

/// Subtract merge strategy (union semantics)
#[derive(Debug, Clone, Copy)]
pub struct SubMerge;

impl MergeStrategy for SubMerge {
    const OP: SparseMergeOp = SparseMergeOp::Sub;
    const IS_UNION: bool = true;
}

/// Multiply merge strategy (intersection semantics)
#[derive(Debug, Clone, Copy)]
pub struct MulMerge;

impl MergeStrategy for MulMerge {
    const OP: SparseMergeOp = SparseMergeOp::Mul;
    const IS_UNION: bool = false;
}

/// Divide merge strategy (intersection semantics)
#[derive(Debug, Clone, Copy)]
pub struct DivMerge;

impl MergeStrategy for DivMerge {
    const OP: SparseMergeOp = SparseMergeOp::Div;
    const IS_UNION: bool = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_op_semantics() {
        assert!(SparseMergeOp::Add.is_union());
        assert!(SparseMergeOp::Sub.is_union());
        assert!(SparseMergeOp::Mul.is_intersection());
        assert!(SparseMergeOp::Div.is_intersection());
    }

    #[test]
    fn test_strategy_count_kernels() {
        // Union strategies use generic merge_count
        assert_eq!(
            AddMerge::count_kernel_name(SparseFormat::Csr),
            "csr_merge_count"
        );
        assert_eq!(
            SubMerge::count_kernel_name(SparseFormat::Csc),
            "csc_merge_count"
        );

        // Intersection strategies use operation-specific kernels
        assert_eq!(
            MulMerge::count_kernel_name(SparseFormat::Csr),
            "csr_mul_count"
        );
        assert_eq!(
            DivMerge::count_kernel_name(SparseFormat::Csr),
            "csr_mul_count"
        );
    }

    #[test]
    fn test_strategy_compute_kernels() {
        assert_eq!(
            AddMerge::compute_kernel_name(SparseFormat::Csr, "f32"),
            "csr_add_f32"
        );
        assert_eq!(
            SubMerge::compute_kernel_name(SparseFormat::Csc, "f64"),
            "csc_sub_f64"
        );
        assert_eq!(
            MulMerge::compute_kernel_name(SparseFormat::Coo, "f32"),
            "coo_mul_f32"
        );
    }
}
