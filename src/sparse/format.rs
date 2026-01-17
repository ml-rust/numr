//! Sparse format definitions and traits

use crate::dtype::DType;

/// Sparse matrix storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Coordinate format (COO)
    ///
    /// Stores explicit (row, col, value) triplets.
    /// Best for: construction, format conversion, random access insertion
    /// Storage: O(3 * nnz)
    Coo,

    /// Compressed Sparse Row (CSR)
    ///
    /// Row pointers + column indices + values.
    /// Best for: row slicing, SpMV, most sparse operations
    /// Storage: O(2 * nnz + nrows + 1)
    Csr,

    /// Compressed Sparse Column (CSC)
    ///
    /// Column pointers + row indices + values.
    /// Best for: column slicing, transposed operations
    /// Storage: O(2 * nnz + ncols + 1)
    Csc,
}

impl SparseFormat {
    /// Returns true if format is efficient for row operations
    #[inline]
    pub fn is_row_major(&self) -> bool {
        matches!(self, SparseFormat::Csr)
    }

    /// Returns true if format is efficient for column operations
    #[inline]
    pub fn is_col_major(&self) -> bool {
        matches!(self, SparseFormat::Csc)
    }

    /// Returns the format name as a string
    pub fn name(&self) -> &'static str {
        match self {
            SparseFormat::Coo => "COO",
            SparseFormat::Csr => "CSR",
            SparseFormat::Csc => "CSC",
        }
    }
}

impl std::fmt::Display for SparseFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for sparse storage backends
///
/// This trait defines the common interface for all sparse storage formats.
/// Each format (COO, CSR, CSC) implements this trait.
pub trait SparseStorage: Sized {
    /// Returns the sparse format type
    fn format(&self) -> SparseFormat;

    /// Returns the shape as [nrows, ncols]
    fn shape(&self) -> [usize; 2];

    /// Returns the number of rows
    #[inline]
    fn nrows(&self) -> usize {
        self.shape()[0]
    }

    /// Returns the number of columns
    #[inline]
    fn ncols(&self) -> usize {
        self.shape()[1]
    }

    /// Returns the number of non-zero elements
    fn nnz(&self) -> usize;

    /// Returns the data type of values
    fn dtype(&self) -> DType;

    /// Returns the sparsity ratio (fraction of zeros)
    ///
    /// Sparsity = 1.0 - (nnz / total_elements)
    #[inline]
    fn sparsity(&self) -> f64 {
        let total = (self.nrows() * self.ncols()) as f64;
        if total == 0.0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total)
        }
    }

    /// Returns the density ratio (fraction of non-zeros)
    ///
    /// Density = nnz / total_elements = 1.0 - sparsity
    #[inline]
    fn density(&self) -> f64 {
        1.0 - self.sparsity()
    }

    /// Returns true if the matrix is empty (no non-zeros)
    #[inline]
    fn is_empty(&self) -> bool {
        self.nnz() == 0
    }

    /// Returns the memory usage in bytes (approximate)
    fn memory_usage(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_format_display() {
        assert_eq!(SparseFormat::Coo.to_string(), "COO");
        assert_eq!(SparseFormat::Csr.to_string(), "CSR");
        assert_eq!(SparseFormat::Csc.to_string(), "CSC");
    }

    #[test]
    fn test_format_properties() {
        assert!(!SparseFormat::Coo.is_row_major());
        assert!(SparseFormat::Csr.is_row_major());
        assert!(!SparseFormat::Csc.is_row_major());

        assert!(!SparseFormat::Coo.is_col_major());
        assert!(!SparseFormat::Csr.is_col_major());
        assert!(SparseFormat::Csc.is_col_major());
    }
}
