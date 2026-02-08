//! CSC matrix multiplication: spmv, spmm

use super::CscData;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

impl<R: Runtime> CscData<R> {
    /// Sparse matrix-vector multiplication: y = A * x
    ///
    /// Converts to CSR format (optimal for SpMV) and performs the multiplication.
    ///
    /// # Arguments
    ///
    /// * `x` - Dense vector of length `ncols`
    ///
    /// # Returns
    ///
    /// Dense vector of length `nrows`
    ///
    /// # Performance
    ///
    /// This method converts to CSR first, adding conversion overhead.
    /// For repeated SpMV, convert to CSR once and reuse.
    /// CSC is optimal for transposed SpMV (y = A^T * x).
    pub fn spmv(&self, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        let csr = self.to_csr()?;
        csr.spmv(x)
    }

    /// Sparse matrix-dense matrix multiplication: C = A * B
    ///
    /// Converts to CSR format and performs the multiplication.
    ///
    /// # Arguments
    ///
    /// * `b` - Dense matrix of shape [K, N] where K == ncols of sparse matrix
    ///
    /// # Returns
    ///
    /// Dense matrix of shape [M, N]
    ///
    /// # Performance
    ///
    /// This method converts to CSR first, adding conversion overhead.
    /// For repeated SpMM, convert to CSR once and reuse.
    pub fn spmm(&self, b: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        let csr = self.to_csr()?;
        csr.spmm(b)
    }

    /// Transpose the sparse matrix: B = A^T
    ///
    /// Returns the transpose as a CSR matrix. This is an O(1) operation
    /// that reinterprets the CSC structure as CSR:
    /// - col_ptrs become row_ptrs
    /// - row_indices become col_indices
    /// - values remain the same
    /// - shape is swapped
    ///
    /// # Returns
    ///
    /// CSR matrix representing the transpose
    ///
    /// # Performance
    ///
    /// O(1) - structural reinterpretation, no data copying beyond cloning tensors.
    ///
    /// # Returns
    ///
    /// Returns `CsrData<R>` (not `CscData<R>`) because the transpose of a
    /// column-compressed matrix is mathematically equivalent to a row-compressed
    /// matrix: **CSC^T = CSR**.
    ///
    /// This is an optimization that avoids unnecessary format conversion:
    /// - CSC stores data column-wise (compressed columns)
    /// - Transposing swaps rows↔columns
    /// - The result naturally becomes row-wise (CSR format)
    ///
    /// # Mathematical Equivalence
    ///
    /// For a CSC matrix with:
    /// - `col_ptrs[j]...col_ptrs[j+1]`: non-zeros in column j
    /// - `row_indices[k]`: row index of k-th non-zero
    ///
    /// The transpose becomes CSR with:
    /// - `row_ptrs = col_ptrs` (columns become rows)
    /// - `col_indices = row_indices` (rows become columns)
    ///
    /// # Performance
    ///
    /// O(1) - Just swaps index arrays and dimensions, no data copying.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // CSC matrix [2, 3]:
    /// // [1, 0, 2]
    /// // [0, 3, 0]
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo(&[0, 0, 1], &[0, 2, 1], &[1.0f32, 2.0, 3.0], &[2, 3], &device)?.to_csc()?;
    /// # if let numr::sparse::SparseTensor::Csc(csc) = sp {
    ///
    /// // Transpose returns CSR [3, 2]:
    /// // [1, 0]
    /// // [0, 3]
    /// // [2, 0]
    /// let csr = csc.transpose();
    /// assert_eq!(csr.shape(), [3, 2]);
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn transpose(&self) -> CsrData<R> {
        let [nrows, ncols] = self.shape;
        // CSC col_ptrs -> CSR row_ptrs
        // CSC row_indices -> CSR col_indices
        // Shape [nrows, ncols] -> [ncols, nrows]
        CsrData {
            row_ptrs: self.col_ptrs.clone(),
            col_indices: self.row_indices.clone(),
            values: self.values.clone(),
            shape: [ncols, nrows],
        }
    }
}

// No tests needed - these just delegate to CSR
