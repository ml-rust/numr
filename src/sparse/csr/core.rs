//! Core CSR implementation: struct, creation, getters

use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::super::format::{SparseFormat, SparseStorage};

/// CSR (Compressed Sparse Row) sparse matrix data
#[derive(Debug, Clone)]
pub struct CsrData<R: Runtime> {
    pub(crate) row_ptrs: Tensor<R>,
    pub(crate) col_indices: Tensor<R>,
    pub(crate) values: Tensor<R>,
    pub(crate) shape: [usize; 2],
}

impl<R: Runtime> CsrData<R> {
    /// Create a new CSR matrix from components
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - Row pointers (length: nrows + 1)
    /// * `col_indices` - Column indices for each non-zero
    /// * `values` - Values at each position
    /// * `shape` - Matrix shape [nrows, ncols]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - row_ptrs length != nrows + 1
    /// - col_indices and values have different lengths
    /// - Index tensors are not I64
    pub fn new(
        row_ptrs: Tensor<R>,
        col_indices: Tensor<R>,
        values: Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Self> {
        let [nrows, _ncols] = shape;
        let nnz = values.numel();

        // Validate row_ptrs length
        if row_ptrs.numel() != nrows + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![nrows + 1],
                got: vec![row_ptrs.numel()],
            });
        }

        // Validate col_indices length
        if col_indices.numel() != nnz {
            return Err(Error::ShapeMismatch {
                expected: vec![nnz],
                got: vec![col_indices.numel()],
            });
        }

        // Validate dtypes
        if row_ptrs.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: row_ptrs.dtype(),
            });
        }
        if col_indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: col_indices.dtype(),
            });
        }

        // Validate 1D
        if row_ptrs.ndim() != 1 || col_indices.ndim() != 1 || values.ndim() != 1 {
            return Err(Error::Internal(format!(
                "Expected 1D tensors, got row_ptrs: {}D, col_indices: {}D, values: {}D",
                row_ptrs.ndim(),
                col_indices.ndim(),
                values.ndim()
            )));
        }

        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            shape,
        })
    }

    /// Create an empty CSR matrix
    pub fn empty(shape: [usize; 2], dtype: DType, device: &R::Device) -> Self {
        let [nrows, _ncols] = shape;
        // Row pointers are all zeros for empty matrix
        let row_ptrs_data: Vec<i64> = vec![0; nrows + 1];

        Self {
            row_ptrs: Tensor::from_slice(&row_ptrs_data, &[nrows + 1], device),
            col_indices: Tensor::empty(&[0], DType::I64, device),
            values: Tensor::empty(&[0], dtype, device),
            shape,
        }
    }

    /// Returns the row pointers tensor
    pub fn row_ptrs(&self) -> &Tensor<R> {
        &self.row_ptrs
    }

    /// Returns the column indices tensor
    pub fn col_indices(&self) -> &Tensor<R> {
        &self.col_indices
    }

    /// Returns the values tensor
    pub fn values(&self) -> &Tensor<R> {
        &self.values
    }

    /// Returns the number of non-zeros in a specific row
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if row >= nrows (only in debug mode)
    ///
    /// # Note
    ///
    /// This method transfers row_ptrs from device memory to host (if on GPU).
    /// For batch queries, consider using the row_ptrs tensor directly.
    pub fn row_nnz(&self, row: usize) -> usize {
        debug_assert!(row < self.nrows());
        // FIXME: GPU-incompatible - requires host memory access
        // For GPU tensors, this causes device-to-host transfer
        let row_ptrs: Vec<i64> = self.row_ptrs.to_vec();
        (row_ptrs[row + 1] - row_ptrs[row]) as usize
    }
}

impl<R: Runtime> SparseStorage for CsrData<R> {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csr
    }

    fn shape(&self) -> [usize; 2] {
        self.shape
    }

    fn nnz(&self) -> usize {
        self.values.numel()
    }

    fn dtype(&self) -> DType {
        self.values.dtype()
    }

    fn memory_usage(&self) -> usize {
        // row_ptrs (I64) + col_indices (I64) + values
        let ptr_size = (self.nrows() + 1) * std::mem::size_of::<i64>();
        let index_size = self.nnz() * std::mem::size_of::<i64>();
        let value_size = self.nnz() * self.dtype().size_in_bytes();
        ptr_size + index_size + value_size
    }
}

/// Create CSR data from host arrays
impl<R: Runtime> CsrData<R> {
    /// Create CSR matrix from host slices
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - Row pointers (length: nrows + 1)
    /// * `col_indices` - Column indices
    /// * `values` - Non-zero values
    /// * `shape` - Matrix shape [nrows, ncols]
    /// * `device` - Target device
    pub fn from_slices<T: Element>(
        row_ptrs: &[i64],
        col_indices: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        let [nrows, ncols] = shape;

        // Validate row_ptrs length
        if row_ptrs.len() != nrows + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![nrows + 1],
                got: vec![row_ptrs.len()],
            });
        }

        // Validate lengths match
        if col_indices.len() != values.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![values.len()],
                got: vec![col_indices.len()],
            });
        }

        // Validate row_ptrs is monotonic and matches nnz
        let nnz = values.len();
        if row_ptrs[0] != 0 || row_ptrs[nrows] as usize != nnz {
            return Err(Error::Internal(format!(
                "Invalid row_ptrs: expected [0]=0 and [{}]={}, got [0]={} and [{}]={}",
                nrows, nnz, row_ptrs[0], nrows, row_ptrs[nrows]
            )));
        }

        // Validate column indices
        for &c in col_indices {
            if c < 0 {
                return Err(Error::Internal(format!("Negative column index: {}", c)));
            }
            if c as usize >= ncols {
                return Err(Error::IndexOutOfBounds {
                    index: c as usize,
                    size: ncols,
                });
            }
        }

        let row_ptrs_tensor = Tensor::from_slice(row_ptrs, &[row_ptrs.len()], device);
        let col_indices_tensor = Tensor::from_slice(col_indices, &[col_indices.len()], device);
        let values_tensor = Tensor::from_slice(values, &[values.len()], device);

        Self::new(row_ptrs_tensor, col_indices_tensor, values_tensor, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::SparseFormat;
    use crate::tensor::Tensor;

    #[test]
    fn test_csr_creation() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device);
        assert!(csr.is_ok());

        let csr = csr.unwrap();
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), [3, 3]);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.dtype(), DType::F32);
    }

    #[test]
    fn test_csr_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csr = CsrData::<CpuRuntime>::empty([100, 200], DType::F64, &device);

        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.shape(), [100, 200]);
        assert!(csr.is_empty());
        assert_eq!(csr.row_ptrs().numel(), 101); // nrows + 1
    }

    #[test]
    fn test_csr_memory_usage() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // 4 row_ptrs * 8 bytes + 5 col_indices * 8 bytes + 5 values * 4 bytes
        // = 32 + 40 + 20 = 92 bytes
        assert_eq!(csr.memory_usage(), 92);
    }

    #[test]
    fn test_csr_invalid_row_ptrs() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let row_ptrs = vec![0i64, 2, 3]; // Wrong length (should be 4 for 3 rows)
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let result =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device);
        assert!(result.is_err());
    }
}
