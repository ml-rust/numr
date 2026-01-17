//! Core CSC implementation: struct, creation, getters

use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::super::format::{SparseFormat, SparseStorage};

/// CSC (Compressed Sparse Column) sparse matrix data
#[derive(Debug, Clone)]
pub struct CscData<R: Runtime> {
    pub(crate) col_ptrs: Tensor<R>,
    pub(crate) row_indices: Tensor<R>,
    pub(crate) values: Tensor<R>,
    pub(crate) shape: [usize; 2],
}

impl<R: Runtime> CscData<R> {
    /// Create a new CSC matrix from components
    pub fn new(
        col_ptrs: Tensor<R>,
        row_indices: Tensor<R>,
        values: Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Self> {
        let [_nrows, ncols] = shape;
        let nnz = values.numel();

        // Validate col_ptrs length
        if col_ptrs.numel() != ncols + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols + 1],
                got: vec![col_ptrs.numel()],
            });
        }

        // Validate row_indices length
        if row_indices.numel() != nnz {
            return Err(Error::ShapeMismatch {
                expected: vec![nnz],
                got: vec![row_indices.numel()],
            });
        }

        // Validate dtypes
        if col_ptrs.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: col_ptrs.dtype(),
            });
        }
        if row_indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: row_indices.dtype(),
            });
        }

        // Validate 1D
        if col_ptrs.ndim() != 1 || row_indices.ndim() != 1 || values.ndim() != 1 {
            return Err(Error::Internal(format!(
                "Expected 1D tensors, got col_ptrs: {}D, row_indices: {}D, values: {}D",
                col_ptrs.ndim(),
                row_indices.ndim(),
                values.ndim()
            )));
        }

        Ok(Self {
            col_ptrs,
            row_indices,
            values,
            shape,
        })
    }

    /// Create an empty CSC matrix
    pub fn empty(shape: [usize; 2], dtype: DType, device: &R::Device) -> Self {
        let [_nrows, ncols] = shape;
        let col_ptrs_data: Vec<i64> = vec![0; ncols + 1];

        Self {
            col_ptrs: Tensor::from_slice(&col_ptrs_data, &[ncols + 1], device),
            row_indices: Tensor::empty(&[0], DType::I64, device),
            values: Tensor::empty(&[0], dtype, device),
            shape,
        }
    }

    /// Returns the column pointers tensor
    pub fn col_ptrs(&self) -> &Tensor<R> {
        &self.col_ptrs
    }

    /// Returns the row indices tensor
    pub fn row_indices(&self) -> &Tensor<R> {
        &self.row_indices
    }

    /// Returns the values tensor
    pub fn values(&self) -> &Tensor<R> {
        &self.values
    }
}

impl<R: Runtime> SparseStorage for CscData<R> {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csc
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
        let ptr_size = (self.ncols() + 1) * std::mem::size_of::<i64>();
        let index_size = self.nnz() * std::mem::size_of::<i64>();
        let value_size = self.nnz() * self.dtype().size_in_bytes();
        ptr_size + index_size + value_size
    }
}

impl<R: Runtime> CscData<R> {
    /// Create CSC matrix from host slices
    pub fn from_slices<T: Element>(
        col_ptrs: &[i64],
        row_indices: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        let [nrows, ncols] = shape;

        if col_ptrs.len() != ncols + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols + 1],
                got: vec![col_ptrs.len()],
            });
        }

        if row_indices.len() != values.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![values.len()],
                got: vec![row_indices.len()],
            });
        }

        let nnz = values.len();
        if col_ptrs[0] != 0 || col_ptrs[ncols] as usize != nnz {
            return Err(Error::Internal(format!(
                "Invalid col_ptrs: expected [0]=0 and [{}]={}, got [0]={} and [{}]={}",
                ncols, nnz, col_ptrs[0], ncols, col_ptrs[ncols]
            )));
        }

        for &r in row_indices {
            if r < 0 {
                return Err(Error::Internal(format!("Negative row index: {}", r)));
            }
            if r as usize >= nrows {
                return Err(Error::IndexOutOfBounds {
                    index: r as usize,
                    size: nrows,
                });
            }
        }

        let col_ptrs_tensor = Tensor::from_slice(col_ptrs, &[col_ptrs.len()], device);
        let row_indices_tensor = Tensor::from_slice(row_indices, &[row_indices.len()], device);
        let values_tensor = Tensor::from_slice(values, &[values.len()], device);

        Self::new(col_ptrs_tensor, row_indices_tensor, values_tensor, shape)
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
    fn test_csc_creation() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix (same as CSR test, but column-oriented):
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device);
        assert!(csc.is_ok());

        let csc = csc.unwrap();
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), [3, 3]);
        assert_eq!(csc.format(), SparseFormat::Csc);
    }

    #[test]
    fn test_csc_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csc = CscData::<CpuRuntime>::empty([100, 200], DType::F64, &device);

        assert_eq!(csc.nnz(), 0);
        assert_eq!(csc.shape(), [100, 200]);
        assert!(csc.is_empty());
        assert_eq!(csc.col_ptrs().numel(), 201); // ncols + 1
    }
}
