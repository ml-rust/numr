//! Element-wise subtraction for COO matrices

use super::super::CooData;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseStorage};

impl<R: Runtime> CooData<R> {
    /// Element-wise subtraction: C = A - B
    ///
    /// Computes the difference of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another COO matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new COO matrix containing the element-wise difference (sorted by row, then column)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Concatenates triplets from both matrices, sorts, and merges duplicates.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O((nnz_a + nnz_b) log(nnz_a + nnz_b)) for sorting
    /// - GPU: O((nnz_a + nnz_b) log(nnz_a + nnz_b)) parallel sort-merge
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A - B:
    /// // [5, 0]      [2, 1]      [3, -1]
    /// // [0, 4]  -   [0, 3]  =   [0,  1]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[5.0f32, 4.0], [2, 2], &device)?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[2.0f32, 1.0, 3.0], [2, 2], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(a) = a_sp { if let numr::sparse::SparseTensor::Coo(b) = b_sp {
    /// let c = a.sub(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape[0], self.shape[1]],
                got: vec![other.shape[0], other.shape[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        let dtype = self.dtype();
        let device = self.values.device();

        // Get client for runtime dispatch
        let client = R::default_client(device);

        // Dispatch to runtime-specific implementation
        crate::dispatch_dtype!(dtype, T => {
            let (out_row_indices, out_col_indices, out_values) = client.sub_coo::<T>(
                &self.row_indices,
                &self.col_indices,
                &self.values,
                &other.row_indices,
                &other.col_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                row_indices: out_row_indices,
                col_indices: out_col_indices,
                values: out_values,
                shape: self.shape,
                sorted: true,  // Backend guarantees sorted output
            })
        }, "coo_sub")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_coo_sub_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 1]
        // [0, 4]     [0, 3]
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 1],
            &[2.0f32, 1.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        // C = A - B:
        // [3, -1]
        // [0,  1]
        assert_eq!(c.shape(), [2, 2]);
        assert!(c.is_sorted());
    }
}
