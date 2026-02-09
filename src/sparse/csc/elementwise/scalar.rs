//! Scalar operations for CSC matrices

use super::super::CscData;
use crate::error::Result;
use crate::ops::ScalarOps;
use crate::runtime::Runtime;

impl<R: Runtime> CscData<R> {
    /// Scalar multiplication: C = A * scalar
    ///
    /// Multiplies all non-zero values by a scalar constant.
    /// Preserves the sparsity pattern.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply with
    ///
    /// # Performance
    ///
    /// O(nnz) - simply scales the values tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0], &[0], &[1.0f32], [1, 1], &device)?.to_csc()?;
    /// # if let numr::sparse::SparseTensor::Csc(csc) = sp {
    /// let result = csc.scalar_mul(2.0)?;  // Multiply all values by 2
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn scalar_mul(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        let device = self.values.device();
        let client = R::default_client(device);

        let scaled_values = client.mul_scalar(&self.values, scalar)?;

        Ok(Self {
            col_ptrs: self.col_ptrs.clone(),
            row_indices: self.row_indices.clone(),
            values: scaled_values,
            shape: self.shape,
        })
    }

    /// Add scalar to non-zero elements only (sparsity-preserving)
    ///
    /// # ⚠️ Important: This is NOT Standard Scalar Addition!
    ///
    /// Standard mathematical scalar addition (`A + s`) adds `s` to **ALL** elements
    /// including implicit zeros, creating a dense matrix. This operation only adds
    /// to existing non-zero values, preserving the sparse structure.
    ///
    /// Use this when you want to shift all non-zero values by a constant without
    /// densifying the matrix. For true scalar addition, convert to dense first.
    ///
    /// # Mathematical Behavior
    ///
    /// - **Non-zero elements**: `A[i,j] + s`
    /// - **Zero elements**: remain `0` (NOT `s`!)
    ///
    /// # Performance
    ///
    /// O(nnz) - simply adds to the values tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // Sparse matrix:        After scalar_add(10):
    /// // [1, 0, 2]             [11,  0, 12]
    /// // [0, 3, 0]             [ 0, 13,  0]
    /// //
    /// // Note: Zeros stay 0, not 10!
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 2, 1], &[1.0f32, 2.0, 3.0], [2, 3], &device)?.to_csc()?;
    /// # if let numr::sparse::SparseTensor::Csc(csc) = sp {
    /// let result = csc.scalar_add(10.0)?;
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`add_to_nonzeros()`](Self::add_to_nonzeros) - Clearer alias for this method
    pub fn scalar_add(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        // Handle empty tensor case (no values to add to)
        if self.values.numel() == 0 {
            return Ok(Self {
                col_ptrs: self.col_ptrs.clone(),
                row_indices: self.row_indices.clone(),
                values: self.values.clone(),
                shape: self.shape,
            });
        }

        let device = self.values.device();
        let client = R::default_client(device);

        let shifted_values = client.add_scalar(&self.values, scalar)?;

        Ok(Self {
            col_ptrs: self.col_ptrs.clone(),
            row_indices: self.row_indices.clone(),
            values: shifted_values,
            shape: self.shape,
        })
    }

    /// Alias for [`scalar_add()`](Self::scalar_add) with clearer naming
    ///
    /// Adds a scalar value to all non-zero elements, preserving sparsity.
    /// This name makes it explicit that only non-zero elements are modified.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // Clearer intent than scalar_add
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0], &[0], &[1.0f32], [1, 1], &device)?.to_csc()?;
    /// # if let numr::sparse::SparseTensor::Csc(csc) = sp {
    /// let result = csc.add_to_nonzeros(5.0)?;
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    #[inline]
    pub fn add_to_nonzeros(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        self.scalar_add(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::SparseStorage;

    #[test]
    fn test_csc_scalar_mul() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let csc = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 4.0, 5.0, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = csc.scalar_mul(2.0).unwrap();

        assert_eq!(result.nnz(), 5);
        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![2.0, 8.0, 10.0, 4.0, 6.0]);
    }

    #[test]
    fn test_csc_scalar_add() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let csc = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 4.0, 5.0, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = csc.scalar_add(1.0).unwrap();

        assert_eq!(result.nnz(), 5);
        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![2.0, 5.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn test_csc_scalar_ops_preserves_structure() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let csc = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 4.0, 5.0, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = csc.scalar_mul(3.0).unwrap();

        // Verify structure is preserved
        let col_ptrs: Vec<i64> = result.col_ptrs().to_vec();
        let row_indices: Vec<i64> = result.row_indices().to_vec();

        assert_eq!(col_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(row_indices, vec![0, 2, 2, 0, 1]);
    }
}
