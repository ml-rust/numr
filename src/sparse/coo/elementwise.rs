//! COO element-wise operations: add, sub, mul, div, scalar_mul, scalar_add
//!
//! All element-wise operations dispatch to runtime-specific implementations
//! via the SparseOps trait, enabling GPU acceleration when available.

use super::CooData;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseStorage};
use crate::tensor::Tensor;

impl<R: Runtime> CooData<R> {
    /// Element-wise addition: C = A + B
    ///
    /// Computes the sum of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another COO matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new COO matrix containing the element-wise sum (sorted by row, then column)
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
    /// ```ignore
    /// // A:          B:          C = A + B:
    /// // [1, 0]      [0, 2]      [1, 2]
    /// // [0, 3]  +   [4, 0]  =   [4, 3]
    /// let c = a.add(&b)?;
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self>
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
            let (out_row_indices, out_col_indices, out_values) = client.add_coo::<T>(
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
        }, "coo_add")
    }

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
    /// ```ignore
    /// // A:          B:          C = A - B:
    /// // [5, 0]      [2, 1]      [3, -1]
    /// // [0, 4]  -   [0, 3]  =   [0,  1]
    /// let c = a.sub(&b)?;
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

    /// Element-wise multiplication (Hadamard product): C = A .* B
    ///
    /// Computes the element-wise product of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another COO matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new COO matrix containing the element-wise product (sorted by row, then column)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Sorts both matrices, then performs linear merge to find matching positions.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a log nnz_a + nnz_b log nnz_b + nnz_a + nnz_b)
    /// - GPU: Parallel sort-merge on device
    /// - Result has at most min(nnz_a, nnz_b) non-zeros
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A:          B:          C = A .* B:
    /// // [2, 3]      [4, 0]      [8, 0]
    /// // [0, 5]  .*  [6, 7]  =   [0, 35]
    /// let c = a.mul(&b)?;
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self>
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
            let (out_row_indices, out_col_indices, out_values) = client.mul_coo::<T>(
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
        }, "coo_mul")
    }

    /// Element-wise division: C = A ./ B
    ///
    /// Computes the element-wise quotient of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another COO matrix with the same shape and dtype (divisor)
    ///
    /// # Returns
    ///
    /// A new COO matrix containing the element-wise quotient (sorted by row, then column)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// 1. Sort both matrices by (row, col)
    /// 2. Linear merge to find matching positions
    /// 3. Divide values at matching positions (A / B)
    ///
    /// # Performance
    ///
    /// O(nnz_a log nnz_a + nnz_b log nnz_b) for sorting, O(nnz_a + nnz_b) for merge.
    /// Result has at most min(nnz_a, nnz_b) non-zeros.
    ///
    /// # Note
    ///
    /// Division by zero can occur if B stores an explicit zero value. The result
    /// will be infinity or NaN depending on the numerator.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A:          B:          C = A ./ B:
    /// // [8, 3]      [4, 0]      [2, 0]
    /// // [0, 10]  ./ [6, 2]  =   [0, 5]
    /// let c = a.div(&b)?;
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self> {
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

        // Handle empty cases - if either is empty, result is empty
        if self.nnz() == 0 || other.nnz() == 0 {
            return Ok(Self::empty(self.shape, dtype, device));
        }

        // Read indices to host
        let rows_a: Vec<i64> = self.row_indices.to_vec();
        let cols_a: Vec<i64> = self.col_indices.to_vec();
        let rows_b: Vec<i64> = other.row_indices.to_vec();
        let cols_b: Vec<i64> = other.col_indices.to_vec();

        // Sort both matrices by (row, col)
        let mut perm_a: Vec<usize> = (0..self.nnz()).collect();
        perm_a.sort_by(|&i, &j| {
            let row_cmp = rows_a[i].cmp(&rows_a[j]);
            if row_cmp != std::cmp::Ordering::Equal {
                row_cmp
            } else {
                cols_a[i].cmp(&cols_a[j])
            }
        });

        let mut perm_b: Vec<usize> = (0..other.nnz()).collect();
        perm_b.sort_by(|&i, &j| {
            let row_cmp = rows_b[i].cmp(&rows_b[j]);
            if row_cmp != std::cmp::Ordering::Equal {
                row_cmp
            } else {
                cols_b[i].cmp(&cols_b[j])
            }
        });

        // Dispatch on dtype to merge and divide values
        crate::dispatch_dtype!(dtype, T => {
            let vals_a: Vec<T> = self.values.to_vec();
            let vals_b: Vec<T> = other.values.to_vec();

            // Linear merge to find matching positions
            let mut result_rows: Vec<i64> = Vec::new();
            let mut result_cols: Vec<i64> = Vec::new();
            let mut result_vals: Vec<T> = Vec::new();

            let mut i = 0;
            let mut j = 0;

            while i < perm_a.len() && j < perm_b.len() {
                let idx_a = perm_a[i];
                let idx_b = perm_b[j];

                let row_a = rows_a[idx_a];
                let col_a = cols_a[idx_a];
                let row_b = rows_b[idx_b];
                let col_b = cols_b[idx_b];

                if (row_a, col_a) < (row_b, col_b) {
                    // A only - skip (result is 0)
                    i += 1;
                } else if (row_a, col_a) > (row_b, col_b) {
                    // B only - skip (result is 0)
                    j += 1;
                } else {
                    // Same position - divide values
                    let quotient = vals_a[idx_a].to_f64() / vals_b[idx_b].to_f64();
                    result_rows.push(row_a);
                    result_cols.push(col_a);
                    result_vals.push(T::from_f64(quotient));
                    i += 1;
                    j += 1;
                }
            }

            // Handle empty result
            if result_rows.is_empty() {
                return Ok(Self::empty(self.shape, dtype, device));
            }

            let row_tensor = Tensor::from_slice(&result_rows, &[result_rows.len()], device);
            let col_tensor = Tensor::from_slice(&result_cols, &[result_cols.len()], device);
            let val_tensor = Tensor::from_slice(&result_vals, &[result_vals.len()], device);

            return Ok(Self {
                row_indices: row_tensor,
                col_indices: col_tensor,
                values: val_tensor,
                shape: self.shape,
                sorted: true,
            });
        }, "COO element-wise div");
    }

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
    /// ```ignore
    /// let result = coo.scalar_mul(2.0)?;  // Multiply all values by 2
    /// ```
    pub fn scalar_mul(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        let device = self.values.device();
        let client = R::default_client(device);

        let scaled_values = client.mul_scalar(&self.values, scalar)?;

        Ok(Self {
            row_indices: self.row_indices.clone(),
            col_indices: self.col_indices.clone(),
            values: scaled_values,
            shape: self.shape,
            sorted: self.sorted,
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
    /// ```ignore
    /// // Sparse matrix:        After scalar_add(10):
    /// // [1, 0, 2]             [11,  0, 12]
    /// // [0, 3, 0]             [ 0, 13,  0]
    /// //
    /// // Note: Zeros stay 0, not 10!
    /// let result = coo.scalar_add(10.0)?;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`add_to_nonzeros()`](Self::add_to_nonzeros) - Clearer alias for this method
    pub fn scalar_add(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        let device = self.values.device();
        let client = R::default_client(device);

        let shifted_values = client.add_scalar(&self.values, scalar)?;

        Ok(Self {
            row_indices: self.row_indices.clone(),
            col_indices: self.col_indices.clone(),
            values: shifted_values,
            shape: self.shape,
            sorted: self.sorted,
        })
    }

    /// Alias for [`scalar_add()`](Self::scalar_add) with clearer naming
    ///
    /// Adds a scalar value to all non-zero elements, preserving sparsity.
    /// This name makes it explicit that only non-zero elements are modified.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Clearer intent than scalar_add
    /// let result = coo.add_to_nonzeros(5.0)?;
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
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::SparseFormat;
    use crate::tensor::Tensor;

    // =========================================================================
    // Element-wise add tests
    // =========================================================================

    #[test]
    fn test_coo_add_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        // C = A + B:
        // [1, 2]
        // [4, 3]
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.nnz(), 4);
        assert!(c.is_sorted());

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 0, 1, 1]);
        assert_eq!(cols, vec![0, 1, 0, 1]);
        assert_eq!(vals, vec![1.0, 2.0, 4.0, 3.0]);
    }

    #[test]
    fn test_coo_add_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 2]     [3, 0]
        // [0, 0]     [0, 4]
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 0],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[3.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        // C = A + B:
        // [4, 2]   (1+3=4 at (0,0))
        // [0, 4]
        assert_eq!(c.nnz(), 3);

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 0, 1]);
        assert_eq!(cols, vec![0, 1, 1]);
        assert_eq!(vals, vec![4.0, 2.0, 4.0]);
    }

    #[test]
    fn test_coo_add_empty_a() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device);
        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        assert_eq!(c.nnz(), 2);
    }

    #[test]
    fn test_coo_add_empty_b() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device);

        let c = a.add(&b).unwrap();

        assert_eq!(c.nnz(), 2);
    }

    #[test]
    fn test_coo_add_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 3], DType::F32, &device);
        let b = CooData::<CpuRuntime>::empty([3, 2], DType::F32, &device);

        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_add_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device);
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F64, &device);

        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_add_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::from_slices(&[0i64], &[0i64], &[1.5f64], [2, 2], &device)
            .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(&[0i64], &[0i64], &[2.5f64], [2, 2], &device)
            .unwrap();

        let c = a.add(&b).unwrap();

        assert_eq!(c.dtype(), DType::F64);
        let vals: Vec<f64> = c.values().to_vec();
        assert_eq!(vals, vec![4.0]);
    }

    // =========================================================================
    // Element-wise mul tests
    // =========================================================================

    #[test]
    fn test_coo_mul_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        // C = A .* B:
        // [8, 0]    (2*4=8 at (0,0), 3*0=0, 0*6=0)
        // [0, 35]   (0*6=0, 5*7=35)
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.nnz(), 2); // Only positions where both have values

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 1]);
        assert_eq!(cols, vec![0, 1]);
        assert_eq!(vals, vec![8.0, 35.0]);
    }

    #[test]
    fn test_coo_mul_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // Completely disjoint - no overlapping positions
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_coo_mul_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device);

        // Multiply with empty matrix gives empty result
        let c = a.mul(&b).unwrap();
        assert_eq!(c.nnz(), 0);

        let c2 = b.mul(&a).unwrap();
        assert_eq!(c2.nnz(), 0);
    }

    #[test]
    fn test_coo_mul_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 3], DType::F32, &device);
        let b = CooData::<CpuRuntime>::empty([3, 2], DType::F32, &device);

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_mul_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device);
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F64, &device);

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_mul_same_positions() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Both have values at exactly the same positions
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[5.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert_eq!(c.nnz(), 3);

        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![10.0, 18.0, 28.0]); // 2*5, 3*6, 4*7
    }

    // =========================================================================
    // Element-wise div tests
    // =========================================================================

    #[test]
    fn test_coo_div_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [8, 0]     [2, 5]
        // [0, 35]    [7, 7]
        // Overlapping at (0,0) and (1,1)
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 0, 1, 1],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Only positions where both have values: (0,0) and (1,1)
        // 8/2=4, 35/7=5
        assert_eq!(c.nnz(), 2);

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 1]);
        assert_eq!(cols, vec![0, 1]);
        assert_eq!(vals, vec![4.0, 5.0]);
    }

    #[test]
    fn test_coo_div_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // Completely disjoint - no overlapping positions
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_coo_div_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device);

        // Divide with empty matrix gives empty result
        let c = a.div(&b).unwrap();
        assert_eq!(c.nnz(), 0);

        let c2 = b.div(&a).unwrap();
        assert_eq!(c2.nnz(), 0);
    }

    #[test]
    fn test_coo_div_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 3], DType::F32, &device);
        let b = CooData::<CpuRuntime>::empty([3, 2], DType::F32, &device);

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_div_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device);
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F64, &device);

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_div_same_positions() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Both have values at exactly the same positions
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[10.0f32, 18.0, 28.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[2.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 3);

        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![5.0, 3.0, 4.0]); // 10/2, 18/6, 28/7
    }

    // =========================================================================
    // Scalar operation tests
    // =========================================================================

    #[test]
    fn test_coo_scalar_mul() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[5.0f32, 3.0, 7.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(2.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert_eq!(result.shape(), [3, 3]);

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![10.0, 6.0, 14.0]);
    }

    #[test]
    fn test_coo_scalar_mul_zero() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(0.0).unwrap();

        assert_eq!(result.nnz(), 2);
        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![0.0, 0.0]);
    }

    #[test]
    fn test_coo_scalar_mul_negative() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, -3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(-2.0).unwrap();

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![-10.0, 6.0]);
    }

    #[test]
    fn test_coo_scalar_add() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[5.0f32, 3.0, 7.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = coo.scalar_add(1.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert_eq!(result.shape(), [3, 3]);

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![6.0, 4.0, 8.0]);
    }

    #[test]
    fn test_coo_scalar_add_negative() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_add(-2.0).unwrap();

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![3.0, 1.0]);
    }

    #[test]
    fn test_coo_scalar_add_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::empty([3, 3], DType::F32, &device);

        let result = coo.scalar_add(5.0).unwrap();

        assert_eq!(result.nnz(), 0); // Empty stays empty
    }

    #[test]
    fn test_coo_scalar_mul_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[2.5f64, 3.5],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(2.0).unwrap();

        let vals: Vec<f64> = result.values().to_vec();
        assert_eq!(vals, vec![5.0, 7.0]);
    }
}
