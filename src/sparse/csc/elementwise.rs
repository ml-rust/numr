//! CSC element-wise operations: add, sub, mul, div, scalar_mul, scalar_add
//!
//! All element-wise operations dispatch to runtime-specific implementations
//! via the SparseOps trait, enabling GPU acceleration when available.

use super::CscData;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseStorage};
use crate::tensor::Tensor;

impl<R: Runtime> CscData<R> {
    /// Element-wise addition: C = A + B
    ///
    /// Computes the sum of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSC matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSC matrix containing the element-wise sum
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Column-by-column merge of sorted row indices using union semantics.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a + nnz_b) sequential merge
    /// - GPU: O(nnz_a + nnz_b) parallel per-column merge
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
            let (out_col_ptrs, out_row_indices, out_values) = client.add_csc::<T>(
                &self.col_ptrs,
                &self.row_indices,
                &self.values,
                &other.col_ptrs,
                &other.row_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                col_ptrs: out_col_ptrs,
                row_indices: out_row_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csc_add")
    }

    /// Element-wise subtraction: C = A - B
    ///
    /// Computes the difference of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSC matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSC matrix containing the element-wise difference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Column-by-column merge of sorted row indices using union semantics.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a + nnz_b) sequential merge
    /// - GPU: O(nnz_a + nnz_b) parallel per-column merge
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A:          B:          C = A - B:
    /// // [5, 0]      [2, 3]      [3, -3]
    /// // [0, 8]  -   [4, 0]  =   [-4, 8]
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
            let (out_col_ptrs, out_row_indices, out_values) = client.sub_csc::<T>(
                &self.col_ptrs,
                &self.row_indices,
                &self.values,
                &other.col_ptrs,
                &other.row_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                col_ptrs: out_col_ptrs,
                row_indices: out_row_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csc_sub")
    }

    /// Element-wise multiplication (Hadamard product): C = A .* B
    ///
    /// Computes the element-wise product of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSC matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSC matrix containing the element-wise product
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Column-by-column intersection of sorted row indices using intersection semantics.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a + nnz_b) sequential merge
    /// - GPU: O(nnz_a + nnz_b) parallel per-column merge
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
            let (out_col_ptrs, out_row_indices, out_values) = client.mul_csc::<T>(
                &self.col_ptrs,
                &self.row_indices,
                &self.values,
                &other.col_ptrs,
                &other.row_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                col_ptrs: out_col_ptrs,
                row_indices: out_row_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csc_mul")
    }

    /// Element-wise division: C = A ./ B
    ///
    /// Computes the element-wise quotient of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSC matrix with the same shape and dtype (divisor)
    ///
    /// # Returns
    ///
    /// A new CSC matrix containing the element-wise quotient
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Column-by-column intersection of sorted row indices, dividing values
    /// at matching positions.
    ///
    /// # Performance
    ///
    /// O(nnz_a + nnz_b) - linear merge since rows are sorted within columns.
    /// Result has at most min(nnz_a, nnz_b) non-zeros.
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
        let [_nrows, ncols] = self.shape;

        // Handle empty cases - if either is empty, result is empty
        if self.nnz() == 0 || other.nnz() == 0 {
            return Ok(Self::empty(self.shape, dtype, device));
        }

        // Read CSC data from both matrices
        let col_ptrs_a: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices_a: Vec<i64> = self.row_indices.to_vec();
        let col_ptrs_b: Vec<i64> = other.col_ptrs.to_vec();
        let row_indices_b: Vec<i64> = other.row_indices.to_vec();

        // Dispatch on dtype to merge and divide values
        crate::dispatch_dtype!(dtype, T => {
            let vals_a: Vec<T> = self.values.to_vec();
            let vals_b: Vec<T> = other.values.to_vec();

            // Build result arrays column by column
            let mut result_col_ptrs: Vec<i64> = Vec::with_capacity(ncols + 1);
            let mut result_rows: Vec<i64> = Vec::new();
            let mut result_vals: Vec<T> = Vec::new();

            result_col_ptrs.push(0);

            for col in 0..ncols {
                let start_a = col_ptrs_a[col] as usize;
                let end_a = col_ptrs_a[col + 1] as usize;
                let start_b = col_ptrs_b[col] as usize;
                let end_b = col_ptrs_b[col + 1] as usize;

                // Intersect sorted rows for this column
                let mut i = start_a;
                let mut j = start_b;

                while i < end_a && j < end_b {
                    let row_a = row_indices_a[i];
                    let row_b = row_indices_b[j];

                    if row_a < row_b {
                        // A only - skip (result is 0)
                        i += 1;
                    } else if row_b < row_a {
                        // B only - skip (result is 0)
                        j += 1;
                    } else {
                        // Same row - divide values
                        let quotient = vals_a[i].to_f64() / vals_b[j].to_f64();
                        result_rows.push(row_a);
                        result_vals.push(T::from_f64(quotient));
                        i += 1;
                        j += 1;
                    }
                }

                result_col_ptrs.push(result_rows.len() as i64);
            }

            let col_ptrs_tensor = Tensor::from_slice(&result_col_ptrs, &[result_col_ptrs.len()], device);
            let row_indices_tensor = Tensor::from_slice(&result_rows, &[result_rows.len()], device);
            let values_tensor = Tensor::from_slice(&result_vals, &[result_vals.len()], device);

            return Self::new(col_ptrs_tensor, row_indices_tensor, values_tensor, self.shape);
        }, "CSC element-wise div");
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
    /// let result = csc.scalar_mul(2.0)?;  // Multiply all values by 2
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
    /// ```ignore
    /// // Sparse matrix:        After scalar_add(10):
    /// // [1, 0, 2]             [11,  0, 12]
    /// // [0, 3, 0]             [ 0, 13,  0]
    /// //
    /// // Note: Zeros stay 0, not 10!
    /// let result = csc.scalar_add(10.0)?;
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
    /// ```ignore
    /// // Clearer intent than scalar_add
    /// let result = csc.add_to_nonzeros(5.0)?;
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
    fn test_csc_add_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,1], values=[1,3]
        // CSC for B: col_ptrs=[0,1,2], row_indices=[1,0], values=[4,2]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[4.0f32, 2.0],
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

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 2, 4]);
        assert_eq!(row_indices, vec![0, 1, 0, 1]);
        assert_eq!(vals, vec![1.0, 4.0, 2.0, 3.0]);
    }

    #[test]
    fn test_csc_add_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 2]     [3, 0]
        // [0, 0]     [0, 4]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,0], values=[1,2]
        // CSC for B: col_ptrs=[0,1,2], row_indices=[0,1], values=[3,4]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 0],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
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

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 1, 3]);
        assert_eq!(row_indices, vec![0, 0, 1]);
        assert_eq!(vals, vec![4.0, 2.0, 4.0]);
    }

    #[test]
    fn test_csc_add_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 2], DType::F32, &device);
        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();
        assert_eq!(c.nnz(), 2);

        let c2 = b.add(&a).unwrap();
        assert_eq!(c2.nnz(), 2);
    }

    #[test]
    fn test_csc_add_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 3], DType::F32, &device);
        let b = CscData::<CpuRuntime>::empty([3, 2], DType::F32, &device);

        let result = a.add(&b);
        assert!(result.is_err());
    }

    // =========================================================================
    // Element-wise sub tests
    // =========================================================================

    #[test]
    fn test_csc_sub_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,1], values=[5,8]
        // CSC for B: col_ptrs=[0,2,3], row_indices=[0,1,0], values=[2,4,3]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 4.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        // C = A - B:
        // [3, -3]
        // [-4, 8]
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.nnz(), 4);

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 2, 4]);
        assert_eq!(row_indices, vec![0, 1, 0, 1]);
        assert_eq!(vals, vec![3.0, -4.0, -3.0, 8.0]);
    }

    #[test]
    fn test_csc_sub_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 2]     [3, 0]
        // [0, 0]     [0, 4]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,0], values=[5,2]
        // CSC for B: col_ptrs=[0,1,2], row_indices=[0,1], values=[3,4]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 0],
            &[5.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[3.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        // C = A - B:
        // [2, 2]   (5-3=2 at (0,0))
        // [0, -4]
        assert_eq!(c.nnz(), 3);

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 1, 3]);
        assert_eq!(row_indices, vec![0, 0, 1]);
        assert_eq!(vals, vec![2.0, 2.0, -4.0]);
    }

    #[test]
    fn test_csc_sub_empty_a() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 2], DType::F32, &device);
        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();
        assert_eq!(c.nnz(), 2);

        // Result should be -B
        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![-1.0, -2.0]);
    }

    #[test]
    fn test_csc_sub_empty_b() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CscData::<CpuRuntime>::empty([2, 2], DType::F32, &device);

        let c = a.sub(&b).unwrap();
        assert_eq!(c.nnz(), 2);

        // Result should be A
        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn test_csc_sub_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 3], DType::F32, &device);
        let b = CscData::<CpuRuntime>::empty([3, 2], DType::F32, &device);

        let result = a.sub(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_csc_sub_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A - A should be all zeros (empty sparse matrix)
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[1.0f32, 2.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&a).unwrap();
        // Sparse matrices don't store explicit zeros - result should be empty
        assert_eq!(c.nnz(), 0);
    }

    // =========================================================================
    // Element-wise mul tests
    // =========================================================================

    #[test]
    fn test_csc_mul_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        // CSC for A: col_ptrs=[0,1,3], row_indices=[0,0,1], values=[2,3,5]
        // CSC for B: col_ptrs=[0,2,3], row_indices=[0,1,1], values=[4,6,7]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 3],
            &[0i64, 0, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        // C = A .* B:
        // [8, 0]    (2*4=8 at (0,0))
        // [0, 35]   (5*7=35 at (1,1))
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.nnz(), 2);

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 1, 2]);
        assert_eq!(row_indices, vec![0, 1]);
        assert_eq!(vals, vec![8.0, 35.0]);
    }

    #[test]
    fn test_csc_mul_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[4.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_csc_mul_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CscData::<CpuRuntime>::empty([2, 2], DType::F32, &device);

        let c = a.mul(&b).unwrap();
        assert_eq!(c.nnz(), 0);

        let c2 = b.mul(&a).unwrap();
        assert_eq!(c2.nnz(), 0);
    }

    #[test]
    fn test_csc_mul_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 3], DType::F32, &device);
        let b = CscData::<CpuRuntime>::empty([3, 2], DType::F32, &device);

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_csc_mul_same_positions() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Both have values at exactly the same positions
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
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
    fn test_csc_div_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [8, 0]     [2, 5]
        // [0, 35]    [7, 7]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,1], values=[8,35]
        // CSC for B: col_ptrs=[0,2,4], row_indices=[0,1,0,1], values=[2,7,5,7]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 4],
            &[0i64, 1, 0, 1],
            &[2.0f32, 7.0, 5.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Only positions where both have values: (0,0) and (1,1)
        // 8/2=4, 35/7=5
        assert_eq!(c.nnz(), 2);

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 1, 2]);
        assert_eq!(row_indices, vec![0, 1]);
        assert_eq!(vals, vec![4.0, 5.0]);
    }

    #[test]
    fn test_csc_div_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,1], values=[1,3]
        // CSC for B: col_ptrs=[0,1,2], row_indices=[1,0], values=[4,2]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[4.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_csc_div_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CscData::<CpuRuntime>::empty([2, 2], DType::F32, &device);

        // Divide with empty matrix gives empty result
        let c = a.div(&b).unwrap();
        assert_eq!(c.nnz(), 0);

        let c2 = b.div(&a).unwrap();
        assert_eq!(c2.nnz(), 0);
    }

    #[test]
    fn test_csc_div_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 3], DType::F32, &device);
        let b = CscData::<CpuRuntime>::empty([3, 2], DType::F32, &device);

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_csc_div_same_positions() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Both have values at exactly the same positions
        // Matrix layout:
        // [10, 0]    [2, 0]
        // [18, 28]   [6, 7]
        // CSC for A: col_ptrs=[0,2,3], row_indices=[0,1,1], values=[10,18,28]
        // CSC for B: col_ptrs=[0,2,3], row_indices=[0,1,1], values=[2,6,7]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[10.0f32, 18.0, 28.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
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
