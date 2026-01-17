//! SparseTensor element-wise operations: add, sub, mul, div, scalar_mul, scalar_add

use super::SparseTensor;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::runtime::Runtime;
use crate::sparse::SparseOps;

impl<R: Runtime> SparseTensor<R> {
    /// Element-wise addition: C = A + B
    ///
    /// Computes the sum of two sparse tensors with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another sparse tensor with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new sparse tensor containing the element-wise sum
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Format Handling
    ///
    /// - Same format: Uses native add implementation
    /// - Different formats: Converts to COO, adds, returns COO
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A:          B:          C = A + B:
    /// // [1, 0]      [0, 2]      [1, 2]
    /// // [0, 3]  +   [4, 0]  =   [4, 3]
    /// let c = a.add(&b)?;
    /// ```
    pub fn add(&self, other: &SparseTensor<R>) -> Result<SparseTensor<R>>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape()[0], self.shape()[1]],
                got: vec![other.shape()[0], other.shape()[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        // If same format, use native add
        match (self, other) {
            (SparseTensor::Coo(a), SparseTensor::Coo(b)) => Ok(SparseTensor::Coo(a.add(b)?)),
            (SparseTensor::Csr(a), SparseTensor::Csr(b)) => Ok(SparseTensor::Csr(a.add(b)?)),
            (SparseTensor::Csc(a), SparseTensor::Csc(b)) => Ok(SparseTensor::Csc(a.add(b)?)),
            // Different formats: convert to COO and add
            _ => {
                let coo_a = self.to_coo()?;
                let coo_b = other.to_coo()?;
                let coo_a_data = coo_a.as_coo().unwrap();
                let coo_b_data = coo_b.as_coo().unwrap();
                Ok(SparseTensor::Coo(coo_a_data.add(coo_b_data)?))
            }
        }
    }

    /// Element-wise subtraction: C = A - B
    ///
    /// Computes the difference of two sparse tensors with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another sparse tensor with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new sparse tensor containing the element-wise difference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Format Handling
    ///
    /// - Same format: Uses native sub implementation
    /// - Different formats: Converts to COO, subtracts, returns COO
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A:          B:          C = A - B:
    /// // [5, 0]      [2, 3]      [3, -3]
    /// // [0, 8]  -   [4, 0]  =   [-4, 8]
    /// let c = a.sub(&b)?;
    /// ```
    pub fn sub(&self, other: &SparseTensor<R>) -> Result<SparseTensor<R>>
    where
        R::Client: SparseOps<R> + ScalarOps<R>,
    {
        // Validate shapes match
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape()[0], self.shape()[1]],
                got: vec![other.shape()[0], other.shape()[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        // If same format, use native sub
        match (self, other) {
            (SparseTensor::Coo(a), SparseTensor::Coo(b)) => Ok(SparseTensor::Coo(a.sub(b)?)),
            (SparseTensor::Csr(a), SparseTensor::Csr(b)) => Ok(SparseTensor::Csr(a.sub(b)?)),
            (SparseTensor::Csc(a), SparseTensor::Csc(b)) => Ok(SparseTensor::Csc(a.sub(b)?)),
            // Different formats: convert to COO and subtract
            _ => {
                let coo_a = self.to_coo()?;
                let coo_b = other.to_coo()?;
                let coo_a_data = coo_a.as_coo().unwrap();
                let coo_b_data = coo_b.as_coo().unwrap();
                Ok(SparseTensor::Coo(coo_a_data.sub(coo_b_data)?))
            }
        }
    }

    /// Element-wise multiplication (Hadamard product): C = A .* B
    ///
    /// Computes the element-wise product of two sparse tensors with the same shape.
    /// Only positions where BOTH tensors have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another sparse tensor with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new sparse tensor containing the element-wise product
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Format Handling
    ///
    /// - Same format: Uses native mul implementation
    /// - Different formats: Converts to COO, multiplies, returns COO
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A:          B:          C = A .* B:
    /// // [2, 3]      [4, 0]      [8, 0]
    /// // [0, 5]  .*  [6, 7]  =   [0, 35]
    /// let c = a.mul(&b)?;
    /// ```
    pub fn mul(&self, other: &SparseTensor<R>) -> Result<SparseTensor<R>>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape()[0], self.shape()[1]],
                got: vec![other.shape()[0], other.shape()[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        // If same format, use native mul
        match (self, other) {
            (SparseTensor::Coo(a), SparseTensor::Coo(b)) => Ok(SparseTensor::Coo(a.mul(b)?)),
            (SparseTensor::Csr(a), SparseTensor::Csr(b)) => Ok(SparseTensor::Csr(a.mul(b)?)),
            (SparseTensor::Csc(a), SparseTensor::Csc(b)) => Ok(SparseTensor::Csc(a.mul(b)?)),
            // Different formats: convert to COO and multiply
            _ => {
                let coo_a = self.to_coo()?;
                let coo_b = other.to_coo()?;
                let coo_a_data = coo_a.as_coo().unwrap();
                let coo_b_data = coo_b.as_coo().unwrap();
                Ok(SparseTensor::Coo(coo_a_data.mul(coo_b_data)?))
            }
        }
    }

    /// Element-wise division: C = A ./ B
    ///
    /// Computes the element-wise quotient of two sparse tensors with the same shape.
    /// Only positions where BOTH tensors have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another sparse tensor with the same shape and dtype (divisor)
    ///
    /// # Returns
    ///
    /// A new sparse tensor containing the element-wise quotient
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Format Handling
    ///
    /// - Same format: Uses native div implementation
    /// - Different formats: Converts to COO, divides, returns COO
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A:          B:          C = A ./ B:
    /// // [8, 3]      [4, 0]      [2, 0]
    /// // [0, 10]  ./ [6, 2]  =   [0, 5]
    /// let c = a.div(&b)?;
    /// ```
    pub fn div(&self, other: &SparseTensor<R>) -> Result<SparseTensor<R>>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape()[0], self.shape()[1]],
                got: vec![other.shape()[0], other.shape()[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        // If same format, use native div
        match (self, other) {
            (SparseTensor::Coo(a), SparseTensor::Coo(b)) => Ok(SparseTensor::Coo(a.div(b)?)),
            (SparseTensor::Csr(a), SparseTensor::Csr(b)) => Ok(SparseTensor::Csr(a.div(b)?)),
            (SparseTensor::Csc(a), SparseTensor::Csc(b)) => Ok(SparseTensor::Csc(a.div(b)?)),
            // Different formats: convert to COO and divide
            _ => {
                let coo_a = self.to_coo()?;
                let coo_b = other.to_coo()?;
                let coo_a_data = coo_a.as_coo().unwrap();
                let coo_b_data = coo_b.as_coo().unwrap();
                Ok(SparseTensor::Coo(coo_a_data.div(coo_b_data)?))
            }
        }
    }

    /// Scalar multiplication: C = A * scalar
    ///
    /// Multiplies all non-zero values by a scalar constant.
    /// Preserves the sparsity pattern and format.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply with
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = sparse.scalar_mul(2.0)?;  // Multiply all values by 2
    /// ```
    pub fn scalar_mul(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        match self {
            SparseTensor::Coo(d) => Ok(SparseTensor::Coo(d.scalar_mul(scalar)?)),
            SparseTensor::Csr(d) => Ok(SparseTensor::Csr(d.scalar_mul(scalar)?)),
            SparseTensor::Csc(d) => Ok(SparseTensor::Csc(d.scalar_mul(scalar)?)),
        }
    }

    /// Scalar addition: C = A + scalar (on non-zeros only)
    ///
    /// **Warning**: This adds the scalar ONLY to non-zero elements, preserving sparsity.
    /// This is NOT standard mathematical scalar addition (which would densify the matrix).
    ///
    /// Standard scalar addition would add to ALL positions (including implicit zeros),
    /// making the result dense. This operation is a practical sparse-preserving variant.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to add
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = sparse.scalar_add(1.0)?;  // Add 1 to all non-zero values
    /// ```
    pub fn scalar_add(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        match self {
            SparseTensor::Coo(d) => Ok(SparseTensor::Coo(d.scalar_add(scalar)?)),
            SparseTensor::Csr(d) => Ok(SparseTensor::Csr(d.scalar_add(scalar)?)),
            SparseTensor::Csc(d) => Ok(SparseTensor::Csc(d.scalar_add(scalar)?)),
        }
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
    fn test_add_coo_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        assert!(c.is_coo());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 2.0, 4.0, 3.0]);
    }

    #[test]
    fn test_add_csr_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        assert!(c.is_csr());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 2.0, 4.0, 3.0]);
    }

    #[test]
    fn test_add_csc_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[4.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        assert!(c.is_csc());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 2.0, 4.0, 3.0]);
    }

    #[test]
    fn test_add_mixed_formats() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A (COO):   B (CSR):
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        assert!(a.is_coo());
        assert!(b.is_csr());

        let c = a.add(&b).unwrap();

        // Mixed formats convert to COO
        assert!(c.is_coo());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 2.0, 4.0, 3.0]);
    }

    #[test]
    fn test_add_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 2]     [3, 0]
        // [0, 0]     [0, 4]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
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

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 3], DType::F32, SparseFormat::Csr, &device);
        let b = SparseTensor::<CpuRuntime>::empty([3, 2], DType::F32, SparseFormat::Csr, &device);

        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F32, SparseFormat::Csr, &device);
        let b = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F64, SparseFormat::Csr, &device);

        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Create sparse matrices from dense
        let dense_a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 3.0], &[2, 2], &device);
        let dense_b = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 2.0, 4.0, 0.0], &[2, 2], &device);

        let a = SparseTensor::from_dense(&dense_a, 1e-10).unwrap();
        let b = SparseTensor::from_dense(&dense_b, 1e-10).unwrap();

        let c = a.add(&b).unwrap();

        let dense_c = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense_c.to_vec();
        assert_eq!(data, vec![1.0, 2.0, 4.0, 3.0]);
    }

    // =========================================================================
    // Element-wise sub tests
    // =========================================================================

    #[test]
    fn test_sub_coo_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        assert!(c.is_coo());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_csr_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        assert!(c.is_csr());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_csc_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 4.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        assert!(c.is_csc());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_mixed_formats() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A (COO):   B (CSR):
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        assert!(a.is_coo());
        assert!(b.is_csr());

        let c = a.sub(&b).unwrap();

        // Mixed formats convert to COO
        assert!(c.is_coo());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 2]     [3, 0]
        // [0, 0]     [0, 4]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 2],
            &[0i64, 1],
            &[5.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
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

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 2.0, 0.0, -4.0]);
    }

    #[test]
    fn test_sub_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 3], DType::F32, SparseFormat::Csr, &device);
        let b = SparseTensor::<CpuRuntime>::empty([3, 2], DType::F32, SparseFormat::Csr, &device);

        let result = a.sub(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F32, SparseFormat::Csr, &device);
        let b = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F64, SparseFormat::Csr, &device);

        let result = a.sub(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Create sparse matrices from dense
        let dense_a = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 0.0, 0.0, 8.0], &[2, 2], &device);
        let dense_b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0, 0.0], &[2, 2], &device);

        let a = SparseTensor::from_dense(&dense_a, 1e-10).unwrap();
        let b = SparseTensor::from_dense(&dense_b, 1e-10).unwrap();

        let c = a.sub(&b).unwrap();

        let dense_c = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense_c.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A - A should be all zeros
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[1.0f32, 2.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&a).unwrap();

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add_then_sub() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // (A + B) - B should equal A
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let sum = a.add(&b).unwrap();
        let result = sum.sub(&b).unwrap();

        let dense_result = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense_result.to_vec();
        assert_eq!(data, vec![5.0, 0.0, 0.0, 8.0]);
    }

    // =========================================================================
    // Element-wise mul tests
    // =========================================================================

    #[test]
    fn test_mul_coo_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert!(c.is_coo());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_csr_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 3],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert!(c.is_csr());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_csc_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        // CSC for A: col_ptrs=[0,1,3], row_indices=[0,0,1], values=[2,3,5]
        // CSC for B: col_ptrs=[0,2,3], row_indices=[0,1,1], values=[4,6,7]
        let a = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 3],
            &[0i64, 0, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert!(c.is_csc());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_mixed_formats() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A (COO):   B (CSR):
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 3],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        assert!(a.is_coo());
        assert!(b.is_csr());

        let c = a.mul(&b).unwrap();

        // Mixed formats convert to COO
        assert!(c.is_coo());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // Completely disjoint positions
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 3], DType::F32, SparseFormat::Csr, &device);
        let b = SparseTensor::<CpuRuntime>::empty([3, 2], DType::F32, SparseFormat::Csr, &device);

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F32, SparseFormat::Csr, &device);
        let b = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F64, SparseFormat::Csr, &device);

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Create sparse matrices from dense
        let dense_a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 0.0, 5.0], &[2, 2], &device);
        let dense_b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 0.0, 6.0, 7.0], &[2, 2], &device);

        let a = SparseTensor::from_dense(&dense_a, 1e-10).unwrap();
        let b = SparseTensor::from_dense(&dense_b, 1e-10).unwrap();

        let c = a.mul(&b).unwrap();

        let dense_c = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense_c.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A .* A = A^2 (element-wise)
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[3.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&a).unwrap();

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![9.0, 0.0, 0.0, 16.0]);
    }

    #[test]
    fn test_mul_identity_sparse() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Multiplying by a sparse "all ones" matrix at same positions = same matrix
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let ones = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[1.0f32, 1.0, 1.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&ones).unwrap();

        assert_eq!(c.nnz(), 3);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 3.0, 0.0, 5.0]);
    }

    // =========================================================================
    // Element-wise div tests
    // =========================================================================

    #[test]
    fn test_div_coo_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [8, 0]     [2, 5]
        // [0, 35]    [7, 7]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1, 1],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Only overlapping positions: (0,0) and (1,1)
        // 8/2=4, 35/7=5
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_csr_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 4],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_csc_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // CSC format: col_ptrs point to column starts in row_indices
        let a = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 4],
            &[0i64, 1, 0, 1],
            &[2.0f32, 7.0, 5.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_mixed_formats() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // CSR + COO -> COO
        let csr = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let coo = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1, 1],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = csr.div(&coo).unwrap();

        assert!(matches!(c, SparseTensor::Coo(_)));
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // No overlapping positions
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_div_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f32],
            [2, 3],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f32],
            [3, 2],
            &device,
        )
        .unwrap();

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f32],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f64],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A = [10, 0; 0, 20]
        // B = [2, 0; 5, 4]
        let dense_a =
            Tensor::<CpuRuntime>::from_slice(&[10.0f32, 0.0, 0.0, 20.0], &[2, 2], &device);
        let dense_b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.0, 5.0, 4.0], &[2, 2], &device);

        let sparse_a = SparseTensor::from_dense(&dense_a, 1e-10).unwrap();
        let sparse_b = SparseTensor::from_dense(&dense_b, 1e-10).unwrap();

        let c = sparse_a.div(&sparse_b).unwrap();

        // Only (0,0) overlaps: 10/2 = 5
        // (1,1) in A but (1,0) and (1,1) in B - only (1,1) overlaps: 20/4 = 5
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![5.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        // A / A = all ones at non-zero positions
        let c = a.div(&a).unwrap();

        assert_eq!(c.nnz(), 3);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_div_by_ones() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A / 1 = A (where ones has same sparsity pattern)
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let ones = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[1.0f32, 1.0, 1.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&ones).unwrap();

        assert_eq!(c.nnz(), 3);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 3.0, 0.0, 5.0]);
    }

    // =========================================================================
    // Scalar operation tests
    // =========================================================================

    #[test]
    fn test_sparse_tensor_scalar_mul_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[5.0f32, 3.0, 7.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(2.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert!(result.is_coo());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 10.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 14.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_mul_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(3.0).unwrap();

        assert_eq!(result.nnz(), 5);
        assert!(result.is_csr());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, 0.0, 6.0, 0.0, 0.0, 9.0, 12.0, 15.0, 0.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_mul_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 4.0, 5.0, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(2.0).unwrap();

        assert_eq!(result.nnz(), 5);
        assert!(result.is_csc());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 0.0, 4.0, 0.0, 0.0, 6.0, 8.0, 10.0, 0.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_add_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_add(1.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert!(result.is_coo());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, 0.0, 4.0, 6.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_mul_zero() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(0.0).unwrap();

        assert_eq!(result.nnz(), 2); // Still has 2 entries, but values are 0
        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_add_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse =
            SparseTensor::<CpuRuntime>::empty([3, 3], DType::F32, SparseFormat::Coo, &device);

        let result = sparse.scalar_add(5.0).unwrap();

        assert_eq!(result.nnz(), 0); // Empty stays empty
    }

    #[test]
    fn test_sparse_tensor_scalar_ops_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[2.5f64, 3.5],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(2.0).unwrap();

        assert_eq!(result.dtype(), DType::F64);
        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f64> = dense.to_vec();
        assert_eq!(data, vec![5.0, 0.0, 0.0, 7.0]);
    }
}
