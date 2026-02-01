//! Utility operations trait.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Utility operations
pub trait UtilityOps<R: Runtime> {
    /// Clamp tensor values to a range: clamp(x, min, max) = min(max(x, min), max)
    ///
    /// Element-wise clamps each value to be within [min_val, max_val].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `min_val` - Minimum value (inclusive)
    /// * `max_val` - Maximum value (inclusive)
    ///
    /// # Returns
    ///
    /// Tensor with same shape and dtype as input, with values clamped to range
    fn clamp(&self, a: &Tensor<R>, min_val: f64, max_val: f64) -> Result<Tensor<R>>;

    /// Fill tensor with a constant value
    ///
    /// Creates a new tensor with the specified shape and dtype, filled with the given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `value` - Value to fill the tensor with
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// New tensor filled with the constant value
    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<R>>;

    /// Create a 1D tensor with evenly spaced values within a half-open interval [start, stop)
    ///
    /// Values are generated using the formula: start + step * i for i in 0..n
    /// where n = ceil((stop - start) / step)
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the interval (inclusive)
    /// * `stop` - End of the interval (exclusive)
    /// * `step` - Spacing between values (must be positive if start < stop, negative if start > stop)
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// 1D tensor with evenly spaced values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = client.arange(0.0, 5.0, 1.0, DType::F32)?; // [0, 1, 2, 3, 4]
    /// let t = client.arange(0.0, 5.0, 2.0, DType::F32)?; // [0, 2, 4]
    /// let t = client.arange(5.0, 0.0, -1.0, DType::F32)?; // [5, 4, 3, 2, 1]
    /// ```
    fn arange(&self, start: f64, stop: f64, step: f64, dtype: DType) -> Result<Tensor<R>>;

    /// Create a 1D tensor with evenly spaced values over a specified interval
    ///
    /// Returns `steps` evenly spaced values from `start` to `stop` (inclusive).
    /// Values are: start + (stop - start) * i / (steps - 1) for i in 0..steps
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the interval
    /// * `stop` - End of the interval (inclusive)
    /// * `steps` - Number of values to generate (must be >= 2)
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// 1D tensor with evenly spaced values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = client.linspace(0.0, 10.0, 5, DType::F32)?; // [0, 2.5, 5, 7.5, 10]
    /// let t = client.linspace(0.0, 1.0, 3, DType::F64)?; // [0, 0.5, 1]
    /// ```
    fn linspace(&self, start: f64, stop: f64, steps: usize, dtype: DType) -> Result<Tensor<R>>;

    /// Create a 2D identity matrix (or batch of identity matrices)
    ///
    /// Creates a tensor where the diagonal elements are 1 and all others are 0.
    /// For rectangular matrices, the diagonal is the main diagonal.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows
    /// * `m` - Number of columns (if None, defaults to n for square matrix)
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// 2D tensor of shape [n, m] with ones on the diagonal
    ///
    /// # Example
    ///
    /// ```ignore
    /// let eye = client.eye(3, None, DType::F32)?;    // 3x3 identity matrix
    /// let rect = client.eye(2, Some(4), DType::F32)?; // 2x4 matrix with diagonal ones
    /// ```
    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<R>>;
}
