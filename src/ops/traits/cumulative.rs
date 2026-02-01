//! Cumulative operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Cumulative operations
pub trait CumulativeOps<R: Runtime> {
    /// Cumulative sum along a dimension
    ///
    /// Returns the cumulative sum of elements along the specified dimension.
    /// For input [a, b, c, d], output is [a, a+b, a+b+c, a+b+c+d].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to compute cumulative sum (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input containing cumulative sums
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let result = client.cumsum(&a, 0)?; // [1, 3, 6, 10]
    /// ```
    fn cumsum(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>;

    /// Cumulative product along a dimension
    ///
    /// Returns the cumulative product of elements along the specified dimension.
    /// For input [a, b, c, d], output is [a, a*b, a*b*c, a*b*c*d].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to compute cumulative product (supports negative indexing)
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input containing cumulative products
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);
    /// let result = client.cumprod(&a, 0)?; // [1, 2, 6, 24]
    /// ```
    fn cumprod(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>;

    /// Log-sum-exp along specified dimensions (numerically stable)
    ///
    /// Computes log(sum(exp(x))) in a numerically stable way:
    /// logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    ///
    /// This is commonly used in softmax computation and log-probability calculations.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    ///
    /// # Returns
    ///
    /// Tensor containing log-sum-exp values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &device);
    /// let result = client.logsumexp(&a, &[0], false)?;
    /// // result ≈ log(exp(1) + exp(2) + exp(3)) ≈ 3.4076
    /// ```
    fn logsumexp(&self, a: &Tensor<R>, dims: &[usize], keepdim: bool) -> Result<Tensor<R>>;
}
