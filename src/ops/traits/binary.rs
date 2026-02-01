//! Binary operations trait.
//!
//! This trait defines element-wise binary operations on tensors.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Element-wise binary operations on tensors.
///
/// This trait defines operations that take two input tensors and produce one output tensor.
/// All binary operations support broadcasting.
///
/// # Broadcasting
///
/// Binary operations follow NumPy-style broadcasting rules:
/// - Dimensions are compared element-wise, from the trailing dimensions backward
/// - Two dimensions are compatible when they are equal, or when one of them is 1
/// - Dimensions of size 1 are stretched to match the other dimension
/// - The output has shape equal to the pairwise maximum of the input shapes
///
/// # Example
///
/// ```ignore
/// use numr::prelude::*;
///
/// let device = CpuDevice::new();
/// let client = CpuRuntime::default_client(&device);
///
/// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
/// let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);
///
/// let c = client.add(&a, &b)?;  // [6.0, 8.0, 10.0, 12.0]
/// ```
pub trait BinaryOps<R: Runtime> {
    /// Element-wise addition: a + b
    ///
    /// Adds two tensors element-wise, supporting broadcasting.
    ///
    /// # Arguments
    /// * `a` - Left operand
    /// * `b` - Right operand (shape must be broadcastable with `a`)
    ///
    /// # Returns
    /// A new tensor with the result of the addition.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.add(&tensor_a, &tensor_b)?;
    /// ```
    fn add(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise subtraction: a - b
    ///
    /// Subtracts two tensors element-wise, supporting broadcasting.
    ///
    /// # Arguments
    /// * `a` - Left operand (minuend)
    /// * `b` - Right operand (subtrahend, shape must be broadcastable with `a`)
    ///
    /// # Returns
    /// A new tensor with the result of the subtraction.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.sub(&tensor_a, &tensor_b)?;
    /// ```
    fn sub(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise multiplication: a * b
    ///
    /// Multiplies two tensors element-wise, supporting broadcasting.
    ///
    /// # Arguments
    /// * `a` - Left operand
    /// * `b` - Right operand (shape must be broadcastable with `a`)
    ///
    /// # Returns
    /// A new tensor with the result of the multiplication.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.mul(&tensor_a, &tensor_b)?;
    /// ```
    fn mul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise division: a / b
    ///
    /// Divides two tensors element-wise, supporting broadcasting.
    /// Division by zero is undefined behavior (implementation-dependent).
    ///
    /// # Arguments
    /// * `a` - Left operand (dividend/numerator)
    /// * `b` - Right operand (divisor/denominator, shape must be broadcastable with `a`)
    ///
    /// # Returns
    /// A new tensor with the result of the division.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.div(&tensor_a, &tensor_b)?;
    /// ```
    fn div(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise power: a^b
    ///
    /// Raises the elements of the first tensor to the power of the elements
    /// of the second tensor, element-wise, supporting broadcasting.
    ///
    /// # Arguments
    /// * `a` - Base tensor
    /// * `b` - Exponent tensor (shape must be broadcastable with `a`)
    ///
    /// # Returns
    /// A new tensor with the result of the power operation.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.pow(&base, &exponent)?;
    /// ```
    fn pow(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise maximum: max(a, b)
    ///
    /// Computes the element-wise maximum of two tensors, supporting broadcasting.
    ///
    /// # Arguments
    /// * `a` - First tensor
    /// * `b` - Second tensor (shape must be broadcastable with `a`)
    ///
    /// # Returns
    /// A new tensor containing the maximum of corresponding elements.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.maximum(&tensor_a, &tensor_b)?;
    /// ```
    fn maximum(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise minimum: min(a, b)
    ///
    /// Computes the element-wise minimum of two tensors, supporting broadcasting.
    ///
    /// # Arguments
    /// * `a` - First tensor
    /// * `b` - Second tensor (shape must be broadcastable with `a`)
    ///
    /// # Returns
    /// A new tensor containing the minimum of corresponding elements.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.minimum(&tensor_a, &tensor_b)?;
    /// ```
    fn minimum(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Two-argument arctangent: atan2(y, x)
    ///
    /// Computes the angle in radians between the positive x-axis and the point (x, y),
    /// element-wise, supporting broadcasting.
    ///
    /// The result is in the range [-π, π]. This function is essential for converting
    /// Cartesian coordinates to polar coordinates and for spatial algorithms.
    ///
    /// # Arguments
    /// * `y` - Y-coordinate tensor
    /// * `x` - X-coordinate tensor (shape must be broadcastable with `y`)
    ///
    /// # Returns
    /// A new tensor with the angle in radians for each (y, x) pair.
    ///
    /// # Errors
    /// Returns an error if shapes are not broadcastable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let angles = client.atan2(&y_coords, &x_coords)?;
    /// ```
    fn atan2(&self, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;
}
