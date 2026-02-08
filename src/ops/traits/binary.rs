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
/// ```
/// use numr::prelude::*;
///
/// let device = CpuDevice::new();
/// let client = CpuRuntime::default_client(&device);
///
/// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
/// let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);
///
/// let c = client.add(&a, &b)?;  // [6.0, 8.0, 10.0, 12.0]
/// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    /// let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);
    /// let result = client.add(&a, &b)?;
    /// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 8.0], &[2], &device);
    /// let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0], &[2], &device);
    /// let result = client.sub(&a, &b)?;
    /// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device);
    /// let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device);
    /// let result = client.mul(&a, &b)?;
    /// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 9.0], &[2], &device);
    /// let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device);
    /// let result = client.div(&a, &b)?;
    /// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let base = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device);
    /// let exponent = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 2.0], &[2], &device);
    /// let result = client.pow(&base, &exponent)?;
    /// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0], &[2], &device);
    /// let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 2.0], &[2], &device);
    /// let result = client.maximum(&a, &b)?;
    /// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0], &[2], &device);
    /// let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 2.0], &[2], &device);
    /// let result = client.minimum(&a, &b)?;
    /// # Ok::<(), numr::error::Error>(())
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
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2], &device);
    /// let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device);
    /// let angles = client.atan2(&y, &x)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn atan2(&self, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;
}
