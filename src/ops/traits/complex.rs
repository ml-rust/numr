//! Complex number operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Complex number operations
pub trait ComplexOps<R: Runtime> {
    /// Complex conjugate: conj(a + bi) = a - bi
    ///
    /// Returns the complex conjugate of the input tensor.
    /// For real tensors, returns the input unchanged.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (Complex64, Complex128, or real types)
    ///
    /// # Returns
    ///
    /// * Complex types: Tensor with same shape and dtype, imaginary part negated
    /// * Real types: Returns input tensor unchanged (real numbers equal their conjugate)
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends (identity operation)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let conj_z = client.conj(&z)?;
    /// // Result: [1.0 - 2.0i, 3.0 + 4.0i]
    /// ```
    fn conj(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Extract real part of complex tensor: real(a + bi) = a
    ///
    /// Extracts the real component from a complex tensor.
    /// For real tensors, returns a copy of the input.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with same shape
    /// * Complex128 input → F64 tensor with same shape
    /// * Real input → Copy of input tensor
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let re = client.real(&z)?;  // F32 tensor: [1.0, 3.0]
    /// ```
    fn real(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Extract imaginary part of complex tensor: imag(a + bi) = b
    ///
    /// Extracts the imaginary component from a complex tensor.
    /// For real tensors, returns a zero tensor with the same shape.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with same shape
    /// * Complex128 input → F64 tensor with same shape
    /// * Real input → Zero tensor with same shape and dtype
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let im = client.imag(&z)?;  // F32 tensor: [2.0, 4.0]
    /// ```
    fn imag(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute phase angle of complex tensor: angle(a + bi) = atan2(b, a)
    ///
    /// Returns the phase angle (argument) of complex numbers in radians.
    /// The result is in the range [-π, π].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// * Complex64 input → F32 tensor with angles in radians
    /// * Complex128 input → F64 tensor with angles in radians
    /// * Real input → Zero tensor (real numbers have phase angle 0 for positive, π for negative)
    ///
    /// # Supported Types
    ///
    /// * Complex64: All backends (CPU, CUDA, WebGPU)
    /// * Complex128: CPU and CUDA only (WebGPU does not support F64)
    /// * Real types: All backends
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = Tensor::<CpuRuntime>::from_slice(
    ///     &[Complex64::new(1.0, 1.0), Complex64::new(-1.0, 0.0)],
    ///     &[2],
    ///     &device
    /// );
    /// let angles = client.angle(&z)?;  // F32 tensor: [π/4, π]
    /// ```
    ///
    /// # Mathematical Notes
    ///
    /// For complex z = a + bi, returns atan2(b, a) in radians [-π, π].
    /// For real x, returns 0 if x ≥ 0, π if x < 0.
    /// To compute magnitude, use abs(z) = sqrt(re² + im²) separately.
    fn angle(&self, a: &Tensor<R>) -> Result<Tensor<R>>;
}
