//! Conditional operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Conditional operations
pub trait ConditionalOps<R: Runtime> {
    /// Conditional select: where(cond, x, y) = cond ? x : y
    ///
    /// Performs element-wise conditional selection. For each position,
    /// returns x if condition is true (non-zero), otherwise y.
    ///
    /// # Arguments
    ///
    /// * `cond` - Condition tensor (any numeric dtype: 0 = false, non-zero = true)
    /// * `x` - Values to select when condition is true
    /// * `y` - Values to select when condition is false
    ///
    /// # Condition Dtype
    ///
    /// The condition tensor accepts any numeric dtype (U8, I32, F32, F64, etc.).
    /// Non-zero values are treated as true, zero as false. This allows using
    /// comparison results directly (e.g., from `eq`, `lt`, `gt`) without dtype
    /// conversion:
    ///
    /// ```ignore
    /// let mask = client.gt(&a, &threshold)?;  // Returns same dtype as a
    /// let result = client.where_cond(&mask, &x, &y)?;  // Works directly
    /// ```
    ///
    /// For optimal performance, U8 conditions use SIMD-optimized kernels on
    /// supported platforms (x86-64 with AVX2/AVX-512).
    ///
    /// # Returns
    ///
    /// Tensor with same shape and dtype as x and y
    ///
    /// # Backend Notes
    ///
    /// - CPU: Native support for all condition dtypes with SIMD optimization for U8
    /// - CUDA: Native support for F32, F64, I32, I64, U32 conditions (optimized U8)
    /// - WebGPU: Native support for F32, I32, U32 conditions with broadcasting
    fn where_cond(&self, cond: &Tensor<R>, x: &Tensor<R>, y: &Tensor<R>) -> Result<Tensor<R>>;
}
