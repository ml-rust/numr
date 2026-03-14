//! FP8 matrix multiplication operations trait.
//!
//! FP8 matmul differs from standard matmul in two key ways:
//! 1. Per-tensor scale factors compensate for the limited dynamic range of FP8
//! 2. Accumulation is always in FP32 for numerical accuracy
//!
//! The output dtype can differ from input dtype (typically F32, F16, or BF16).

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// FP8 matrix multiplication operations with per-tensor scaling.
///
/// FP8 GEMM computes: `output = (scale_a * A) @ (scale_b * B)` where A and B are
/// FP8 tensors, arithmetic is performed in FP32, and the result is cast to `out_dtype`.
///
/// # Scale Factors
///
/// FP8 has very limited dynamic range (~[-448, 448] for E4M3, ~[-57344, 57344] for E5M2).
/// Per-tensor scale factors map the original tensor range into the FP8 representable range:
///
/// ```text
/// quantize:   fp8_tensor = original_tensor / scale
/// dequantize: original_tensor = fp8_tensor * scale
/// matmul:     C = (A * scale_a) @ (B * scale_b) = scale_a * scale_b * (A_fp8 @ B_fp8)
/// ```
///
/// # Use Cases
///
/// - `fp8_matmul`: E4M3 x E4M3 — forward pass (weights and activations)
/// - `fp8_matmul_e5m2`: E5M2 x E4M3 — backward pass (gradients x weights)
pub trait Fp8MatmulOps<R: Runtime> {
    /// FP8 E4M3 x E4M3 matrix multiplication with per-tensor scaling.
    ///
    /// Computes: `output = scale_a * scale_b * (a_e4m3 @ b_e4m3)`
    /// with FP32 accumulation, then casts to `out_dtype`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of shape `[..., M, K]` with dtype FP8E4M3
    /// * `b` - Weight tensor of shape `[..., K, N]` with dtype FP8E4M3
    /// * `scale_a` - Per-tensor scale factor for A (scalar f32)
    /// * `scale_b` - Per-tensor scale factor for B (scalar f32)
    /// * `out_dtype` - Output dtype (F32, F16, or BF16)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., M, N]` with dtype `out_dtype`.
    ///
    /// # Errors
    ///
    /// - `DTypeMismatch` if inputs are not FP8E4M3
    /// - `ShapeMismatch` if inner dimensions don't match
    /// - `UnsupportedDType` if `out_dtype` is not F32/F16/BF16
    fn fp8_matmul(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<R>>;

    /// FP8 E5M2 x E4M3 matrix multiplication with per-tensor scaling.
    ///
    /// Used for backward pass: gradients (E5M2, larger range) x weights (E4M3, higher precision).
    ///
    /// Computes: `output = scale_a * scale_b * (a_e5m2 @ b_e4m3)`
    /// with FP32 accumulation, then casts to `out_dtype`.
    ///
    /// # Arguments
    ///
    /// * `a` - Gradient tensor of shape `[..., M, K]` with dtype FP8E5M2
    /// * `b` - Weight tensor of shape `[..., K, N]` with dtype FP8E4M3
    /// * `scale_a` - Per-tensor scale factor for A (scalar f32)
    /// * `scale_b` - Per-tensor scale factor for B (scalar f32)
    /// * `out_dtype` - Output dtype (F32, F16, or BF16)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., M, N]` with dtype `out_dtype`.
    fn fp8_matmul_e5m2(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<R>>;
}
