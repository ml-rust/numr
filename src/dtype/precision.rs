//! Mixed precision configuration for intermediate calculations.

/// Compute precision for intermediate calculations with reduced-precision types.
///
/// When operating on reduced-precision types (F16, BF16, FP8), values are typically
/// converted to a higher precision format for computation, then converted back.
/// This allows trading off speed vs precision.
///
/// # Precision Comparison
///
/// | Precision | Decimal Digits | Speed   | Use Case |
/// |-----------|----------------|---------|----------|
/// | **F64**   | ~15-16         | Slowest | Scientific computing requiring maximum precision |
/// | **F32**   | ~7             | Medium  | High-precision ML, when BF16 isn't enough |
/// | **BF16**  | ~3             | Fastest | ML training/inference (default, industry standard) |
///
/// # Applicability
///
/// - **FP8**: Always needs upcasting (8-bit storage, compute in BF16, F32, or F64)
/// - **F16/BF16**: Can optionally upcast to F32/F64 for higher precision
/// - **F32**: Can upcast to F64 for scientific computing
/// - **F64**: No upcasting needed (already highest precision)
///
/// # Resolution Order
///
/// `per-operation > tensor-level > client default`
///
/// # Default
///
/// BF16 is the default, as it provides good speed with the same dynamic range as F32.
/// This is the industry standard for mixed-precision ML training.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ComputePrecision {
    /// Compute in F64 (highest precision, slowest)
    /// Use for: scientific simulations, physics, when F32 precision is insufficient
    F64,
    /// Compute in F32 (high precision, medium speed)
    /// Use for: high-precision ML, numerical algorithms sensitive to rounding
    F32,
    /// Compute in BF16 (lower precision, fastest, industry standard for ML)
    /// Use for: ML training/inference, when speed matters more than precision
    #[default]
    BF16,
}
