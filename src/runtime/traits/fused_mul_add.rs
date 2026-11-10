//! Provider guarantees for floating-point fused multiply-add operations.

/// Declared fused multiply-add semantics for one dtype under the required active environment.
///
/// Algorithms must check this declaration and their active execution environment independently.
/// Arithmetic probes cannot establish a stronger provider contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FusedMulAddContract {
    /// No portable single-rounding or gradual-underflow guarantee is declared for this dtype.
    #[default]
    BackendDefined,
    /// Multiply-add rounds the infinite-precision `a * b + c` result once to the declared dtype.
    ///
    /// The operation never rounds or overflows an intermediate product separately.
    /// This declaration requires round-to-nearest ties-to-even and gradual input/output underflow.
    /// The caller preserves that active environment throughout execution.
    /// The provider does not detect or change rounding, flush-to-zero, or denormals modes.
    ///
    /// The guarantee covers tensor multiply-add and scalar multiply-add after conversion to the declared dtype.
    /// Existing shape restrictions apply. This contract grants no broadcasting support.
    /// Scalar conversion, add-multiply, reductions, and other operations remain outside this contract.
    /// IEEE special-value behavior applies without NaN payload or cross-platform bitwise identity promises.
    Ieee754Fused,
}
