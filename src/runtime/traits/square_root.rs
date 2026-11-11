//! Provider guarantees for floating-point tensor square roots.

/// Declared square-root semantics for one dtype under the required active environment.
///
/// Algorithms must check this declaration and their active execution environment independently.
/// Arithmetic probes cannot establish a stronger provider contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SquareRootContract {
    /// No portable correct-rounding or gradual-underflow guarantee is declared for this dtype.
    #[default]
    BackendDefined,
    /// Square root rounds the infinite-precision result correctly to the declared dtype.
    ///
    /// This declaration requires round-to-nearest ties-to-even and gradual input/output underflow.
    /// The caller preserves that active environment throughout execution.
    /// The provider does not detect or change rounding, flush-to-zero, or denormals modes.
    ///
    /// The guarantee covers the declared dtype's tensor square-root dispatch, including strided inputs and scalar tails.
    /// Other roots, reciprocal square root, scalar conversion, and composite operations remain outside this contract.
    /// IEEE special-value behavior applies without NaN payload or cross-platform bitwise identity promises.
    Ieee754RoundToNearest,
}
