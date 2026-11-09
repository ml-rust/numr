//! Provider guarantees for separate basic floating-point tensor operations.

/// Declared provider contract for one floating dtype, independent of current execution-mode detection.
///
/// Algorithms requiring strict arithmetic must check this declaration and their active execution environment.
/// Sampled arithmetic probes cannot establish a stronger provider contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FloatArithmeticContract {
    /// No portable IEEE rounding or gradual-underflow guarantee is declared for this dtype.
    #[default]
    BackendDefined,
    /// Separate basic addition, subtraction, multiplication, and division follow IEEE 754 arithmetic.
    ///
    /// This declaration requires a valid active IEEE environment with round-to-nearest ties-to-even
    /// and gradual input/output underflow. The caller preserves that environment throughout execution.
    /// The provider does not detect or change the active rounding, flush-to-zero, or denormals mode.
    ///
    /// The guarantee covers the declared dtype's basic tensor dispatch, including broadcast, strided,
    /// and scalar routes after their scalar operand is represented in that dtype.
    /// It excludes scalar conversion, fused operations, reductions, and transcendental functions.
    /// IEEE special-value behavior applies without a NaN payload or cross-platform bitwise identity promise.
    Ieee754Basic,
}
