//! Semiring matrix multiplication trait.

use crate::error::{Error, Result};
use crate::ops::semiring::SemiringOp;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Generalized matrix multiplication over arbitrary semirings.
///
/// Standard matmul computes `C[i,j] = Σ_k A[i,k] * B[k,j]` using the (+, ×) semiring.
/// This trait generalizes to `C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])` for any semiring (⊕, ⊗).
///
/// # Supported Semirings
///
/// | Variant   | Reduce (⊕) | Combine (⊗) | Application |
/// |-----------|-----------|------------|-------------|
/// | MinPlus   | min       | +          | Shortest paths |
/// | MaxPlus   | max       | +          | Longest paths |
/// | MaxMin    | max       | min        | Bottleneck/capacity |
/// | MinMax    | min       | max        | Fuzzy relations |
/// | OrAnd     | OR        | AND        | Transitive closure |
/// | PlusMax   | +         | max        | DP formulations |
///
/// # DType Support
///
/// - **F32, F64, I32, I64**: All semiring ops except OrAnd
/// - **F16, BF16**: All semiring ops except OrAnd (with `f16` feature)
/// - **Bool**: OrAnd only
///
/// # Batched Operation
///
/// Supports batched matmul with the same broadcasting rules as standard matmul.
pub trait SemiringMatmulOps<R: Runtime> {
    /// Generalized semiring matrix multiplication.
    ///
    /// Computes `C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])` where ⊕ and ⊗ are defined by `op`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of shape `[..., M, K]`
    /// * `b` - Input tensor of shape `[..., K, N]`
    /// * `op` - The semiring operation to use
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., M, N]`
    fn semiring_matmul(&self, a: &Tensor<R>, b: &Tensor<R>, op: SemiringOp) -> Result<Tensor<R>> {
        let _ = (a, b, op);
        Err(Error::NotImplemented {
            feature: "SemiringMatmulOps::semiring_matmul",
        })
    }
}
