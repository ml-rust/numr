//! Einsum operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Einstein summation convention operations.
///
/// Einsum provides a concise way to express many common multi-dimensional
/// linear algebraic array operations using a notation inspired by Einstein summation.
///
/// # Notation
///
/// The notation string has the form `"subscripts_input1,subscripts_input2,...->subscripts_output"`.
/// Each subscript is a single lowercase letter representing a dimension.
///
/// - Repeated subscripts across inputs indicate contraction (summation) over that dimension
/// - The output subscripts specify which dimensions appear in the result
/// - If `->` is omitted, the output is the sorted list of subscripts appearing exactly once
///
/// # Examples
///
/// ```ignore
/// // Matrix multiplication: C_ik = sum_j A_ij * B_jk
/// let c = client.einsum("ij,jk->ik", &[&a, &b])?;
///
/// // Batch matrix multiplication
/// let c = client.einsum("bij,bjk->bik", &[&a, &b])?;
///
/// // Trace of a matrix
/// let trace = client.einsum("ii->", &[&a])?;
///
/// // Outer product
/// let outer = client.einsum("i,j->ij", &[&a, &b])?;
///
/// // Element-wise multiplication (Hadamard product)
/// let hadamard = client.einsum("ij,ij->ij", &[&a, &b])?;
///
/// // Sum all elements
/// let total = client.einsum("ij->", &[&a])?;
///
/// // Transpose
/// let at = client.einsum("ij->ji", &[&a])?;
/// ```
pub trait EinsumOps<R: Runtime> {
    /// Evaluate an einsum expression.
    ///
    /// # Arguments
    ///
    /// * `notation` - Einsum notation string (e.g., `"ij,jk->ik"`)
    /// * `inputs` - Slice of input tensor references
    ///
    /// # Returns
    ///
    /// Result tensor computed according to the einsum expression
    fn einsum(&self, notation: &str, inputs: &[&Tensor<R>]) -> Result<Tensor<R>>;
}
