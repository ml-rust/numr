//! 2:4 structured sparsity operations trait.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::Sparse24Tensor;
use crate::tensor::Tensor;

/// Operations for 2:4 structured sparsity
///
/// Provides pruning (dense → 2:4 compressed), decompression (2:4 → dense),
/// and sparse matrix multiplication using the compressed format.
pub trait Sparse24Ops<R: Runtime> {
    /// Prune a dense matrix to 2:4 structured sparsity
    ///
    /// For each group of 4 consecutive elements along the K dimension,
    /// keeps the 2 with largest magnitude and zeros the rest.
    ///
    /// # Arguments
    /// * `dense` - Input tensor of shape [M, K] where K is divisible by 4
    ///
    /// # Returns
    /// A `Sparse24Tensor` containing the compressed values and metadata
    fn prune_to_24(&self, dense: &Tensor<R>) -> Result<Sparse24Tensor<R>> {
        let _ = dense;
        Err(Error::NotImplemented {
            feature: "Sparse24Ops::prune_to_24",
        })
    }

    /// Decompress a 2:4 sparse tensor back to dense format
    ///
    /// Reconstructs the dense [M, K] matrix by placing non-zero values
    /// at their original positions (zeros elsewhere).
    fn sparse_24_to_dense(&self, sparse: &Sparse24Tensor<R>) -> Result<Tensor<R>> {
        let _ = sparse;
        Err(Error::NotImplemented {
            feature: "Sparse24Ops::sparse_24_to_dense",
        })
    }

    /// Matrix multiplication with 2:4 sparse weight matrix
    ///
    /// Computes `input @ weight^T` where weight is in 2:4 compressed format.
    ///
    /// # Arguments
    /// * `input` - Dense input tensor of shape [N, K]
    /// * `weight` - 2:4 sparse weight of original shape [M, K]
    ///
    /// # Returns
    /// Dense output tensor of shape [N, M]
    fn sparse_24_matmul(&self, input: &Tensor<R>, weight: &Sparse24Tensor<R>) -> Result<Tensor<R>> {
        let _ = (input, weight);
        Err(Error::NotImplemented {
            feature: "Sparse24Ops::sparse_24_matmul",
        })
    }
}
