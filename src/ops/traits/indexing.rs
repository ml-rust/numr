//! Indexing operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Indexing operations
pub trait IndexingOps<R: Runtime> {
    /// Argmax: returns indices of maximum values along a dimension.
    ///
    /// Returns a tensor of I64 indices indicating the position of the maximum
    /// value along the specified dimension. The output shape is the input shape
    /// with the specified dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to find the maximum index
    /// * `keepdim` - If true, the reduced dimension is retained with size 1
    ///
    /// # Returns
    ///
    /// Tensor of I64 containing indices of maximum values
    fn argmax(&self, a: &Tensor<R>, dim: usize, keepdim: bool) -> Result<Tensor<R>>;

    /// Argmin: returns indices of minimum values along a dimension.
    ///
    /// Returns a tensor of I64 indices indicating the position of the minimum
    /// value along the specified dimension. The output shape is the input shape
    /// with the specified dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to find the minimum index
    /// * `keepdim` - If true, the reduced dimension is retained with size 1
    ///
    /// # Returns
    ///
    /// Tensor of I64 containing indices of minimum values
    fn argmin(&self, a: &Tensor<R>, dim: usize, keepdim: bool) -> Result<Tensor<R>>;

    /// Gather elements along a dimension using an index tensor.
    ///
    /// For a 3D tensor with dim=1:
    /// `out[i][j][k] = input[i][index[i][j][k]][k]`
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to gather
    /// * `index` - Index tensor (I64) with same number of dimensions as input
    ///
    /// # Returns
    ///
    /// Tensor with same shape as index tensor, same dtype as input
    fn gather(&self, a: &Tensor<R>, dim: usize, index: &Tensor<R>) -> Result<Tensor<R>>;

    /// Scatter values into a tensor at positions specified by an index tensor.
    ///
    /// Creates a new tensor (copy of `a`) with values from `src` scattered at positions
    /// specified by `index` along dimension `dim`.
    ///
    /// For a 3D tensor with dim=1:
    /// `out[i][index[i][j][k]][k] = src[i][j][k]`
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (values to scatter into)
    /// * `dim` - Dimension along which to scatter
    /// * `index` - Index tensor (I64) specifying scatter positions
    /// * `src` - Source tensor with values to scatter
    ///
    /// # Returns
    ///
    /// New tensor with scattered values
    fn scatter(
        &self,
        a: &Tensor<R>,
        dim: usize,
        index: &Tensor<R>,
        src: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Select elements along a dimension using a 1D index tensor.
    ///
    /// Simpler than gather - the index tensor is 1D and applies to all positions
    /// in the specified dimension.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to select
    /// * `index` - 1D index tensor (I64) of length m
    ///
    /// # Returns
    ///
    /// Tensor with dimension `dim` having size m (length of index)
    ///
    /// # Bounds Checking
    ///
    /// Returns `IndexOutOfBounds` error if any index is negative or >= dim_size.
    /// Indices must be in the range `[0, dim_size)`. Negative indices are not supported.
    fn index_select(&self, a: &Tensor<R>, dim: usize, index: &Tensor<R>) -> Result<Tensor<R>>;

    /// Put values at specified indices along a dimension.
    ///
    /// This is the inverse of `index_select` - it assigns values from `src` into
    /// positions specified by `index` along dimension `dim`.
    ///
    /// # Example
    ///
    /// ```text
    /// # Replace row 2 of a [5, 3] matrix with new values:
    /// a = [[1, 2, 3],
    ///      [4, 5, 6],
    ///      [7, 8, 9],      # <- row 2 will be replaced
    ///      [10, 11, 12],
    ///      [13, 14, 15]]
    /// index = [2]          # indices along dim 0
    /// src = [[100, 200, 300]]
    ///
    /// result = index_put(a, 0, index, src)
    /// # result = [[1, 2, 3],
    /// #           [4, 5, 6],
    /// #           [100, 200, 300],  # replaced!
    /// #           [10, 11, 12],
    /// #           [13, 14, 15]]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor to modify (copied, not mutated)
    /// * `dim` - Dimension along which to put values
    /// * `index` - 1D index tensor (I64) specifying positions
    /// * `src` - Source tensor with values to insert. Shape must match `a` except
    ///   at `dim` where it must equal `index.numel()`
    ///
    /// # Returns
    ///
    /// New tensor with values at indexed positions replaced by `src`
    ///
    /// # Bounds Checking
    ///
    /// Returns `IndexOutOfBounds` error if any index is negative or >= dim_size.
    /// Indices must be in the range `[0, dim_size)`. Negative indices are not supported.
    fn index_put(
        &self,
        a: &Tensor<R>,
        dim: usize,
        index: &Tensor<R>,
        src: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Select elements where mask is true, returning a flattened 1D tensor.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `mask` - Boolean mask tensor (U8: 0=false, non-zero=true), must be broadcastable to `a`
    ///
    /// # Returns
    ///
    /// 1D tensor containing only elements where mask is true
    fn masked_select(&self, a: &Tensor<R>, mask: &Tensor<R>) -> Result<Tensor<R>>;

    /// Fill elements where mask is true with a scalar value.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `mask` - Boolean mask tensor (U8: 0=false, non-zero=true), must be broadcastable to `a`
    /// * `value` - Value to fill where mask is true
    ///
    /// # Returns
    ///
    /// New tensor with masked positions filled with value
    fn masked_fill(&self, a: &Tensor<R>, mask: &Tensor<R>, value: f64) -> Result<Tensor<R>>;

    /// Look up embeddings from an embedding table using indices.
    ///
    /// This is the standard embedding lookup operation used in neural networks
    /// for word embeddings, entity embeddings, etc. It is equivalent to
    /// `index_select(embeddings, 0, indices)` but optimized for the common case
    /// where the embedding table is 2D and indices index into the first dimension.
    ///
    /// # Algorithm
    ///
    /// For each index value i in the indices tensor:
    /// ```text
    /// output[..., i, :] = embeddings[indices[..., i], :]
    /// ```
    ///
    /// The output shape is `indices.shape() + [embedding_dim]` where `embedding_dim`
    /// is `embeddings.shape()[1]`.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - 2D embedding table of shape `[vocab_size, embedding_dim]`
    /// * `indices` - Index tensor of any shape containing indices into the embedding table.
    ///   Must be I64 (or I32 on WebGPU). Values must be in range `[0, vocab_size)`.
    ///
    /// # Returns
    ///
    /// Tensor of shape `indices.shape() + [embedding_dim]` containing the looked-up embeddings.
    ///
    /// # Example
    ///
    /// ```text
    /// embeddings = [[1.0, 2.0],   # word 0
    ///               [3.0, 4.0],   # word 1
    ///               [5.0, 6.0]]   # word 2
    /// indices = [2, 0, 1]
    ///
    /// output = [[5.0, 6.0],   # word 2
    ///           [1.0, 2.0],   # word 0
    ///           [3.0, 4.0]]   # word 1
    /// ```
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if embeddings is not 2D
    /// * `DTypeMismatch` - if indices is not I64 (or I32 on WebGPU)
    /// * Index out of bounds results in undefined behavior (implementation may return zeros)
    ///
    /// # Performance
    ///
    /// On GPU, this operation is memory-bound and optimized for coalesced reads
    /// from the embedding table. Each thread handles one index lookup and writes
    /// a full embedding vector.
    fn embedding_lookup(&self, embeddings: &Tensor<R>, indices: &Tensor<R>) -> Result<Tensor<R>>;
}
