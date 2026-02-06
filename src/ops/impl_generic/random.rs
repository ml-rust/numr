//! Generic implementations of random composite operations.
//!
//! These implementations are shared across GPU backends (CUDA, WebGPU) to ensure
//! numerical parity and eliminate code duplication.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{RandomOps, SortingOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Random permutation of `[0, 1, ..., n-1]` via random keys + argsort (GPU-native)
///
/// Algorithm: Generate n random F32 keys, then argsort to get a permutation.
/// This is O(n log n) but fully parallelizable on GPU, unlike Fisher-Yates
/// which is inherently sequential.
pub fn randperm_impl<R, C>(client: &C, n: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RandomOps<R> + SortingOps<R>,
{
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "randperm requires n > 0".to_string(),
        });
    }

    // Generate n random F32 keys, then argsort to get a permutation
    let keys = client.rand(&[n], DType::F32)?;
    let perm = client.argsort(&keys, 0, false)?;

    // argsort returns I64, which is what we want
    Ok(perm)
}
