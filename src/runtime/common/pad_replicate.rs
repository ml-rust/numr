//! Validation for replicate-mode pad.

use super::shape_ops::{PadParams, validate_pad};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Validate inputs for replicate-mode pad and compute output parameters.
///
/// Same `padding` convention as [`validate_pad`]. Replicate clamps to the edge
/// element, so any pad size is valid on a non-empty dimension. A padded
/// dimension of size 0 has no edge element to copy and returns an error that
/// names the dimension.
pub fn validate_replicate_pad<R: Runtime>(
    tensor: &Tensor<R>,
    padding: &[usize],
) -> Result<PadParams> {
    let params = validate_pad(tensor, padding)?;

    for (dim, (&(before, after), &size)) in params
        .pad_per_dim
        .iter()
        .zip(tensor.shape().iter())
        .enumerate()
    {
        if size == 0 && (before > 0 || after > 0) {
            return Err(Error::InvalidArgument {
                arg: "padding",
                reason: format!(
                    "replicate pad (before={before}, after={after}) on dim {dim} has no edge \
                     element to copy because that dimension has size 0; set the padding on \
                     dim {dim} to 0 or pass an input with size >= 1 on dim {dim}"
                ),
            });
        }
    }

    Ok(params)
}
