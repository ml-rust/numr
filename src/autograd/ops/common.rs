//! Shared utilities for autograd backward implementations

use crate::autograd::Var;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Ensure a tensor is contiguous, making a copy if necessary.
///
/// Several `reshape` calls in backward implementations operate on tensors
/// that may be non-contiguous views (e.g. saved inputs, or grad_output after
/// a transpose/permute). `Tensor::reshape` intentionally errors on
/// non-contiguous tensors rather than silently materializing them, so
/// backward passes must call this helper first.
///
/// This intentionally duplicates [`crate::runtime::ensure_contiguous`] rather
/// than reusing it: that helper requires `R: Runtime<DType = DType>`, a bound
/// most `GradFn` impls in this module do not carry (and adding it would leak
/// into every `GradFn<R>` signature just to satisfy this one helper). Do NOT
/// collapse the two without first removing that bound from the callers here.
#[inline]
pub(crate) fn ensure_contiguous<R: Runtime>(tensor: &Tensor<R>) -> Result<Tensor<R>> {
    if tensor.is_contiguous() {
        Ok(tensor.clone())
    } else {
        tensor.contiguous()
    }
}

/// The matrix a rank-1 operand stands for in `matmul`: `[k]` on the left is
/// `[1, k]`, on the right `[k, 1]`, the same rule `matmul_output_shape` uses.
/// Higher ranks pass through. Gradients are computed against this shape and
/// reshaped back to the operand's own.
pub(crate) fn as_matrix_shape(shape: &[usize], is_lhs: bool) -> Vec<usize> {
    match shape {
        [k] if is_lhs => vec![1, *k],
        [k] => vec![*k, 1],
        _ => shape.to_vec(),
    }
}

pub(crate) fn as_matrix<R: Runtime>(t: &Tensor<R>, is_lhs: bool) -> Result<Tensor<R>> {
    let shape = as_matrix_shape(t.shape(), is_lhs);
    if shape == t.shape() {
        return Ok(t.clone());
    }
    t.contiguous()?.reshape(&shape)
}

pub(crate) fn restore_shape<R: Runtime>(t: Tensor<R>, shape: &[usize]) -> Result<Tensor<R>> {
    if t.shape() == shape {
        return Ok(t);
    }
    t.contiguous()?.reshape(shape)
}

/// [`restore_shape`] through the graph: a no-op when the shape already matches.
pub(crate) fn restore_var_shape<R: Runtime>(var: Var<R>, shape: &[usize]) -> Result<Var<R>> {
    if var.shape() == shape {
        return Ok(var);
    }
    super::shape::var_reshape(&var, shape)
}
