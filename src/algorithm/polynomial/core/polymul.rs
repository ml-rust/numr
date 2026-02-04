//! Polynomial multiplication via convolution

use super::DTypeSupport;
use crate::algorithm::polynomial::helpers::{
    validate_polynomial_coeffs, validate_polynomial_dtype,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, UtilityOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Multiply two polynomials via convolution
///
/// # Algorithm
///
/// For polynomials a(x) and b(x), c(x) = a(x) * b(x) where:
/// c[k] = Σᵢ a[i] * b[k-i] for valid i
///
/// Result has length: len(a) + len(b) - 1
///
/// # Implementation
///
/// Uses tensor-based convolution with outer product and scatter_reduce
/// to perform the computation on-device without GPU↔CPU transfers.
pub fn polymul_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + ScalarOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + ReduceOps<R>,
{
    validate_polynomial_dtype(a.dtype())?;
    validate_polynomial_dtype(b.dtype())?;
    dtype_support.check(a.dtype(), "polymul")?;

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    let n_a = validate_polynomial_coeffs(a.shape())?;
    let n_b = validate_polynomial_coeffs(b.shape())?;

    let dtype = a.dtype();
    let device = client.device();
    let out_len = n_a + n_b - 1;

    // Handle edge cases
    if n_a == 0 || n_b == 0 {
        return Ok(Tensor::zeros(&[0], dtype, device));
    }

    // Convolution via outer product + scatter_reduce
    //
    // 1. Compute outer product: outer[i,j] = a[i] * b[j]
    // 2. Create index tensor: indices[i,j] = i + j
    // 3. Flatten both and use scatter_reduce(Sum) to accumulate

    // Reshape a to [n_a, 1] and b to [1, n_b] for broadcasting
    let a_col = a.reshape(&[n_a, 1])?;
    let b_row = b.reshape(&[1, n_b])?;

    // Outer product via broadcasting: [n_a, 1] * [1, n_b] = [n_a, n_b]
    let outer = client.mul(&a_col, &b_row)?;

    // Create index tensor for output positions
    // indices[i,j] = i + j
    let i_indices = client.arange(0.0, n_a as f64, 1.0, DType::I64)?; // [n_a]
    let j_indices = client.arange(0.0, n_b as f64, 1.0, DType::I64)?; // [n_b]

    let i_col = i_indices.reshape(&[n_a, 1])?; // [n_a, 1]
    let j_row = j_indices.reshape(&[1, n_b])?; // [1, n_b]

    // Broadcast add to get output indices: [n_a, n_b]
    let out_indices = client.add(&i_col, &j_row)?;

    // Flatten both outer product and indices
    let outer_flat = outer.reshape(&[n_a * n_b])?;
    let indices_flat = out_indices.reshape(&[n_a * n_b])?;

    // Create output tensor of zeros
    let output = Tensor::zeros(&[out_len], dtype, device);

    // Use scatter_reduce with Sum to accumulate products at correct positions
    let result = client.scatter_reduce(
        &output,
        0,
        &indices_flat,
        &outer_flat,
        crate::ops::ScatterReduceOp::Sum,
        true, // include_self (start with zeros)
    )?;

    Ok(result)
}
