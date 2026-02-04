//! Polynomial construction from roots via iterative convolution

use super::{DTypeSupport, create_index_tensor};
use crate::algorithm::polynomial::helpers::{validate_polynomial_dtype, validate_polynomial_roots};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, UtilityOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Construct polynomial from roots via iterative convolution
///
/// # Algorithm
///
/// Uses complex polynomial multiplication to handle both real and complex roots.
/// For polynomial p(x) and root r = a + bi, we compute p(x) * (x - r) using
/// complex arithmetic on separate real/imaginary tensor pairs.
///
/// Complex polynomial multiplication:
/// - (p_real + i*p_imag) * (q_real + i*q_imag)
/// - result_real = conv(p_real, q_real) - conv(p_imag, q_imag)
/// - result_imag = conv(p_real, q_imag) + conv(p_imag, q_real)
///
/// If roots come in proper conjugate pairs, the final imaginary part will be ~0.
///
/// # Implementation
///
/// Uses only tensor operations (no GPU↔CPU transfers):
/// - `index_select` to access individual roots
/// - `cat` to build linear factors
/// - Tensor-based convolution via outer product + scatter_reduce
pub fn polyfromroots_impl<R, C>(
    client: &C,
    roots_real: &Tensor<R>,
    roots_imag: &Tensor<R>,
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
    validate_polynomial_dtype(roots_real.dtype())?;
    validate_polynomial_dtype(roots_imag.dtype())?;
    dtype_support.check(roots_real.dtype(), "polyfromroots")?;

    if roots_real.dtype() != roots_imag.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: roots_real.dtype(),
            rhs: roots_imag.dtype(),
        });
    }

    let n_roots = validate_polynomial_roots(roots_real.shape())?;
    let n_imag = validate_polynomial_roots(roots_imag.shape())?;

    if n_roots != n_imag {
        return Err(Error::ShapeMismatch {
            expected: vec![n_roots],
            got: vec![n_imag],
        });
    }

    let dtype = roots_real.dtype();
    let device = client.device();

    // Empty roots → constant polynomial [1.0]
    if n_roots == 0 {
        return Ok(Tensor::full_scalar(&[1], dtype, 1.0, device));
    }

    // Start with polynomial p(x) = 1 (represented as complex: real=[1], imag=[0])
    let mut p_real = Tensor::full_scalar(&[1], dtype, 1.0, device);
    let mut p_imag = Tensor::full_scalar(&[1], dtype, 0.0, device);

    // For each root r, multiply polynomial by (x - r)
    // Factor (x - r) where r = r_real + i*r_imag:
    //   factor_real = [-r_real, 1]
    //   factor_imag = [-r_imag, 0]
    for i in 0..n_roots {
        let idx = create_index_tensor::<R>(i, device);

        // Get root i as [1]-shaped tensors
        let r_real = client.index_select(roots_real, 0, &idx)?;
        let r_imag = client.index_select(roots_imag, 0, &idx)?;

        // Build factor (x - r):
        // factor_real = [-r_real, 1]
        // factor_imag = [-r_imag, 0]
        let neg_r_real = client.neg(&r_real)?;
        let neg_r_imag = client.neg(&r_imag)?;
        let one = Tensor::full_scalar(&[1], dtype, 1.0, device);
        let zero = Tensor::full_scalar(&[1], dtype, 0.0, device);

        let factor_real = client.cat(&[&neg_r_real, &one], 0)?;
        let factor_imag = client.cat(&[&neg_r_imag, &zero], 0)?;

        // Complex polynomial multiplication:
        // new_real = conv(p_real, factor_real) - conv(p_imag, factor_imag)
        // new_imag = conv(p_real, factor_imag) + conv(p_imag, factor_real)
        let conv_rr = convolve_tensors(client, &p_real, &factor_real)?;
        let conv_ii = convolve_tensors(client, &p_imag, &factor_imag)?;
        let conv_ri = convolve_tensors(client, &p_real, &factor_imag)?;
        let conv_ir = convolve_tensors(client, &p_imag, &factor_real)?;

        p_real = client.sub(&conv_rr, &conv_ii)?;
        p_imag = client.add(&conv_ri, &conv_ir)?;
    }

    // If roots were proper conjugate pairs, p_imag should be ~0
    // Return only the real part
    Ok(p_real)
}

/// Tensor-based 1D convolution for polynomial multiplication
///
/// Computes convolution using outer product + scatter_reduce, entirely on-device.
/// This is used internally by polyfromroots for complex polynomial multiplication.
fn convolve_tensors<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + ReduceOps<R>,
{
    let n_a = a.shape()[0];
    let n_b = b.shape()[0];
    let dtype = a.dtype();
    let device = client.device();
    let out_len = n_a + n_b - 1;

    if n_a == 0 || n_b == 0 {
        return Ok(Tensor::zeros(&[0], dtype, device));
    }

    // Convolution via outer product + scatter_reduce
    // 1. Compute outer product: outer[i,j] = a[i] * b[j]
    // 2. Create index tensor: indices[i,j] = i + j
    // 3. Flatten both and use scatter_reduce(Sum) to accumulate

    let a_col = a.reshape(&[n_a, 1])?;
    let b_row = b.reshape(&[1, n_b])?;

    // Outer product via broadcasting: [n_a, 1] * [1, n_b] = [n_a, n_b]
    let outer = client.mul(&a_col, &b_row)?;

    // Create index tensor for output positions: indices[i,j] = i + j
    let i_indices = client.arange(0.0, n_a as f64, 1.0, DType::I64)?;
    let j_indices = client.arange(0.0, n_b as f64, 1.0, DType::I64)?;

    let i_col = i_indices.reshape(&[n_a, 1])?;
    let j_row = j_indices.reshape(&[1, n_b])?;

    // Broadcast add to get output indices: [n_a, n_b]
    let out_indices = client.add(&i_col, &j_row)?;

    // Flatten both
    let outer_flat = outer.reshape(&[n_a * n_b])?;
    let indices_flat = out_indices.reshape(&[n_a * n_b])?;

    // Create output tensor of zeros
    let output = Tensor::zeros(&[out_len], dtype, device);

    // Use scatter_reduce with Sum to accumulate products at correct positions
    client.scatter_reduce(
        &output,
        0,
        &indices_flat,
        &outer_flat,
        crate::ops::ScatterReduceOp::Sum,
        true,
    )
}
