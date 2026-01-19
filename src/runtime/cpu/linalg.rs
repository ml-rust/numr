//! CPU implementation of linear algebra algorithms
//!
//! This module implements the [`LinearAlgebraAlgorithms`] trait for CPU.
//! All algorithms follow the exact specification in the trait documentation
//! to ensure backend parity with CUDA/WebGPU implementations.

use super::jacobi::{
    self, JacobiRotation, LinalgElement, apply_rotation_to_columns, apply_two_sided_rotation,
    argsort_by_magnitude_desc, argsort_desc, compute_gram_elements, identity_matrix,
    normalize_columns, permute_columns,
};
use super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    CholeskyDecomposition, EigenDecomposition, LinearAlgebraAlgorithms, LuDecomposition,
    MatrixNormOrder, QrDecomposition, SvdDecomposition, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

impl LinearAlgebraAlgorithms<CpuRuntime> for CpuClient {
    fn lu_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<LuDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => lu_decompose_impl::<f32>(self, a, m, n),
            DType::F64 => lu_decompose_impl::<f64>(self, a, m, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "lu_decompose",
            }),
        }
    }

    fn cholesky_decompose(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<CholeskyDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => cholesky_decompose_impl::<f32>(self, a, n),
            DType::F64 => cholesky_decompose_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "cholesky_decompose",
            }),
        }
    }

    fn qr_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<QrDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => qr_decompose_impl::<f32>(self, a, m, n, false),
            DType::F64 => qr_decompose_impl::<f64>(self, a, m, n, false),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "qr_decompose",
            }),
        }
    }

    fn qr_decompose_thin(&self, a: &Tensor<CpuRuntime>) -> Result<QrDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => qr_decompose_impl::<f32>(self, a, m, n, true),
            DType::F64 => qr_decompose_impl::<f64>(self, a, m, n, true),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "qr_decompose_thin",
            }),
        }
    }

    fn solve(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => solve_impl::<f32>(self, a, b, n),
            DType::F64 => solve_impl::<f64>(self, a, b, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "solve",
            }),
        }
    }

    fn solve_triangular_lower(
        &self,
        l: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        unit_diagonal: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(l.dtype())?;
        if l.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: l.dtype(),
                rhs: b.dtype(),
            });
        }
        let n = validate_square_matrix(l.shape())?;

        match l.dtype() {
            DType::F32 => solve_triangular_lower_impl::<f32>(self, l, b, n, unit_diagonal),
            DType::F64 => solve_triangular_lower_impl::<f64>(self, l, b, n, unit_diagonal),
            _ => Err(Error::UnsupportedDType {
                dtype: l.dtype(),
                op: "solve_triangular_lower",
            }),
        }
    }

    fn solve_triangular_upper(
        &self,
        u: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(u.dtype())?;
        if u.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: u.dtype(),
                rhs: b.dtype(),
            });
        }
        let n = validate_square_matrix(u.shape())?;

        match u.dtype() {
            DType::F32 => solve_triangular_upper_impl::<f32>(self, u, b, n),
            DType::F64 => solve_triangular_upper_impl::<f64>(self, u, b, n),
            _ => Err(Error::UnsupportedDType {
                dtype: u.dtype(),
                op: "solve_triangular_upper",
            }),
        }
    }

    fn lstsq(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => lstsq_impl::<f32>(self, a, b, m, n),
            DType::F64 => lstsq_impl::<f64>(self, a, b, m, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "lstsq",
            }),
        }
    }

    fn inverse(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => inverse_impl::<f32>(self, a, n),
            DType::F64 => inverse_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "inverse",
            }),
        }
    }

    fn det(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => det_impl::<f32>(self, a, n),
            DType::F64 => det_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "det",
            }),
        }
    }

    fn trace(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => trace_impl::<f32>(self, a, m, n),
            DType::F64 => trace_impl::<f64>(self, a, m, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "trace",
            }),
        }
    }

    fn diag(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => diag_impl::<f32>(self, a, m, n),
            DType::F64 => diag_impl::<f64>(self, a, m, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "diag",
            }),
        }
    }

    fn diagflat(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        if a.ndim() != 1 {
            return Err(Error::Internal(format!(
                "diagflat expects 1D tensor, got {}D",
                a.ndim()
            )));
        }

        match a.dtype() {
            DType::F32 => diagflat_impl::<f32>(self, a),
            DType::F64 => diagflat_impl::<f64>(self, a),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "diagflat",
            }),
        }
    }

    fn matrix_rank(&self, a: &Tensor<CpuRuntime>, tol: Option<f64>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => matrix_rank_impl::<f32>(self, a, m, n, tol),
            DType::F64 => matrix_rank_impl::<f64>(self, a, m, n, tol),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "matrix_rank",
            }),
        }
    }

    fn matrix_norm(
        &self,
        a: &Tensor<CpuRuntime>,
        ord: MatrixNormOrder,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (_m, _n) = validate_matrix_2d(a.shape())?;

        match ord {
            MatrixNormOrder::Frobenius => match a.dtype() {
                DType::F32 => frobenius_norm_impl::<f32>(self, a),
                DType::F64 => frobenius_norm_impl::<f64>(self, a),
                _ => Err(Error::UnsupportedDType {
                    dtype: a.dtype(),
                    op: "matrix_norm",
                }),
            },
            MatrixNormOrder::Spectral | MatrixNormOrder::Nuclear => Err(Error::Internal(
                "Spectral and nuclear norms require SVD (not yet implemented)".to_string(),
            )),
        }
    }

    fn svd_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<SvdDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => svd_decompose_impl::<f32>(self, a, m, n),
            DType::F64 => svd_decompose_impl::<f64>(self, a, m, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "svd_decompose",
            }),
        }
    }

    fn eig_decompose_symmetric(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<EigenDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => eig_decompose_symmetric_impl::<f32>(self, a, n),
            DType::F64 => eig_decompose_symmetric_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "eig_decompose_symmetric",
            }),
        }
    }
}

// ============================================================================
// Implementation Functions
// ============================================================================

/// LU decomposition with partial pivoting (Doolittle algorithm)
fn lu_decompose_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<LuDecomposition<CpuRuntime>> {
    let device = client.device();
    let k = m.min(n);

    // Copy input to working buffer (will be modified in-place)
    let lu_data: Vec<T> = a.to_vec();
    let mut lu: Vec<T> = lu_data;
    let mut pivots: Vec<i64> = vec![0; k];
    let mut num_swaps = 0usize;

    // LU decomposition with partial pivoting
    for col in 0..k {
        // Find pivot: max absolute value in column col, rows col..m
        let mut pivot_row = col;
        let mut max_val = lu[col * n + col].abs_val();

        for row in (col + 1)..m {
            let val = lu[row * n + col].abs_val();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }

        pivots[col] = pivot_row as i64;

        // Swap rows if needed
        if pivot_row != col {
            for j in 0..n {
                lu.swap(col * n + j, pivot_row * n + j);
            }
            num_swaps += 1;
        }

        // Check for zero pivot (singular matrix)
        let pivot = lu[col * n + col];
        if pivot.abs_val().to_f64() < T::epsilon_val() {
            return Err(Error::Internal("Matrix is singular".to_string()));
        }

        // Compute multipliers (L column)
        for row in (col + 1)..m {
            lu[row * n + col] = lu[row * n + col] / pivot;
        }

        // Update trailing submatrix
        for row in (col + 1)..m {
            let multiplier = lu[row * n + col];
            for j in (col + 1)..n {
                let update = multiplier * lu[col * n + j];
                lu[row * n + j] = lu[row * n + j] - update;
            }
        }
    }

    // Create output tensors
    let lu_tensor = Tensor::<CpuRuntime>::from_slice(&lu, &[m, n], device);
    let pivots_tensor = Tensor::<CpuRuntime>::from_slice(&pivots, &[k], device);

    Ok(LuDecomposition {
        lu: lu_tensor,
        pivots: pivots_tensor,
        num_swaps,
    })
}

/// Cholesky decomposition (Cholesky-Banachiewicz algorithm)
fn cholesky_decompose_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<CholeskyDecomposition<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();
    let mut l: Vec<T> = vec![T::zero(); n * n];

    for i in 0..n {
        // Compute diagonal element
        let mut sum_sq = T::zero();
        for k in 0..i {
            sum_sq = sum_sq + l[i * n + k] * l[i * n + k];
        }

        let diag = a_data[i * n + i] - sum_sq;
        if diag.to_f64() <= 0.0 {
            return Err(Error::Internal(
                "Matrix is not positive definite".to_string(),
            ));
        }
        l[i * n + i] = diag.sqrt_val();

        // Compute off-diagonal elements in column i
        for j in (i + 1)..n {
            let mut sum_prod = T::zero();
            for k in 0..i {
                sum_prod = sum_prod + l[j * n + k] * l[i * n + k];
            }
            l[j * n + i] = (a_data[j * n + i] - sum_prod) / l[i * n + i];
        }
    }

    let l_tensor = Tensor::<CpuRuntime>::from_slice(&l, &[n, n], device);
    Ok(CholeskyDecomposition { l: l_tensor })
}

/// QR decomposition using Householder reflections
fn qr_decompose_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    thin: bool,
) -> Result<QrDecomposition<CpuRuntime>> {
    let device = client.device();
    let k = m.min(n);

    // R starts as copy of A
    let mut r: Vec<T> = a.to_vec();

    // Q starts as identity (m x m or m x k for thin)
    let q_cols = if thin { k } else { m };
    let mut q: Vec<T> = vec![T::zero(); m * q_cols];
    for i in 0..q_cols.min(m) {
        q[i * q_cols + i] = T::one();
    }

    // Householder vectors storage
    for col in 0..k {
        // Extract column vector below diagonal: x = R[col:m, col]
        let x_len = m - col;
        let mut x: Vec<T> = vec![T::zero(); x_len];
        for i in 0..x_len {
            x[i] = r[(col + i) * n + col];
        }

        // Compute norm of x
        let mut norm_sq = T::zero();
        for &val in &x {
            norm_sq = norm_sq + val * val;
        }
        let norm_x = norm_sq.sqrt_val();

        if norm_x.abs_val().to_f64() < T::epsilon_val() {
            continue;
        }

        // Compute Householder vector v
        // alpha = -sign(x[0]) * ||x||
        let alpha = if x[0].to_f64() >= 0.0 {
            norm_x.neg_val()
        } else {
            norm_x
        };

        // v = x; v[0] -= alpha
        let mut v = x.clone();
        v[0] = v[0] - alpha;

        // Normalize v
        let mut v_norm_sq = T::zero();
        for &val in &v {
            v_norm_sq = v_norm_sq + val * val;
        }
        let v_norm = v_norm_sq.sqrt_val();

        if v_norm.abs_val().to_f64() < T::epsilon_val() {
            continue;
        }

        for val in &mut v {
            *val = *val / v_norm;
        }

        // Apply reflection to R: R[col:m, col:n] -= 2 * v @ (v^T @ R[col:m, col:n])
        // First compute w = v^T @ R[col:m, col:n], shape [n - col]
        let mut w: Vec<T> = vec![T::zero(); n - col];
        for j in 0..(n - col) {
            for i in 0..x_len {
                w[j] = w[j] + v[i] * r[(col + i) * n + (col + j)];
            }
        }

        // R[col:m, col:n] -= 2 * v @ w^T
        let two = T::from_f64(2.0);
        for i in 0..x_len {
            for j in 0..(n - col) {
                let update = two * v[i] * w[j];
                r[(col + i) * n + (col + j)] = r[(col + i) * n + (col + j)] - update;
            }
        }

        // Apply reflection to Q: Q[:, col:m] -= 2 * Q[:, col:m] @ v @ v^T
        // Q[:, col:m] = Q[:, col:m] @ (I - 2*v*v^T)
        // For each row of Q, compute the update
        for row in 0..m {
            // Compute dot product of Q[row, col:m] with v
            let mut dot = T::zero();
            for i in 0..x_len {
                if col + i < q_cols {
                    dot = dot + q[row * q_cols + (col + i)] * v[i];
                }
            }

            // Update Q[row, col:m] -= 2 * dot * v
            for i in 0..x_len {
                if col + i < q_cols {
                    let update = two * dot * v[i];
                    q[row * q_cols + (col + i)] = q[row * q_cols + (col + i)] - update;
                }
            }
        }
    }

    // Create output tensors
    let q_tensor = Tensor::<CpuRuntime>::from_slice(&q, &[m, q_cols], device);

    // R is m x n, but for thin QR we only need k x n
    let r_rows = if thin { k } else { m };
    let mut r_out: Vec<T> = vec![T::zero(); r_rows * n];
    for i in 0..r_rows {
        for j in 0..n {
            r_out[i * n + j] = r[i * n + j];
        }
    }
    let r_tensor = Tensor::<CpuRuntime>::from_slice(&r_out, &[r_rows, n], device);

    Ok(QrDecomposition {
        q: q_tensor,
        r: r_tensor,
    })
}

/// Solve Ax = b using LU decomposition
fn solve_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl::<T>(client, a, n, n)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();
    let pivots_data: Vec<i64> = lu_decomp.pivots.to_vec();

    // Handle 1D or 2D b
    let b_shape = b.shape();
    let (b_rows, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    if b_rows != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: vec![b_rows],
        });
    }

    let b_data: Vec<T> = b.to_vec();
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        // Apply permutation to b
        let mut pb: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            pb[i] = if num_rhs == 1 {
                b_data[i]
            } else {
                b_data[i * num_rhs + rhs]
            };
        }
        for (i, &pivot_idx) in pivots_data.iter().enumerate() {
            let pivot_row = pivot_idx as usize;
            if pivot_row != i {
                pb.swap(i, pivot_row);
            }
        }

        // Forward substitution: Ly = Pb (L has unit diagonal)
        let mut y: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                sum = sum + lu_data[i * n + j] * y[j];
            }
            y[i] = pb[i] - sum;
        }

        // Backward substitution: Ux = y
        let mut x_col: Vec<T> = vec![T::zero(); n];
        for ii in (0..n).rev() {
            let mut s = T::zero();
            for jj in (ii + 1)..n {
                s = s + lu_data[ii * n + jj] * x_col[jj];
            }
            x_col[ii] = (y[ii] - s) / lu_data[ii * n + ii];
        }
        // Copy result
        for ii in 0..n {
            if num_rhs == 1 {
                x[ii] = x_col[ii];
            } else {
                x[ii * num_rhs + rhs] = x_col[ii];
            }
        }
    }

    // Create output tensor
    if b_shape.len() == 1 {
        Ok(Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device))
    } else {
        Ok(Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device))
    }
}

/// Forward substitution for lower triangular system
fn solve_triangular_lower_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    l: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
    unit_diagonal: bool,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let l_data: Vec<T> = l.to_vec();
    let b_shape = b.shape();
    let (_, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    let b_data: Vec<T> = b.to_vec();
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                let x_val = if num_rhs == 1 {
                    x[j]
                } else {
                    x[j * num_rhs + rhs]
                };
                sum = sum + l_data[i * n + j] * x_val;
            }

            let b_val = if num_rhs == 1 {
                b_data[i]
            } else {
                b_data[i * num_rhs + rhs]
            };

            let result = b_val - sum;
            let x_val = if unit_diagonal {
                result
            } else {
                result / l_data[i * n + i]
            };

            if num_rhs == 1 {
                x[i] = x_val;
            } else {
                x[i * num_rhs + rhs] = x_val;
            }
        }
    }

    if b_shape.len() == 1 {
        Ok(Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device))
    } else {
        Ok(Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device))
    }
}

/// Backward substitution for upper triangular system
fn solve_triangular_upper_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    u: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let u_data: Vec<T> = u.to_vec();
    let b_shape = b.shape();
    let (_, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    let b_data: Vec<T> = b.to_vec();
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        for i in (0..n).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..n {
                let x_val = if num_rhs == 1 {
                    x[j]
                } else {
                    x[j * num_rhs + rhs]
                };
                sum = sum + u_data[i * n + j] * x_val;
            }

            let b_val = if num_rhs == 1 {
                b_data[i]
            } else {
                b_data[i * num_rhs + rhs]
            };

            let x_val = (b_val - sum) / u_data[i * n + i];

            if num_rhs == 1 {
                x[i] = x_val;
            } else {
                x[i * num_rhs + rhs] = x_val;
            }
        }
    }

    if b_shape.len() == 1 {
        Ok(Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device))
    } else {
        Ok(Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device))
    }
}

/// Least squares via QR decomposition
fn lstsq_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute thin QR
    let qr = qr_decompose_impl::<T>(client, a, m, n, true)?;
    let q_data: Vec<T> = qr.q.to_vec();
    let r_data: Vec<T> = qr.r.to_vec();

    let b_shape = b.shape();
    let (_, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    let b_data: Vec<T> = b.to_vec();
    let k = m.min(n);

    // Compute Q^T @ b
    let mut qtb: Vec<T> = vec![T::zero(); k * num_rhs];
    for rhs in 0..num_rhs {
        for i in 0..k {
            let mut sum = T::zero();
            for j in 0..m {
                let b_val = if num_rhs == 1 {
                    b_data[j]
                } else {
                    b_data[j * num_rhs + rhs]
                };
                sum = sum + q_data[j * k + i] * b_val;
            }
            if num_rhs == 1 {
                qtb[i] = sum;
            } else {
                qtb[i * num_rhs + rhs] = sum;
            }
        }
    }

    // Solve R @ x = Q^T @ b via back substitution
    // R is k x n, but only the first k columns are used
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        // Solve k x k upper triangular system
        for i in (0..k).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..k {
                let x_val = if num_rhs == 1 {
                    x[j]
                } else {
                    x[j * num_rhs + rhs]
                };
                sum = sum + r_data[i * n + j] * x_val;
            }

            let qtb_val = if num_rhs == 1 {
                qtb[i]
            } else {
                qtb[i * num_rhs + rhs]
            };

            let x_val = (qtb_val - sum) / r_data[i * n + i];

            if num_rhs == 1 {
                x[i] = x_val;
            } else {
                x[i * num_rhs + rhs] = x_val;
            }
        }
        // Remaining x[k..n] are zeros (already initialized)
    }

    if b_shape.len() == 1 {
        Ok(Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device))
    } else {
        Ok(Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device))
    }
}

/// Matrix inverse via LU decomposition
fn inverse_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Create identity matrix
    let mut identity: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        identity[i * n + i] = T::one();
    }
    let identity_tensor = Tensor::<CpuRuntime>::from_slice(&identity, &[n, n], device);

    // Solve A @ X = I
    solve_impl::<T>(client, a, &identity_tensor, n)
}

/// Determinant via LU decomposition
fn det_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle special case n=0
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::from_slice(&[T::one()], &[], device));
    }

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl::<T>(client, a, n, n)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();

    // det = (-1)^num_swaps * product(U[i,i])
    let mut det = if lu_decomp.num_swaps % 2 == 0 {
        T::one()
    } else {
        T::one().neg_val()
    };

    for i in 0..n {
        det = det * lu_data[i * n + i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&[det], &[], device))
}

/// Trace: sum of diagonal elements
fn trace_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut trace = T::zero();
    for i in 0..k {
        trace = trace + a_data[i * n + i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&[trace], &[], device))
}

/// Extract diagonal
fn diag_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut diag: Vec<T> = vec![T::zero(); k];
    for i in 0..k {
        diag[i] = a_data[i * n + i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&diag, &[k], device))
}

/// Create diagonal matrix from 1D tensor
fn diagflat_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let n = a.shape()[0];
    let a_data: Vec<T> = a.to_vec();

    let mut mat: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        mat[i * n + i] = a_data[i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&mat, &[n, n], device))
}

/// Matrix rank via singular value thresholding
/// Uses QR-based approach since SVD is not yet implemented
fn matrix_rank_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    tol: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Use QR decomposition to estimate rank (R diagonal gives singular value bounds)
    let qr = qr_decompose_impl::<T>(client, a, m, n, true)?;
    let r_data: Vec<T> = qr.r.to_vec();

    let k = m.min(n);

    // Find max diagonal element of R (upper bound on max singular value)
    let mut max_diag = T::zero();
    for i in 0..k {
        let val = r_data[i * n + i].abs_val();
        if val.to_f64() > max_diag.to_f64() {
            max_diag = val;
        }
    }

    // Compute tolerance
    let eps = if a.dtype() == DType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    let tolerance = tol.unwrap_or_else(|| m.max(n) as f64 * eps * max_diag.to_f64());

    // Count diagonal elements above tolerance
    let mut rank = 0i64;
    for i in 0..k {
        if r_data[i * n + i].abs_val().to_f64() > tolerance {
            rank += 1;
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&[rank], &[], device))
}

/// Frobenius norm: ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2)
fn frobenius_norm_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    // Sum of squares of all elements
    let mut sum_sq = T::zero();
    for &val in &a_data {
        sum_sq = sum_sq + val * val;
    }

    let norm = sum_sq.sqrt_val();
    Ok(Tensor::<CpuRuntime>::from_slice(&[norm], &[], device))
}

/// SVD decomposition using One-Sided Jacobi algorithm
///
/// Algorithm: One-Sided Jacobi SVD
/// 1. If m < n: Transpose A, compute SVD, swap U↔V^T
/// 2. Initialize: B = A (working copy), V = I_n
/// 3. REPEAT (max 30 sweeps):
///    FOR each pair (p, q) where p < q:
///      - Compute Gram elements: a_pp, a_qq, a_pq = B[:,p]·B[:,q]
///      - If |a_pq| > tol: compute Jacobi rotation (c,s), apply to B and V columns
///      - Check convergence: sqrt(Σ a_pq²) < n * epsilon
/// 4. Extract: S[j] = ||B[:,j]||, U[:,j] = B[:,j]/S[j]
/// 5. Sort S descending, reorder U and V columns accordingly
/// 6. Return U, S, V^T = V.transpose()
fn svd_decompose_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<SvdDecomposition<CpuRuntime>> {
    let device = client.device();
    let k = m.min(n);

    // Handle empty matrix
    if m == 0 || n == 0 {
        let u = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[m, k], device);
        let s = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[k], device);
        let vt = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[k, n], device);
        return Ok(SvdDecomposition { u, s, vt });
    }

    // If m < n, transpose and swap U/V at the end
    let transpose = m < n;
    let (work_m, work_n) = if transpose { (n, m) } else { (m, n) };

    // Get input data, transposing if needed
    let mut b: Vec<T> = if transpose {
        // Transpose input: A[i,j] -> A^T[j,i]
        let a_data: Vec<T> = a.to_vec();
        let mut b_transposed = vec![T::zero(); work_m * work_n];
        for i in 0..m {
            for j in 0..n {
                b_transposed[j * work_n + i] = a_data[i * n + j];
            }
        }
        b_transposed
    } else {
        a.to_vec()
    };

    let work_k = work_m.min(work_n);

    // Initialize V as identity [work_n x work_n]
    let mut v: Vec<T> = identity_matrix(work_n);

    // Convergence parameters
    let eps = T::epsilon_val();
    let tol = (work_n as f64) * eps;
    let max_sweeps = 30;

    // One-Sided Jacobi iterations
    for _sweep in 0..max_sweeps {
        let mut off_diag_sum = 0.0f64;

        // Process all column pairs (p, q) where p < q
        for p in 0..work_n {
            for q in (p + 1)..work_n {
                // Compute Gram matrix elements for columns p and q
                let (a_pp, a_qq, a_pq) = compute_gram_elements(&b, work_m, work_n, p, q);

                off_diag_sum += a_pq.to_f64() * a_pq.to_f64();

                // Skip if off-diagonal is essentially zero
                if a_pq.abs_val().to_f64() < tol * (a_pp.to_f64() * a_qq.to_f64()).sqrt() {
                    continue;
                }

                // Compute Jacobi rotation using stable LAPACK formula
                let rot = JacobiRotation::compute(a_pp.to_f64(), a_qq.to_f64(), a_pq.to_f64());

                // Apply rotation to B columns
                apply_rotation_to_columns(&mut b, work_m, work_n, p, q, &rot);

                // Apply rotation to V columns
                apply_rotation_to_columns(&mut v, work_n, work_n, p, q, &rot);
            }
        }

        // Check convergence: sqrt(sum of squared off-diagonals) < tolerance
        if off_diag_sum.sqrt() < tol {
            break;
        }
    }

    // Extract singular values and normalize U columns using shared utility
    // S[j] = ||B[:,j]||, U[:,j] = B[:,j] / S[j]
    let singular_values = normalize_columns(&mut b, work_m, work_n, eps);
    let u_data = b; // b is now U after normalization

    // Sort singular values in descending order and reorder U, V accordingly
    let indices = argsort_desc(&singular_values);

    // Reorder singular values (take first work_k)
    let s_sorted: Vec<T> = jacobi::permute_vector(&singular_values, &indices)
        .into_iter()
        .take(work_k)
        .collect();

    // Reorder U columns (take first work_k)
    let u_sorted = permute_columns(&u_data, work_m, work_n, &indices, work_k);

    // Reorder V columns and transpose to get V^T (take first work_k rows)
    // V^T[i, j] = V[j, perm[i]]
    let mut vt_sorted: Vec<T> = vec![T::zero(); work_k * work_n];
    for (new_idx, &old_idx) in indices.iter().take(work_k).enumerate() {
        for j in 0..work_n {
            vt_sorted[new_idx * work_n + j] = v[j * work_n + old_idx];
        }
    }

    // If we transposed at the beginning, swap U and V^T
    if transpose {
        // Original: A = U @ S @ V^T
        // Transposed: A^T = U' @ S @ V'^T
        // So: A = (U' @ S @ V'^T)^T = V' @ S @ U'^T
        // Therefore: U_final = V', V^T_final = U'^T
        //
        // For original A [m x n] with m < n:
        // - U_final should be [m x k] where k = min(m,n) = m
        // - V^T_final should be [k x n] = [m x n]
        //
        // After A^T SVD:
        // - u_sorted is [work_m x work_k] = [n x m]
        // - vt_sorted is [work_k x work_n] = [m x m]

        // U_final = V' = (vt_sorted)^T, shape [m x k] = [m x m]
        // vt_sorted is [work_k x work_n] = [m x m], so transpose to [m x m]
        let mut u_final: Vec<T> = vec![T::zero(); m * k];
        for i in 0..k {
            for j in 0..m {
                // u_final[j, i] = vt_sorted[i, j]
                u_final[j * k + i] = vt_sorted[i * work_n + j];
            }
        }

        // V^T_final = U'^T, shape [k x n] = [m x n]
        // u_sorted is [work_m x work_k] = [n x m], transpose to [m x n]
        let mut vt_final: Vec<T> = vec![T::zero(); k * n];
        for i in 0..work_m {
            for j in 0..work_k {
                // vt_final[j, i] = u_sorted[i, j]
                vt_final[j * n + i] = u_sorted[i * work_k + j];
            }
        }

        let u_tensor = Tensor::<CpuRuntime>::from_slice(&u_final, &[m, k], device);
        let s_tensor = Tensor::<CpuRuntime>::from_slice(&s_sorted, &[k], device);
        let vt_tensor = Tensor::<CpuRuntime>::from_slice(&vt_final, &[k, n], device);

        Ok(SvdDecomposition {
            u: u_tensor,
            s: s_tensor,
            vt: vt_tensor,
        })
    } else {
        let u_tensor = Tensor::<CpuRuntime>::from_slice(&u_sorted, &[m, k], device);
        let s_tensor = Tensor::<CpuRuntime>::from_slice(&s_sorted, &[k], device);
        let vt_tensor = Tensor::<CpuRuntime>::from_slice(&vt_sorted, &[k, n], device);

        Ok(SvdDecomposition {
            u: u_tensor,
            s: s_tensor,
            vt: vt_tensor,
        })
    }
}

/// Eigendecomposition for symmetric matrices using Jacobi algorithm
///
/// Algorithm: Jacobi Eigenvalue Algorithm
/// 1. Initialize: V = I_n (eigenvector matrix starts as identity)
/// 2. REPEAT (max 30 sweeps):
///    FOR each pair (p, q) where p < q:
///      - If |A[p,q]| > tol:
///        a. Compute Jacobi rotation angle θ from A[p,p], A[q,q], A[p,q]
///        b. Apply rotation: A' = J^T @ A @ J (zeros out A[p,q] and A[q,p])
///        c. Update eigenvectors: V = V @ J
///      - Check convergence: max(|A[i,j]| for i≠j) < n * epsilon
/// 3. eigenvalues = diag(A) (diagonal elements after convergence)
/// 4. Sort eigenvalues descending by magnitude, reorder eigenvector columns
fn eig_decompose_symmetric_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<EigenDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle empty matrix
    if n == 0 {
        let eigenvalues = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[0], device);
        let eigenvectors = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[0, 0], device);
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    // Handle 1x1 matrix
    if n == 1 {
        let a_data: Vec<T> = a.to_vec();
        let eigenvalues = Tensor::<CpuRuntime>::from_slice(&[a_data[0]], &[1], device);
        let eigenvectors = Tensor::<CpuRuntime>::from_slice(&[T::one()], &[1, 1], device);
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    // Copy input to working matrix (will be modified in-place)
    // We symmetrize by using only lower triangle: A[i,j] = A[j,i] for i > j
    let a_data: Vec<T> = a.to_vec();
    let mut work: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..=i {
            // Use lower triangular part
            let val = a_data[i * n + j];
            work[i * n + j] = val;
            work[j * n + i] = val;
        }
    }

    // Initialize eigenvector matrix V as identity
    let mut v: Vec<T> = identity_matrix(n);

    // Convergence parameters
    let eps = T::epsilon_val();
    let tol = (n as f64) * eps;
    let max_sweeps = 30;

    // Jacobi iterations
    for _sweep in 0..max_sweeps {
        // Find maximum off-diagonal element
        let mut max_off_diag = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = work[i * n + j].abs_val().to_f64();
                if val > max_off_diag {
                    max_off_diag = val;
                }
            }
        }

        // Check convergence
        if max_off_diag < tol {
            break;
        }

        // Process all element pairs (p, q) where p < q
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = work[p * n + q];

                // Skip if already essentially zero
                if a_pq.abs_val().to_f64() < tol {
                    continue;
                }

                let a_pp = work[p * n + p];
                let a_qq = work[q * n + q];

                // Compute Jacobi rotation using stable LAPACK formula
                let rot = JacobiRotation::compute(a_pp.to_f64(), a_qq.to_f64(), a_pq.to_f64());

                // Apply two-sided rotation to work matrix: A' = J^T @ A @ J
                apply_two_sided_rotation(&mut work, n, p, q, &rot, a_pp, a_qq, a_pq);

                // Update eigenvector matrix: V = V @ J
                apply_rotation_to_columns(&mut v, n, n, p, q, &rot);
            }
        }
    }

    // Extract eigenvalues (diagonal of converged matrix)
    let mut eigenvalues: Vec<T> = vec![T::zero(); n];
    for i in 0..n {
        eigenvalues[i] = work[i * n + i];
    }

    // Sort eigenvalues by magnitude (descending) and reorder eigenvectors
    let indices = argsort_by_magnitude_desc(&eigenvalues);

    // Reorder eigenvalues and eigenvector columns
    let eigenvalues_sorted = jacobi::permute_vector(&eigenvalues, &indices);
    let v_sorted = permute_columns(&v, n, n, &indices, n);

    let eigenvalues_tensor = Tensor::<CpuRuntime>::from_slice(&eigenvalues_sorted, &[n], device);
    let eigenvectors_tensor = Tensor::<CpuRuntime>::from_slice(&v_sorted, &[n, n], device);

    Ok(EigenDecomposition {
        eigenvalues: eigenvalues_tensor,
        eigenvectors: eigenvectors_tensor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    fn create_client() -> CpuClient {
        let device = super::super::CpuDevice::new();
        CpuClient::new(device)
    }

    #[test]
    fn test_lu_decomposition_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[4, 3], [6, 3]]
        let a = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device);

        let lu = client.lu_decompose(&a).unwrap();

        // Verify dimensions
        assert_eq!(lu.lu.shape(), &[2, 2]);
        assert_eq!(lu.pivots.shape(), &[2]);
    }

    #[test]
    fn test_lu_decomposition_3x3() {
        let client = create_client();
        let device = client.device();

        // A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] (tridiagonal matrix)
        let a = Tensor::<CpuRuntime>::from_slice(
            &[2.0f32, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
            &[3, 3],
            device,
        );

        let result = client.lu_decompose(&a);
        assert!(result.is_ok());

        let lu = result.unwrap();
        assert_eq!(lu.lu.shape(), &[3, 3]);
        assert_eq!(lu.pivots.shape(), &[3]);
    }

    #[test]
    fn test_solve_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[2, 1], [1, 2]], b = [3, 3]
        // Solution: x = [1, 1]
        let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device);
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 3.0], &[2], device);

        let x = client.solve(&a, &b).unwrap();
        let x_data: Vec<f32> = x.to_vec();

        // Check solution is approximately [1, 1]
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_det_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[4, 3], [6, 3]]
        // det = 4*3 - 3*6 = 12 - 18 = -6
        let a = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device);

        let det = client.det(&a).unwrap();
        let det_val: Vec<f32> = det.to_vec();

        assert!((det_val[0] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_trace() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]]
        // trace = 1 + 4 = 5
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let tr = client.trace(&a).unwrap();
        let tr_val: Vec<f32> = tr.to_vec();

        assert!((tr_val[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_cholesky_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[4, 2], [2, 2]] - symmetric positive definite
        // L = [[2, 0], [1, 1]]
        let a = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 2.0, 2.0, 2.0], &[2, 2], device);

        let chol = client.cholesky_decompose(&a).unwrap();
        let l_data: Vec<f32> = chol.l.to_vec();

        // Check L is approximately [[2, 0], [1, 1]]
        assert!((l_data[0] - 2.0).abs() < 1e-5); // L[0,0]
        assert!((l_data[1]).abs() < 1e-5); // L[0,1] = 0
        assert!((l_data[2] - 1.0).abs() < 1e-5); // L[1,0]
        assert!((l_data[3] - 1.0).abs() < 1e-5); // L[1,1]
    }

    #[test]
    fn test_inverse_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[4, 7], [2, 6]]
        // det = 24 - 14 = 10
        // A^(-1) = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        let a = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device);

        let inv = client.inverse(&a).unwrap();
        let inv_data: Vec<f32> = inv.to_vec();

        assert!((inv_data[0] - 0.6).abs() < 1e-4);
        assert!((inv_data[1] - (-0.7)).abs() < 1e-4);
        assert!((inv_data[2] - (-0.2)).abs() < 1e-4);
        assert!((inv_data[3] - 0.4).abs() < 1e-4);
    }

    #[test]
    fn test_diag() {
        let client = create_client();
        let device = client.device();

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device);

        let d = client.diag(&a).unwrap();
        let d_data: Vec<f32> = d.to_vec();

        assert_eq!(d_data.len(), 2); // min(2, 3)
        assert!((d_data[0] - 1.0).abs() < 1e-5);
        assert!((d_data[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diagflat() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device);

        let mat = client.diagflat(&a).unwrap();
        let mat_data: Vec<f32> = mat.to_vec();

        // Should be 3x3 with [1, 2, 3] on diagonal
        assert_eq!(mat.shape(), &[3, 3]);
        assert!((mat_data[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((mat_data[4] - 2.0).abs() < 1e-5); // [1,1]
        assert!((mat_data[8] - 3.0).abs() < 1e-5); // [2,2]
        // Off-diagonal should be zero
        assert!((mat_data[1]).abs() < 1e-5);
        assert!((mat_data[2]).abs() < 1e-5);
    }

    #[test]
    fn test_qr_decomposition_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let qr = client.qr_decompose(&a).unwrap();

        // Check dimensions
        assert_eq!(qr.q.shape(), &[2, 2]);
        assert_eq!(qr.r.shape(), &[2, 2]);

        // Q should be orthogonal: Q^T @ Q ≈ I
        let q_data: Vec<f32> = qr.q.to_vec();
        // Check Q^T @ Q diagonal is ~1 and off-diagonal is ~0
        let q00 = q_data[0];
        let q01 = q_data[1];
        let q10 = q_data[2];
        let q11 = q_data[3];

        let qtq_00 = q00 * q00 + q10 * q10; // Should be 1
        let qtq_11 = q01 * q01 + q11 * q11; // Should be 1
        let qtq_01 = q00 * q01 + q10 * q11; // Should be 0

        assert!(
            (qtq_00 - 1.0).abs() < 1e-4,
            "Q^T@Q[0,0] = {} should be 1",
            qtq_00
        );
        assert!(
            (qtq_11 - 1.0).abs() < 1e-4,
            "Q^T@Q[1,1] = {} should be 1",
            qtq_11
        );
        assert!((qtq_01).abs() < 1e-4, "Q^T@Q[0,1] = {} should be 0", qtq_01);

        // R should be upper triangular (R[1,0] = 0)
        let r_data: Vec<f32> = qr.r.to_vec();
        assert!((r_data[2]).abs() < 1e-4, "R[1,0] should be 0");
    }

    #[test]
    fn test_lstsq_exact() {
        let client = create_client();
        let device = client.device();

        // Exact system: A @ x = b with unique solution
        // A = [[2, 1], [1, 2]], b = [3, 3]
        // Solution: x = [1, 1]
        let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device);
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 3.0], &[2], device);

        let x = client.lstsq(&a, &b).unwrap();
        let x_data: Vec<f32> = x.to_vec();

        // Solution should be approximately [1, 1]
        assert!(
            (x_data[0] - 1.0).abs() < 1e-4,
            "x[0] = {} should be 1.0",
            x_data[0]
        );
        assert!(
            (x_data[1] - 1.0).abs() < 1e-4,
            "x[1] = {} should be 1.0",
            x_data[1]
        );
    }

    #[test]
    fn test_matrix_rank_full_rank() {
        let client = create_client();
        let device = client.device();

        // Full rank 2x2 matrix
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let rank = client.matrix_rank(&a, None).unwrap();
        let rank_val: Vec<i64> = rank.to_vec();

        assert_eq!(rank_val[0], 2, "Full rank 2x2 matrix should have rank 2");
    }

    #[test]
    fn test_matrix_rank_rank_deficient() {
        let client = create_client();
        let device = client.device();

        // Rank-deficient 2x2 matrix: second row is multiple of first
        // [[1, 2], [2, 4]] has rank 1
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device);

        let rank = client.matrix_rank(&a, None).unwrap();
        let rank_val: Vec<i64> = rank.to_vec();

        assert_eq!(rank_val[0], 1, "Rank-deficient matrix should have rank 1");
    }

    #[test]
    fn test_frobenius_norm_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]]
        // ||A||_F = sqrt(1² + 2² + 3² + 4²) = sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let norm = client.matrix_norm(&a, MatrixNormOrder::Frobenius).unwrap();
        let norm_val: Vec<f32> = norm.to_vec();

        let expected = (30.0f32).sqrt();
        assert!(
            (norm_val[0] - expected).abs() < 1e-5,
            "Frobenius norm = {} should be {}",
            norm_val[0],
            expected
        );
    }

    #[test]
    fn test_frobenius_norm_3x3() {
        let client = create_client();
        let device = client.device();

        // Identity matrix: ||I||_F = sqrt(3) for 3x3
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            device,
        );

        let norm = client.matrix_norm(&a, MatrixNormOrder::Frobenius).unwrap();
        let norm_val: Vec<f32> = norm.to_vec();

        let expected = (3.0f32).sqrt();
        assert!(
            (norm_val[0] - expected).abs() < 1e-5,
            "Frobenius norm of 3x3 identity = {} should be {}",
            norm_val[0],
            expected
        );
    }

    #[test]
    fn test_spectral_norm_not_implemented() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let result = client.matrix_norm(&a, MatrixNormOrder::Spectral);
        assert!(
            result.is_err(),
            "Spectral norm should not be implemented yet"
        );
    }
}
