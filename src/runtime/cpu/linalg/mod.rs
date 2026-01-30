//! CPU implementation of linear algebra algorithms
//!
//! This module implements the [`LinearAlgebraAlgorithms`] trait for CPU.
//! All algorithms follow the exact specification in the trait documentation
//! to ensure backend parity with CUDA/WebGPU implementations.

mod advanced;
mod decompositions;
mod eig_general;
mod eig_symmetric;
mod matrix_functions;
mod matrix_ops;
mod schur;
mod solvers;
mod statistics;
mod svd;

#[cfg(test)]
mod tests;

use super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    CholeskyDecomposition, ComplexSchurDecomposition, EigenDecomposition,
    GeneralEigenDecomposition, GeneralizedSchurDecomposition, LinearAlgebraAlgorithms,
    LuDecomposition, MatrixFunctionsAlgorithms, MatrixNormOrder, PolarDecomposition,
    QrDecomposition, SchurDecomposition, SvdDecomposition, validate_linalg_dtype,
    validate_matrix_2d, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl LinearAlgebraAlgorithms<CpuRuntime> for CpuClient {
    fn lu_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<LuDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => decompositions::lu_decompose_impl::<f32>(self, a, m, n),
            DType::F64 => decompositions::lu_decompose_impl::<f64>(self, a, m, n),
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
            DType::F32 => decompositions::cholesky_decompose_impl::<f32>(self, a, n),
            DType::F64 => decompositions::cholesky_decompose_impl::<f64>(self, a, n),
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
            DType::F32 => decompositions::qr_decompose_impl::<f32>(self, a, m, n, false),
            DType::F64 => decompositions::qr_decompose_impl::<f64>(self, a, m, n, false),
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
            DType::F32 => decompositions::qr_decompose_impl::<f32>(self, a, m, n, true),
            DType::F64 => decompositions::qr_decompose_impl::<f64>(self, a, m, n, true),
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
            DType::F32 => solvers::solve_impl::<f32>(self, a, b, n),
            DType::F64 => solvers::solve_impl::<f64>(self, a, b, n),
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
            DType::F32 => solvers::solve_triangular_lower_impl::<f32>(self, l, b, n, unit_diagonal),
            DType::F64 => solvers::solve_triangular_lower_impl::<f64>(self, l, b, n, unit_diagonal),
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
            DType::F32 => solvers::solve_triangular_upper_impl::<f32>(self, u, b, n),
            DType::F64 => solvers::solve_triangular_upper_impl::<f64>(self, u, b, n),
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
            DType::F32 => solvers::lstsq_impl::<f32>(self, a, b, m, n),
            DType::F64 => solvers::lstsq_impl::<f64>(self, a, b, m, n),
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
            DType::F32 => matrix_ops::inverse_impl::<f32>(self, a, n),
            DType::F64 => matrix_ops::inverse_impl::<f64>(self, a, n),
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
            DType::F32 => matrix_ops::det_impl::<f32>(self, a, n),
            DType::F64 => matrix_ops::det_impl::<f64>(self, a, n),
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
            DType::F32 => matrix_ops::trace_impl::<f32>(self, a, m, n),
            DType::F64 => matrix_ops::trace_impl::<f64>(self, a, m, n),
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
            DType::F32 => matrix_ops::diag_impl::<f32>(self, a, m, n),
            DType::F64 => matrix_ops::diag_impl::<f64>(self, a, m, n),
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
            DType::F32 => matrix_ops::diagflat_impl::<f32>(self, a),
            DType::F64 => matrix_ops::diagflat_impl::<f64>(self, a),
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
            DType::F32 => matrix_ops::matrix_rank_impl::<f32>(self, a, m, n, tol),
            DType::F64 => matrix_ops::matrix_rank_impl::<f64>(self, a, m, n, tol),
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
                DType::F32 => matrix_ops::frobenius_norm_impl::<f32>(self, a),
                DType::F64 => matrix_ops::frobenius_norm_impl::<f64>(self, a),
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
            DType::F32 => svd::svd_decompose_impl::<f32>(self, a, m, n),
            DType::F64 => svd::svd_decompose_impl::<f64>(self, a, m, n),
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
            DType::F32 => eig_symmetric::eig_decompose_symmetric_impl::<f32>(self, a, n),
            DType::F64 => eig_symmetric::eig_decompose_symmetric_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "eig_decompose_symmetric",
            }),
        }
    }

    fn pinverse(&self, a: &Tensor<CpuRuntime>, rcond: Option<f64>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => statistics::pinverse_impl::<f32>(self, a, m, n, rcond),
            DType::F64 => statistics::pinverse_impl::<f64>(self, a, m, n, rcond),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "pinverse",
            }),
        }
    }

    fn cond(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (_m, _n) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => statistics::cond_impl::<f32>(self, a),
            DType::F64 => statistics::cond_impl::<f64>(self, a),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "cond",
            }),
        }
    }

    fn cov(&self, a: &Tensor<CpuRuntime>, ddof: Option<usize>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (n_samples, n_features) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => statistics::cov_impl::<f32>(self, a, n_samples, n_features, ddof),
            DType::F64 => statistics::cov_impl::<f64>(self, a, n_samples, n_features, ddof),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "cov",
            }),
        }
    }

    fn corrcoef(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (n_samples, n_features) = validate_matrix_2d(a.shape())?;

        match a.dtype() {
            DType::F32 => statistics::corrcoef_impl::<f32>(self, a, n_samples, n_features),
            DType::F64 => statistics::corrcoef_impl::<f64>(self, a, n_samples, n_features),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "corrcoef",
            }),
        }
    }

    fn schur_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<SchurDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => schur::schur_decompose_impl::<f32>(self, a, n),
            DType::F64 => schur::schur_decompose_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "schur_decompose",
            }),
        }
    }

    fn eig_decompose(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<GeneralEigenDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => eig_general::eig_decompose_impl::<f32>(self, a, n),
            DType::F64 => eig_general::eig_decompose_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "eig_decompose",
            }),
        }
    }

    fn rsf2csf(
        &self,
        schur: &SchurDecomposition<CpuRuntime>,
    ) -> Result<ComplexSchurDecomposition<CpuRuntime>> {
        validate_linalg_dtype(schur.t.dtype())?;
        let n = validate_square_matrix(schur.t.shape())?;

        match schur.t.dtype() {
            DType::F32 => advanced::rsf2csf_impl::<f32>(self, schur, n),
            DType::F64 => advanced::rsf2csf_impl::<f64>(self, schur, n),
            _ => Err(Error::UnsupportedDType {
                dtype: schur.t.dtype(),
                op: "rsf2csf",
            }),
        }
    }

    fn qz_decompose(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<GeneralizedSchurDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        let n_a = validate_square_matrix(a.shape())?;
        let n_b = validate_square_matrix(b.shape())?;
        if n_a != n_b {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        match a.dtype() {
            DType::F32 => advanced::qz_decompose_impl::<f32>(self, a, b, n_a),
            DType::F64 => advanced::qz_decompose_impl::<f64>(self, a, b, n_a),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "qz_decompose",
            }),
        }
    }

    fn polar_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<PolarDecomposition<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => advanced::polar_decompose_impl::<f32>(self, a, n),
            DType::F64 => advanced::polar_decompose_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "polar_decompose",
            }),
        }
    }
}

impl MatrixFunctionsAlgorithms<CpuRuntime> for CpuClient {
    fn expm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => matrix_functions::expm_impl::<f32>(self, a, n),
            DType::F64 => matrix_functions::expm_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "expm",
            }),
        }
    }

    fn logm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => matrix_functions::logm_impl::<f32>(self, a, n),
            DType::F64 => matrix_functions::logm_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "logm",
            }),
        }
    }

    fn sqrtm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => matrix_functions::sqrtm_impl::<f32>(self, a, n),
            DType::F64 => matrix_functions::sqrtm_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "sqrtm",
            }),
        }
    }

    fn signm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => matrix_functions::signm_impl::<f32>(self, a, n),
            DType::F64 => matrix_functions::signm_impl::<f64>(self, a, n),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "signm",
            }),
        }
    }

    fn fractional_matrix_power(
        &self,
        a: &Tensor<CpuRuntime>,
        p: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => matrix_functions::fractional_matrix_power_impl::<f32>(self, a, n, p),
            DType::F64 => matrix_functions::fractional_matrix_power_impl::<f64>(self, a, n, p),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "fractional_matrix_power",
            }),
        }
    }

    fn funm<F>(&self, a: &Tensor<CpuRuntime>, f: F) -> Result<Tensor<CpuRuntime>>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;

        match a.dtype() {
            DType::F32 => matrix_functions::funm_impl::<f32, F>(self, a, n, f),
            DType::F64 => matrix_functions::funm_impl::<f64, F>(self, a, n, f),
            _ => Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "funm",
            }),
        }
    }
}
