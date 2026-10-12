//! Shape rules for batched triangular solves.
//!
//! Every backend calls [`validate_triangular_solve`] so the accepted shapes
//! and the error for each rejected shape are the same on CPU, CUDA and WebGPU.

use super::helpers::validate_square_matrix;
use crate::error::{Error, Result};

/// Resolved shape of a triangular solve `A X = B`.
///
/// `A` is `[n, n]` or `[..batch, n, n]`. Each matrix solves against
/// `nrhs` columns. The data layout is row-major:
/// matrix `i` of `A` starts at element `i * n * n`, and matrix `i` of `B`
/// and `X` starts at element `i * n * nrhs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriangularSolveShape {
    /// Number of matrices: the product of the batch dims, 1 for a 2-D `A`.
    pub batch: usize,
    /// Order of each triangular matrix.
    pub n: usize,
    /// Right-hand-side columns per matrix, 1 for a 1-D `B`.
    pub nrhs: usize,
    /// Shape of `X`, equal to the shape of `B`.
    pub out_shape: Vec<usize>,
}

impl TriangularSolveShape {
    /// Number of elements in `X`.
    pub fn out_numel(&self) -> usize {
        self.batch * self.n * self.nrhs
    }
}

/// Check the shapes of a triangular solve and resolve its batch layout.
///
/// Accepted shapes:
/// - `A: [n, n]` with `B: [n]` or `B: [n, m]`.
/// - `A: [..batch, n, n]` with `B: [..batch, n, m]`. The batch dims of `B`
///   equal those of `A` exactly. No broadcasting applies.
///
/// `op` names the operation in the error text.
///
/// # Errors
///
/// - `A` with fewer than 2 dims, or non-square trailing dims.
/// - A 2-D `A` with `B` of 3 or more dims, or a row count other than `n`.
/// - A batched `A` with `B` of another rank, other batch dims, or a row
///   count other than `n`. The error names both shapes.
pub fn validate_triangular_solve(
    op: &'static str,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<TriangularSolveShape> {
    if a_shape.len() <= 2 {
        let n = validate_square_matrix(a_shape)?;
        let nrhs = match b_shape.len() {
            1 => {
                if b_shape[0] != n {
                    return Err(Error::ShapeMismatch {
                        expected: vec![n],
                        got: b_shape.to_vec(),
                    });
                }
                1
            }
            2 => {
                if b_shape[0] != n {
                    return Err(Error::ShapeMismatch {
                        expected: vec![n, b_shape[1]],
                        got: b_shape.to_vec(),
                    });
                }
                b_shape[1]
            }
            _ => {
                return Err(Error::Internal(format!(
                    "{op} requires b to be 1D or 2D, got {}D with shape {:?}",
                    b_shape.len(),
                    b_shape
                )));
            }
        };
        return Ok(TriangularSolveShape {
            batch: 1,
            n,
            nrhs,
            out_shape: b_shape.to_vec(),
        });
    }

    let rank = a_shape.len();
    let (rows, cols) = (a_shape[rank - 2], a_shape[rank - 1]);
    if rows != cols {
        return Err(Error::ShapeMismatch {
            expected: vec![rows, rows],
            got: vec![rows, cols],
        });
    }
    let n = rows;
    let batch_dims = &a_shape[..rank - 2];
    if b_shape.len() != rank || &b_shape[..rank - 2] != batch_dims || b_shape[rank - 2] != n {
        return Err(Error::InvalidArgument {
            arg: "b",
            reason: format!(
                "{op}: b has shape {b_shape:?} and a has shape {a_shape:?}. \
                 b must have rank {rank}, batch dims {batch_dims:?} and {n} rows"
            ),
        });
    }
    Ok(TriangularSolveShape {
        batch: batch_dims.iter().product(),
        n,
        nrhs: b_shape[rank - 1],
        out_shape: b_shape.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const OP: &str = "solve_triangular_lower";

    #[test]
    fn single_matrix_vector_rhs() {
        let s = validate_triangular_solve(OP, &[4, 4], &[4]).unwrap();
        assert_eq!((s.batch, s.n, s.nrhs), (1, 4, 1));
        assert_eq!(s.out_shape, vec![4]);
    }

    #[test]
    fn single_matrix_matrix_rhs() {
        let s = validate_triangular_solve(OP, &[4, 4], &[4, 3]).unwrap();
        assert_eq!((s.batch, s.n, s.nrhs), (1, 4, 3));
        assert_eq!(s.out_numel(), 12);
    }

    #[test]
    fn batched_two_dims() {
        let s = validate_triangular_solve(OP, &[2, 3, 5, 5], &[2, 3, 5, 7]).unwrap();
        assert_eq!((s.batch, s.n, s.nrhs), (6, 5, 7));
        assert_eq!(s.out_shape, vec![2, 3, 5, 7]);
    }

    #[test]
    fn batched_zero_batch() {
        let s = validate_triangular_solve(OP, &[0, 5, 5], &[0, 5, 2]).unwrap();
        assert_eq!((s.batch, s.n, s.nrhs), (0, 5, 2));
        assert_eq!(s.out_numel(), 0);
    }

    #[test]
    fn single_matrix_errors_unchanged() {
        assert!(matches!(
            validate_triangular_solve(OP, &[3, 3], &[4]),
            Err(Error::ShapeMismatch { .. })
        ));
        assert!(matches!(
            validate_triangular_solve(OP, &[3, 3], &[4, 2]),
            Err(Error::ShapeMismatch { .. })
        ));
        assert!(matches!(
            validate_triangular_solve(OP, &[3, 3], &[1, 3, 2]),
            Err(Error::Internal(_))
        ));
        assert!(matches!(
            validate_triangular_solve(OP, &[3, 2], &[3]),
            Err(Error::ShapeMismatch { .. })
        ));
        assert!(matches!(
            validate_triangular_solve(OP, &[3], &[3]),
            Err(Error::Internal(_))
        ));
    }

    #[test]
    fn batched_mismatch_names_both_shapes() {
        for b in [&[3, 4, 2][..], &[2, 5, 2], &[2, 4], &[1, 2, 4, 2]] {
            let err = validate_triangular_solve(OP, &[2, 4, 4], b).unwrap_err();
            let text = err.to_string();
            assert!(text.contains(&format!("{b:?}")), "{text}");
            assert!(text.contains("[2, 4, 4]"), "{text}");
        }
    }

    #[test]
    fn batched_non_square_rejected() {
        assert!(matches!(
            validate_triangular_solve(OP, &[2, 4, 3], &[2, 4, 1]),
            Err(Error::ShapeMismatch { .. })
        ));
    }
}
