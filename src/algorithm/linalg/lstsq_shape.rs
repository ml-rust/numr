//! Shape rules for least-squares solves.
//!
//! The generic least-squares solver calls [`validate_lstsq`], so every backend
//! accepts the same shapes and returns the same error for each rejected shape.

use crate::error::{Error, Result};

/// Resolved shape of a least-squares solve `min ||A X - B||`.
///
/// `A` is `[m, n]` or `[..batch, m, n]`. The data layout is row-major:
/// matrix `i` of `A` starts at element `i * m * n`, and matrix `i` of `B`
/// starts at element `i * m * nrhs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LstsqShape {
    /// Leading batch dims of `A`, empty for a 2-D `A`.
    pub batch_dims: Vec<usize>,
    /// Number of matrices: the product of the batch dims, 1 for a 2-D `A`.
    pub batch: usize,
    /// Rows of each matrix.
    pub m: usize,
    /// Columns of each matrix.
    pub n: usize,
    /// Right-hand-side columns per matrix, 1 for a 1-D `B`.
    pub nrhs: usize,
    /// Whether `B` is a 1-D vector.
    pub b_is_vector: bool,
}

impl LstsqShape {
    /// Shape of `X`: `[n]`, `[n, nrhs]` or `[..batch, n, nrhs]`.
    pub fn solution_shape(&self) -> Vec<usize> {
        if self.b_is_vector {
            return vec![self.n];
        }
        let mut shape = self.batch_dims.clone();
        shape.extend([self.n, self.nrhs]);
        shape
    }

    /// Shape of the singular values: `[min(m, n)]` or `[..batch, min(m, n)]`.
    pub fn singular_values_shape(&self) -> Vec<usize> {
        let mut shape = self.batch_dims.clone();
        shape.push(self.m.min(self.n));
        shape
    }

    /// Shape of the rank: a scalar, or `[..batch]` for a batched `A`.
    pub fn rank_shape(&self) -> Vec<usize> {
        self.batch_dims.clone()
    }
}

/// Check the shapes of a least-squares solve and resolve its batch layout.
///
/// Accepted shapes:
/// - `A: [m, n]` with `B: [m]` or `B: [m, k]`.
/// - `A: [..batch, m, n]` with `B: [..batch, m, k]`. The batch dims of `B`
///   equal those of `A` exactly. No broadcasting applies.
///
/// Any `m`, `n` or `k` is accepted, zero included.
///
/// # Errors
///
/// - `InvalidArgument` for `A` with fewer than 2 dims.
/// - `ShapeMismatch` for `B` with the wrong rank, other batch dims or a row
///   count other than `m`. The error holds the expected and the actual `B`
///   shape.
pub fn validate_lstsq(a_shape: &[usize], b_shape: &[usize]) -> Result<LstsqShape> {
    let rank = a_shape.len();
    if rank < 2 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!("lstsq requires a with at least 2 dims, got shape {a_shape:?}"),
        });
    }
    let (m, n) = (a_shape[rank - 2], a_shape[rank - 1]);
    let batch_dims = a_shape[..rank - 2].to_vec();
    let b_cols = if b_shape.len() >= 2 {
        b_shape[b_shape.len() - 1]
    } else {
        1
    };

    if rank == 2 && b_shape.len() == 1 {
        if b_shape[0] != m {
            return Err(Error::ShapeMismatch {
                expected: vec![m],
                got: b_shape.to_vec(),
            });
        }
        return Ok(LstsqShape {
            batch_dims,
            batch: 1,
            m,
            n,
            nrhs: 1,
            b_is_vector: true,
        });
    }

    let mut expected = batch_dims.clone();
    expected.extend([m, b_cols]);
    if b_shape != expected.as_slice() {
        return Err(Error::ShapeMismatch {
            expected,
            got: b_shape.to_vec(),
        });
    }
    Ok(LstsqShape {
        batch: batch_dims.iter().product(),
        batch_dims,
        m,
        n,
        nrhs: b_cols,
        b_is_vector: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_matrix_vector_rhs() {
        let s = validate_lstsq(&[5, 3], &[5]).unwrap();
        assert_eq!(
            (s.batch, s.m, s.n, s.nrhs, s.b_is_vector),
            (1, 5, 3, 1, true)
        );
        assert_eq!(s.solution_shape(), vec![3]);
        assert_eq!(s.singular_values_shape(), vec![3]);
        assert!(s.rank_shape().is_empty());
    }

    #[test]
    fn single_matrix_matrix_rhs() {
        let s = validate_lstsq(&[3, 5], &[3, 2]).unwrap();
        assert_eq!(
            (s.batch, s.m, s.n, s.nrhs, s.b_is_vector),
            (1, 3, 5, 2, false)
        );
        assert_eq!(s.solution_shape(), vec![5, 2]);
        assert_eq!(s.singular_values_shape(), vec![3]);
    }

    #[test]
    fn batched_two_dims() {
        let s = validate_lstsq(&[2, 3, 4, 2], &[2, 3, 4, 5]).unwrap();
        assert_eq!((s.batch, s.m, s.n, s.nrhs), (6, 4, 2, 5));
        assert_eq!(s.solution_shape(), vec![2, 3, 2, 5]);
        assert_eq!(s.singular_values_shape(), vec![2, 3, 2]);
        assert_eq!(s.rank_shape(), vec![2, 3]);
    }

    #[test]
    fn empty_extents_accepted() {
        let s = validate_lstsq(&[0, 3], &[0]).unwrap();
        assert_eq!(s.solution_shape(), vec![3]);
        assert_eq!(s.singular_values_shape(), vec![0]);
        let s = validate_lstsq(&[3, 0], &[3, 2]).unwrap();
        assert_eq!(s.solution_shape(), vec![0, 2]);
        let s = validate_lstsq(&[0, 4, 3], &[0, 4, 1]).unwrap();
        assert_eq!(s.batch, 0);
        assert_eq!(s.solution_shape(), vec![0, 3, 1]);
    }

    #[test]
    fn wrong_rows_rejected() {
        let err = validate_lstsq(&[3, 2], &[4]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
        let err = validate_lstsq(&[3, 2], &[4, 2]).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("[3, 2]") && text.contains("[4, 2]"), "{text}");
    }

    #[test]
    fn wrong_b_rank_rejected() {
        for b in [&[3, 2, 1][..], &[]] {
            let err = validate_lstsq(&[3, 2], b).unwrap_err();
            assert!(matches!(err, Error::ShapeMismatch { .. }), "{b:?}");
        }
        let err = validate_lstsq(&[2, 4, 3], &[4]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn batched_mismatch_names_b_shape() {
        for b in [&[3, 4, 2][..], &[2, 5, 2], &[2, 4], &[1, 2, 4, 2]] {
            let err = validate_lstsq(&[2, 4, 3], b).unwrap_err();
            assert!(matches!(err, Error::ShapeMismatch { .. }), "{b:?}");
            let text = err.to_string();
            assert!(text.contains(&format!("{b:?}")), "{text}");
        }
    }

    #[test]
    fn low_rank_a_rejected() {
        let err = validate_lstsq(&[3], &[3]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { arg: "a", .. }));
        assert!(err.to_string().contains("[3]"));
    }
}
