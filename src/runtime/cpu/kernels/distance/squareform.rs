//! Conversion between the condensed distance vector and the square matrix.
//!
//! These kernels move already-computed distances. There is no running total and
//! no accumulator: the element type carries the values unchanged.

use crate::dtype::Element;
use num_traits::{Float, Zero};

/// Convert condensed distance vector to square matrix.
///
/// # Safety
///
/// - `condensed` must point to valid data of length `n * (n - 1) / 2`
/// - `square` must point to valid memory of length `n * n`
#[inline]
pub unsafe fn squareform_kernel<T: Element + Float>(condensed: *const T, square: *mut T, n: usize) {
    // Fill diagonal with zeros
    for i in 0..n {
        *square.add(i * n + i) = <T as Zero>::zero();
    }

    // Fill upper and lower triangles
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let val = *condensed.add(k);
            *square.add(i * n + j) = val;
            *square.add(j * n + i) = val;
            k += 1;
        }
    }
}

/// Convert square distance matrix to condensed form.
///
/// # Safety
///
/// - `square` must point to valid data of length `n * n`
/// - `condensed` must point to valid memory of length `n * (n - 1) / 2`
#[inline]
pub unsafe fn squareform_inverse_kernel<T: Element + Float>(
    square: *const T,
    condensed: *mut T,
    n: usize,
) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            *condensed.add(k) = *square.add(i * n + j);
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squareform_expands_the_condensed_vector() {
        let condensed = [1.0f32, 2.0, 3.0]; // d(0,1), d(0,2), d(1,2)
        let mut square = [0.0f32; 9];

        unsafe {
            squareform_kernel(condensed.as_ptr(), square.as_mut_ptr(), 3);
        }

        // Expected:
        // [[0, 1, 2],
        //  [1, 0, 3],
        //  [2, 3, 0]]
        assert_eq!(square, [0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn squareform_inverse_recovers_the_condensed_vector() {
        let square = [0.0f32, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let mut condensed = [0.0f32; 3];

        unsafe {
            squareform_inverse_kernel(square.as_ptr(), condensed.as_mut_ptr(), 3);
        }

        assert_eq!(condensed, [1.0, 2.0, 3.0]);
    }
}
