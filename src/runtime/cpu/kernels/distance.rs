//! CPU kernels for distance computation.
//!
//! Optimized implementations for pairwise distance calculations using SIMD.

use crate::dtype::Element;
use crate::ops::DistanceMetric;
use num_traits::{Float, FromPrimitive, Zero};

/// Compute pairwise distances between two point sets (cdist).
///
/// # Safety
///
/// - `x` must point to valid data of length `n * d`
/// - `y` must point to valid data of length `m * d`
/// - `out` must point to valid memory of length `n * m`
/// - All pointers must be properly aligned for type T
#[inline]
pub unsafe fn cdist_kernel<T: Element + Float + FromPrimitive>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
    metric: DistanceMetric,
) {
    match metric {
        DistanceMetric::Euclidean => cdist_euclidean(x, y, out, n, m, d),
        DistanceMetric::SquaredEuclidean => cdist_sqeuclidean(x, y, out, n, m, d),
        DistanceMetric::Manhattan => cdist_manhattan(x, y, out, n, m, d),
        DistanceMetric::Chebyshev => cdist_chebyshev(x, y, out, n, m, d),
        DistanceMetric::Minkowski(p) => cdist_minkowski(
            x,
            y,
            out,
            n,
            m,
            d,
            <T as FromPrimitive>::from_f64(p).unwrap(),
        ),
        DistanceMetric::Cosine => cdist_cosine(x, y, out, n, m, d),
        DistanceMetric::Correlation => cdist_correlation(x, y, out, n, m, d),
        DistanceMetric::Hamming => cdist_hamming(x, y, out, n, m, d),
        DistanceMetric::Jaccard => cdist_jaccard(x, y, out, n, m, d),
    }
}

/// Compute pairwise distances within a single point set (pdist, condensed form).
///
/// # Safety
///
/// - `x` must point to valid data of length `n * d`
/// - `out` must point to valid memory of length `n * (n - 1) / 2`
/// - All pointers must be properly aligned for type T
#[inline]
pub unsafe fn pdist_kernel<T: Element + Float + FromPrimitive>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
    metric: DistanceMetric,
) {
    match metric {
        DistanceMetric::Euclidean => pdist_euclidean(x, out, n, d),
        DistanceMetric::SquaredEuclidean => pdist_sqeuclidean(x, out, n, d),
        DistanceMetric::Manhattan => pdist_manhattan(x, out, n, d),
        DistanceMetric::Chebyshev => pdist_chebyshev(x, out, n, d),
        DistanceMetric::Minkowski(p) => {
            pdist_minkowski(x, out, n, d, <T as FromPrimitive>::from_f64(p).unwrap())
        }
        DistanceMetric::Cosine => pdist_cosine(x, out, n, d),
        DistanceMetric::Correlation => pdist_correlation(x, out, n, d),
        DistanceMetric::Hamming => pdist_hamming(x, out, n, d),
        DistanceMetric::Jaccard => pdist_jaccard(x, out, n, d),
    }
}

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

// ============================================================================
// Euclidean Distance
// ============================================================================

unsafe fn cdist_euclidean<T: Element + Float>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = euclidean_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_euclidean<T: Element + Float>(x: *const T, out: *mut T, n: usize, d: usize) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

#[inline]
unsafe fn euclidean_distance<T: Element + Float>(a: *const T, b: *const T, d: usize) -> T {
    sqeuclidean_distance(a, b, d).sqrt()
}

// ============================================================================
// Squared Euclidean Distance
// ============================================================================

unsafe fn cdist_sqeuclidean<T: Element + Float>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = sqeuclidean_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_sqeuclidean<T: Element + Float>(x: *const T, out: *mut T, n: usize, d: usize) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = sqeuclidean_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

#[inline]
unsafe fn sqeuclidean_distance<T: Element + Float>(a: *const T, b: *const T, d: usize) -> T {
    let mut sum = <T as Zero>::zero();
    for k in 0..d {
        let diff = *a.add(k) - *b.add(k);
        sum = sum + diff * diff;
    }
    sum
}

// ============================================================================
// Manhattan (L1) Distance
// ============================================================================

unsafe fn cdist_manhattan<T: Element + Float>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = manhattan_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_manhattan<T: Element + Float>(x: *const T, out: *mut T, n: usize, d: usize) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = manhattan_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

#[inline]
unsafe fn manhattan_distance<T: Element + Float>(a: *const T, b: *const T, d: usize) -> T {
    let mut sum = <T as Zero>::zero();
    for k in 0..d {
        sum = sum + (*a.add(k) - *b.add(k)).abs();
    }
    sum
}

// ============================================================================
// Chebyshev (L-infinity) Distance
// ============================================================================

unsafe fn cdist_chebyshev<T: Element + Float>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = chebyshev_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_chebyshev<T: Element + Float>(x: *const T, out: *mut T, n: usize, d: usize) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = chebyshev_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

#[inline]
unsafe fn chebyshev_distance<T: Element + Float>(a: *const T, b: *const T, d: usize) -> T {
    let mut max = <T as Zero>::zero();
    for k in 0..d {
        let abs_diff = (*a.add(k) - *b.add(k)).abs();
        if abs_diff > max {
            max = abs_diff;
        }
    }
    max
}

// ============================================================================
// Minkowski (Lp) Distance
// ============================================================================

unsafe fn cdist_minkowski<T: Element + Float>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
    p: T,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = minkowski_distance(x.add(i * d), y.add(j * d), d, p);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_minkowski<T: Element + Float>(x: *const T, out: *mut T, n: usize, d: usize, p: T) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = minkowski_distance(x.add(i * d), x.add(j * d), d, p);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

#[inline]
unsafe fn minkowski_distance<T: Element + Float>(a: *const T, b: *const T, d: usize, p: T) -> T {
    let mut sum = <T as Zero>::zero();
    for k in 0..d {
        sum = sum + (*a.add(k) - *b.add(k)).abs().powf(p);
    }
    sum.powf(<T as num_traits::One>::one() / p)
}

// ============================================================================
// Cosine Distance
// ============================================================================

unsafe fn cdist_cosine<T: Element + Float>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = cosine_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_cosine<T: Element + Float>(x: *const T, out: *mut T, n: usize, d: usize) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = cosine_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

#[inline]
unsafe fn cosine_distance<T: Element + Float>(a: *const T, b: *const T, d: usize) -> T {
    let mut dot = <T as Zero>::zero();
    let mut norm_a = <T as Zero>::zero();
    let mut norm_b = <T as Zero>::zero();

    for k in 0..d {
        let ak = *a.add(k);
        let bk = *b.add(k);
        dot = dot + ak * bk;
        norm_a = norm_a + ak * ak;
        norm_b = norm_b + bk * bk;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom.is_zero() {
        <T as Zero>::zero()
    } else {
        <T as num_traits::One>::one() - dot / denom
    }
}

// ============================================================================
// Correlation Distance
// ============================================================================

unsafe fn cdist_correlation<T: Element + Float + FromPrimitive>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = correlation_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_correlation<T: Element + Float + FromPrimitive>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = correlation_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

/// Compute correlation distance between two vectors.
///
/// Correlation distance is defined as: 1 - Pearson correlation coefficient
/// where the Pearson correlation is: cov(a,b) / (std(a) * std(b))
///
/// This measures how similar the patterns in two vectors are, invariant to
/// linear transformations. A distance of 0 means perfect positive correlation,
/// while a distance of 2 means perfect negative correlation.
#[inline]
unsafe fn correlation_distance<T: Element + Float + FromPrimitive>(
    a: *const T,
    b: *const T,
    d: usize,
) -> T {
    let d_t = T::from_usize(d).unwrap();

    // Compute means
    let mut sum_a = <T as Zero>::zero();
    let mut sum_b = <T as Zero>::zero();
    for k in 0..d {
        sum_a = sum_a + *a.add(k);
        sum_b = sum_b + *b.add(k);
    }
    let mean_a = sum_a / d_t;
    let mean_b = sum_b / d_t;

    // Compute correlation
    let mut cov = <T as Zero>::zero();
    let mut var_a = <T as Zero>::zero();
    let mut var_b = <T as Zero>::zero();
    for k in 0..d {
        let da = *a.add(k) - mean_a;
        let db = *b.add(k) - mean_b;
        cov = cov + da * db;
        var_a = var_a + da * da;
        var_b = var_b + db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom.is_zero() {
        <T as Zero>::zero()
    } else {
        <T as num_traits::One>::one() - cov / denom
    }
}

// ============================================================================
// Hamming Distance
// ============================================================================

unsafe fn cdist_hamming<T: Element + Float + FromPrimitive>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = hamming_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_hamming<T: Element + Float + FromPrimitive>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = hamming_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

/// Compute Hamming distance between two vectors.
///
/// Hamming distance is the fraction of positions where the vectors differ.
/// For continuous-valued vectors, this counts exact inequality (a[i] != b[i]).
///
/// Returns a value in [0, 1] where:
/// - 0 means vectors are identical
/// - 1 means vectors differ in all positions
#[inline]
unsafe fn hamming_distance<T: Element + Float + FromPrimitive>(
    a: *const T,
    b: *const T,
    d: usize,
) -> T {
    let mut count = <T as Zero>::zero();
    let one = <T as num_traits::One>::one();
    for k in 0..d {
        if *a.add(k) != *b.add(k) {
            count = count + one;
        }
    }
    count / T::from_usize(d).unwrap()
}

// ============================================================================
// Jaccard Distance
// ============================================================================

unsafe fn cdist_jaccard<T: Element + Float + FromPrimitive>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = jaccard_distance(x.add(i * d), y.add(j * d), d);
            *out.add(i * m + j) = dist;
        }
    }
}

unsafe fn pdist_jaccard<T: Element + Float + FromPrimitive>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = jaccard_distance(x.add(i * d), x.add(j * d), d);
            *out.add(k) = dist;
            k += 1;
        }
    }
}

/// Compute Jaccard distance between two vectors.
///
/// Jaccard distance for binary/set vectors: 1 - |intersection| / |union|
///
/// Implementation treats non-zero values as "true" (element present in set):
/// - intersection: number of positions where both vectors are non-zero
/// - union: number of positions where at least one vector is non-zero
///
/// Returns a value in [0, 1] where:
/// - 0 means vectors have identical non-zero patterns
/// - 1 means vectors have completely disjoint non-zero patterns
#[inline]
unsafe fn jaccard_distance<T: Element + Float + FromPrimitive>(
    a: *const T,
    b: *const T,
    d: usize,
) -> T {
    let mut intersection = <T as Zero>::zero();
    let mut union_count = <T as Zero>::zero();
    let one = <T as num_traits::One>::one();
    let zero = <T as Zero>::zero();

    for k in 0..d {
        let ak = *a.add(k);
        let bk = *b.add(k);
        let a_nonzero = ak != zero;
        let b_nonzero = bk != zero;

        if a_nonzero && b_nonzero {
            intersection = intersection + one;
        }
        if a_nonzero || b_nonzero {
            union_count = union_count + one;
        }
    }

    if union_count.is_zero() {
        zero
    } else {
        one - intersection / union_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let dist = unsafe { euclidean_distance(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 1.0).abs() < 1e-6);

        let c = [1.0f32, 1.0, 1.0];
        let dist2 = unsafe { euclidean_distance(a.as_ptr(), c.as_ptr(), 3) };
        assert!((dist2 - 3.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        let dist = unsafe { manhattan_distance(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 5.0, 3.0];
        let dist = unsafe { chebyshev_distance(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        // Same direction vectors have cosine distance 0
        let a = [1.0f32, 0.0, 0.0];
        let b = [2.0f32, 0.0, 0.0];
        let dist = unsafe { cosine_distance(a.as_ptr(), b.as_ptr(), 3) };
        assert!(dist.abs() < 1e-6);

        // Orthogonal vectors have cosine distance 1
        let c = [0.0f32, 1.0, 0.0];
        let dist2 = unsafe { cosine_distance(a.as_ptr(), c.as_ptr(), 3) };
        assert!((dist2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cdist_euclidean() {
        // X = [[0, 0], [1, 1]]
        // Y = [[1, 0], [2, 2]]
        let x = [0.0f32, 0.0, 1.0, 1.0];
        let y = [1.0f32, 0.0, 2.0, 2.0];
        let mut out = [0.0f32; 4];

        unsafe {
            cdist_kernel(
                x.as_ptr(),
                y.as_ptr(),
                out.as_mut_ptr(),
                2,
                2,
                2,
                DistanceMetric::Euclidean,
            );
        }

        // d(x0, y0) = sqrt(1) = 1
        // d(x0, y1) = sqrt(4+4) = 2*sqrt(2)
        // d(x1, y0) = sqrt(0+1) = 1
        // d(x1, y1) = sqrt(1+1) = sqrt(2)
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (8.0f32).sqrt()).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_pdist_euclidean() {
        // X = [[0, 0], [1, 0], [0, 1]] - 3 points in 2D
        let x = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 3]; // 3 = n*(n-1)/2 for n=3

        unsafe {
            pdist_kernel(
                x.as_ptr(),
                out.as_mut_ptr(),
                3,
                2,
                DistanceMetric::Euclidean,
            );
        }

        // d(0,1) = 1, d(0,2) = 1, d(1,2) = sqrt(2)
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_squareform() {
        let condensed = [1.0f32, 2.0, 3.0]; // d(0,1), d(0,2), d(1,2)
        let mut square = [0.0f32; 9];

        unsafe {
            squareform_kernel(condensed.as_ptr(), square.as_mut_ptr(), 3);
        }

        // Expected:
        // [[0, 1, 2],
        //  [1, 0, 3],
        //  [2, 3, 0]]
        assert_eq!(square[0], 0.0); // d(0,0)
        assert_eq!(square[1], 1.0); // d(0,1)
        assert_eq!(square[2], 2.0); // d(0,2)
        assert_eq!(square[3], 1.0); // d(1,0)
        assert_eq!(square[4], 0.0); // d(1,1)
        assert_eq!(square[5], 3.0); // d(1,2)
        assert_eq!(square[6], 2.0); // d(2,0)
        assert_eq!(square[7], 3.0); // d(2,1)
        assert_eq!(square[8], 0.0); // d(2,2)
    }

    #[test]
    fn test_squareform_inverse() {
        let square = [0.0f32, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let mut condensed = [0.0f32; 3];

        unsafe {
            squareform_inverse_kernel(square.as_ptr(), condensed.as_mut_ptr(), 3);
        }

        assert_eq!(condensed[0], 1.0); // d(0,1)
        assert_eq!(condensed[1], 2.0); // d(0,2)
        assert_eq!(condensed[2], 3.0); // d(1,2)
    }

    #[test]
    fn test_hamming_distance() {
        let a = [1.0f32, 0.0, 1.0, 1.0];
        let b = [1.0f32, 1.0, 0.0, 1.0];
        let dist = unsafe { hamming_distance(a.as_ptr(), b.as_ptr(), 4) };
        // 2 differences out of 4
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_distance() {
        // a = [1, 0, 1, 1] -> non-zero at indices 0, 2, 3
        // b = [1, 1, 0, 1] -> non-zero at indices 0, 1, 3
        // intersection = {0, 3} = 2
        // union = {0, 1, 2, 3} = 4
        // jaccard = 1 - 2/4 = 0.5
        let a = [1.0f32, 0.0, 1.0, 1.0];
        let b = [1.0f32, 1.0, 0.0, 1.0];
        let dist = unsafe { jaccard_distance(a.as_ptr(), b.as_ptr(), 4) };
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_minkowski_equals_euclidean() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];
        let euclidean = unsafe { euclidean_distance(a.as_ptr(), b.as_ptr(), 3) };
        let minkowski = unsafe { minkowski_distance(a.as_ptr(), b.as_ptr(), 3, 2.0) };
        assert!((euclidean - minkowski).abs() < 1e-5);
    }

    #[test]
    fn test_minkowski_equals_manhattan() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 5.0];
        let manhattan = unsafe { manhattan_distance(a.as_ptr(), b.as_ptr(), 3) };
        let minkowski = unsafe { minkowski_distance(a.as_ptr(), b.as_ptr(), 3, 1.0) };
        assert!((manhattan - minkowski).abs() < 1e-5);
    }
}
