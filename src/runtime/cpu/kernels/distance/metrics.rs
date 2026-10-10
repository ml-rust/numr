//! Per-pair distance metrics.
//!
//! Each function reduces one pair of `d`-component vectors to a single value.
//! Every one of them is generic over the accumulator `A` rather than computing
//! in the element type, so a narrow-float tensor accumulates in f32 and matches
//! the `AccT` CUDA's `distance.cu` uses. See [`DistAcc`] for which accumulator
//! each element type gets.
//!
//! `sqeuclidean`, `euclidean`, `manhattan` and `cosine` route f32 and f64 pairs
//! to the SIMD kernels in `simd::distance`. Every other pair keeps the
//! sequential loop below.

use super::acc::DistAcc;
use crate::dtype::Element;
use crate::runtime::cpu::kernels::simd::distance as simd;
use std::any::TypeId;

/// Which SIMD kernel width an element and accumulator pair can use.
enum Route {
    /// `T` and `A` are both exactly `f32`.
    F32,
    /// `T` and `A` are both exactly `f64`.
    F64,
    /// Any other pair. It keeps the sequential generic loop.
    Generic,
}

/// Picks the kernel width for `(T, A)`.
///
/// The test compares `TypeId`s, not `T::DTYPE`. A `Route::F32` result proves
/// `T` is `f32` itself, so casting `*const T` to `*const f32` is an identity
/// cast. The comparison folds to a constant at compile time.
#[inline]
fn route<T: Element, A: DistAcc<T>>() -> Route {
    let (t, a) = (TypeId::of::<T>(), TypeId::of::<A>());
    if t == TypeId::of::<f32>() && a == TypeId::of::<f32>() {
        Route::F32
    } else if t == TypeId::of::<f64>() && a == TypeId::of::<f64>() {
        Route::F64
    } else {
        Route::Generic
    }
}

/// Squared Euclidean distance: `sum((a - b)^2)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn sqeuclidean<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    // `route` proved the element type, so each cast keeps the pointee type.
    match route::<T, A>() {
        Route::F32 => A::from_f64(simd::sqeuclidean_f32(a.cast(), b.cast(), d).into()),
        Route::F64 => A::from_f64(simd::sqeuclidean_f64(a.cast(), b.cast(), d)),
        Route::Generic => sqeuclidean_loop::<T, A>(a, b, d),
    }
}

/// Sequential squared Euclidean loop, in the accumulator `A`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
unsafe fn sqeuclidean_loop<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut sum = A::zero();
    for k in 0..d {
        let diff = A::widen(*a.add(k)) - A::widen(*b.add(k));
        sum = sum + diff * diff;
    }
    sum
}

/// Euclidean (L2) distance.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn euclidean<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    sqeuclidean::<T, A>(a, b, d).sqrt()
}

/// Manhattan (L1) distance: `sum(|a - b|)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn manhattan<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    // `route` proved the element type, so each cast keeps the pointee type.
    match route::<T, A>() {
        Route::F32 => A::from_f64(simd::manhattan_f32(a.cast(), b.cast(), d).into()),
        Route::F64 => A::from_f64(simd::manhattan_f64(a.cast(), b.cast(), d)),
        Route::Generic => manhattan_loop::<T, A>(a, b, d),
    }
}

/// Sequential Manhattan loop, in the accumulator `A`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
unsafe fn manhattan_loop<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut sum = A::zero();
    for k in 0..d {
        sum = sum + (A::widen(*a.add(k)) - A::widen(*b.add(k))).abs();
    }
    sum
}

/// Chebyshev (L-infinity) distance: `max(|a - b|)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn chebyshev<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut max = A::zero();
    for k in 0..d {
        let abs_diff = (A::widen(*a.add(k)) - A::widen(*b.add(k))).abs();
        if abs_diff > max {
            max = abs_diff;
        }
    }
    max
}

/// Minkowski (Lp) distance: `sum(|a - b|^p)^(1/p)`.
///
/// `p` arrives in the accumulator's precision, never rounded into the element
/// type first: an exponent rounded into F16 changes which curve is being
/// measured, and the answer changes by more than its last digit.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn minkowski<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize, p: A) -> A {
    let mut sum = A::zero();
    for k in 0..d {
        sum = sum + (A::widen(*a.add(k)) - A::widen(*b.add(k))).abs().powf(p);
    }
    sum.powf(A::one() / p)
}

/// Cosine distance: `1 - dot(a, b) / (|a| * |b|)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn cosine<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    // `route` proved the element type, so each cast keeps the pointee type.
    let (dot, norm_a, norm_b) = match route::<T, A>() {
        Route::F32 => {
            let s = simd::cosine_sums_f32(a.cast(), b.cast(), d);
            (
                A::from_f64(s.dot.into()),
                A::from_f64(s.norm_a.into()),
                A::from_f64(s.norm_b.into()),
            )
        }
        Route::F64 => {
            let s = simd::cosine_sums_f64(a.cast(), b.cast(), d);
            (
                A::from_f64(s.dot),
                A::from_f64(s.norm_a),
                A::from_f64(s.norm_b),
            )
        }
        Route::Generic => cosine_sums_loop::<T, A>(a, b, d),
    };
    cosine_from_sums(dot, norm_a, norm_b)
}

/// Cosine distance from its three sums: `1 - dot / (sqrt(norm_a) * sqrt(norm_b))`.
///
/// A zero denominator gives 0. This is the one place the sums become a
/// distance, for every route here and for `crate::distance`.
///
/// The two roots are taken before the product. The product `norm_a * norm_b`
/// is a fourth-power quantity, so it overflows f32 once `|a| * |b|` passes
/// about 1.8e19 and flushes to zero once it falls below about 3.7e-23, while
/// the roots stay in range.
#[inline]
pub fn cosine_from_sums<A: num_traits::Float>(dot: A, norm_a: A, norm_b: A) -> A {
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom.is_zero() {
        A::zero()
    } else {
        A::one() - dot / denom
    }
}

/// Sequential `(a·b, a·a, b·b)` loop, in the accumulator `A`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
unsafe fn cosine_sums_loop<T: Element, A: DistAcc<T>>(
    a: *const T,
    b: *const T,
    d: usize,
) -> (A, A, A) {
    let mut dot = A::zero();
    let mut norm_a = A::zero();
    let mut norm_b = A::zero();

    for k in 0..d {
        let ak = A::widen(*a.add(k));
        let bk = A::widen(*b.add(k));
        dot = dot + ak * bk;
        norm_a = norm_a + ak * ak;
        norm_b = norm_b + bk * bk;
    }

    (dot, norm_a, norm_b)
}

/// Correlation distance: `1 - Pearson r`.
///
/// Measures how similar the patterns in two vectors are, invariant to linear
/// transformations. 0 means perfect positive correlation, 2 perfect negative.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn correlation<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let d_a = A::count(d);

    let mut sum_a = A::zero();
    let mut sum_b = A::zero();
    for k in 0..d {
        sum_a = sum_a + A::widen(*a.add(k));
        sum_b = sum_b + A::widen(*b.add(k));
    }
    let mean_a = sum_a / d_a;
    let mean_b = sum_b / d_a;

    let mut cov = A::zero();
    let mut var_a = A::zero();
    let mut var_b = A::zero();
    for k in 0..d {
        let da = A::widen(*a.add(k)) - mean_a;
        let db = A::widen(*b.add(k)) - mean_b;
        cov = cov + da * db;
        var_a = var_a + da * da;
        var_b = var_b + db * db;
    }

    let denom = var_a.sqrt() * var_b.sqrt();
    if denom.is_zero() {
        A::zero()
    } else {
        A::one() - cov / denom
    }
}

/// Hamming distance: the fraction of positions where the vectors differ.
///
/// For continuous-valued vectors this counts exact inequality. Widening is
/// value-preserving, so comparing widened components decides the same way
/// comparing elements would; only the count and the division change width.
///
/// Returns a value in `[0, 1]`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn hamming<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut count = A::zero();
    for k in 0..d {
        if A::widen(*a.add(k)) != A::widen(*b.add(k)) {
            count = count + A::one();
        }
    }
    count / A::count(d)
}

/// Jaccard distance for binary/set vectors: `1 - |intersection| / |union|`.
///
/// Non-zero values are treated as "element present in set".
///
/// Returns a value in `[0, 1]`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn jaccard<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut intersection = A::zero();
    let mut union_count = A::zero();

    for k in 0..d {
        let a_nonzero = !A::widen(*a.add(k)).is_zero();
        let b_nonzero = !A::widen(*b.add(k)).is_zero();

        if a_nonzero && b_nonzero {
            intersection = intersection + A::one();
        }
        if a_nonzero || b_nonzero {
            union_count = union_count + A::one();
        }
    }

    if union_count.is_zero() {
        A::zero()
    } else {
        A::one() - intersection / union_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_matches_hand_computation() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let dist: f32 = unsafe { euclidean::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 1.0).abs() < 1e-6);

        let c = [1.0f32, 1.0, 1.0];
        let dist2: f32 = unsafe { euclidean::<f32, f32>(a.as_ptr(), c.as_ptr(), 3) };
        assert!((dist2 - 3.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn manhattan_sums_absolute_differences() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        let dist: f32 = unsafe { manhattan::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 6.0).abs() < 1e-6);
    }

    #[test]
    fn chebyshev_takes_the_largest_difference() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 5.0, 3.0];
        let dist: f32 = unsafe { chebyshev::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_is_zero_for_parallel_and_one_for_orthogonal() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [2.0f32, 0.0, 0.0];
        let dist: f32 = unsafe { cosine::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!(dist.abs() < 1e-6);

        let c = [0.0f32, 1.0, 0.0];
        let dist2: f32 = unsafe { cosine::<f32, f32>(a.as_ptr(), c.as_ptr(), 3) };
        assert!((dist2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hamming_counts_the_fraction_that_differs() {
        let a = [1.0f32, 0.0, 1.0, 1.0];
        let b = [1.0f32, 1.0, 0.0, 1.0];
        let dist: f32 = unsafe { hamming::<f32, f32>(a.as_ptr(), b.as_ptr(), 4) };
        // 2 differences out of 4
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn jaccard_compares_non_zero_patterns() {
        // a = [1, 0, 1, 1] -> non-zero at 0, 2, 3; b = [1, 1, 0, 1] -> 0, 1, 3.
        // intersection = {0, 3} = 2, union = {0, 1, 2, 3} = 4, so 1 - 2/4.
        let a = [1.0f32, 0.0, 1.0, 1.0];
        let b = [1.0f32, 1.0, 0.0, 1.0];
        let dist: f32 = unsafe { jaccard::<f32, f32>(a.as_ptr(), b.as_ptr(), 4) };
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn minkowski_at_p2_equals_euclidean() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];
        let euc: f32 = unsafe { euclidean::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        let mink: f32 = unsafe { minkowski::<f32, f32>(a.as_ptr(), b.as_ptr(), 3, 2.0) };
        assert!((euc - mink).abs() < 1e-5);
    }

    #[test]
    fn minkowski_at_p1_equals_manhattan() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 5.0];
        let man: f32 = unsafe { manhattan::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        let mink: f32 = unsafe { minkowski::<f32, f32>(a.as_ptr(), b.as_ptr(), 3, 1.0) };
        assert!((man - mink).abs() < 1e-5);
    }

    #[test]
    fn cosine_of_a_zero_vector_is_zero_on_every_route() {
        // `denom == 0 -> 0` must hold on the SIMD routes and on the loop route.
        let zero32 = [0.0f32; 37];
        let ones32 = [1.0f32; 37];
        let zero64 = [0.0f64; 37];
        let ones64 = [1.0f64; 37];
        let simd32: f32 = unsafe { cosine::<f32, f32>(zero32.as_ptr(), ones32.as_ptr(), 37) };
        let simd64: f64 = unsafe { cosine::<f64, f64>(zero64.as_ptr(), ones64.as_ptr(), 37) };
        // An f64 accumulator over f32 elements is not a SIMD route.
        let looped: f64 = unsafe { cosine::<f32, f64>(zero32.as_ptr(), ones32.as_ptr(), 37) };
        assert_eq!(simd32, 0.0, "f32 SIMD route, len 37");
        assert_eq!(simd64, 0.0, "f64 SIMD route, len 37");
        assert_eq!(looped, 0.0, "loop route, len 37");
    }

    #[test]
    fn f32_and_f64_pairs_route_to_the_simd_dispatchers() {
        let a32: Vec<f32> = (0..45).map(|i| (i as f32 * 0.37).sin()).collect();
        let b32: Vec<f32> = (0..45).map(|i| (i as f32 * 0.11).cos()).collect();
        let a64: Vec<f64> = a32.iter().map(|&x| f64::from(x)).collect();
        let b64: Vec<f64> = b32.iter().map(|&x| f64::from(x)).collect();
        let (p32, q32, p64, q64) = (a32.as_ptr(), b32.as_ptr(), a64.as_ptr(), b64.as_ptr());
        unsafe {
            assert_eq!(
                sqeuclidean::<f32, f32>(p32, q32, 45).to_bits(),
                simd::sqeuclidean_f32(p32, q32, 45).to_bits()
            );
            assert_eq!(
                sqeuclidean::<f64, f64>(p64, q64, 45).to_bits(),
                simd::sqeuclidean_f64(p64, q64, 45).to_bits()
            );
            assert_eq!(
                manhattan::<f32, f32>(p32, q32, 45).to_bits(),
                simd::manhattan_f32(p32, q32, 45).to_bits()
            );
            assert_eq!(
                manhattan::<f64, f64>(p64, q64, 45).to_bits(),
                simd::manhattan_f64(p64, q64, 45).to_bits()
            );
            let s = simd::cosine_sums_f32(p32, q32, 45);
            let want = 1.0 - s.dot / (s.norm_a * s.norm_b).sqrt();
            assert_eq!(cosine::<f32, f32>(p32, q32, 45).to_bits(), want.to_bits());
            let s = simd::cosine_sums_f64(p64, q64, 45);
            let want = 1.0 - s.dot / (s.norm_a * s.norm_b).sqrt();
            assert_eq!(cosine::<f64, f64>(p64, q64, 45).to_bits(), want.to_bits());
        }
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_manhattan_accumulates_in_f32() {
        // 1024 then 64 terms of 0.25: an f16 accumulator freezes at 1024 (its
        // spacing there is 1.0), an f32 accumulator reaches 1040.
        let mut a = vec![half::f16::from_f32(0.25); 65];
        a[0] = half::f16::from_f32(1024.0);
        let b = vec![half::f16::ZERO; 65];
        let dist: f32 = unsafe { manhattan::<half::f16, f32>(a.as_ptr(), b.as_ptr(), 65) };
        assert_eq!(dist, 1040.0);
    }
}
