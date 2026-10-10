//! Cosine distance at every level.

use super::common::{LENGTHS, Lcg, TestFloat, dot_term, kernels, random_vec, reference};
use numr::distance::{self, Kernels};

/// Cosine distance at one width.
struct Cos<F> {
    pair: fn(&Kernels, &[F], &[F]) -> F,
    many: fn(&Kernels, &[F], &[F], usize, &mut [F]),
}

/// Error bound for cosine distance.
///
/// Each of the three sums has error at most `(len + 4) * eps` relative to its
/// `sum|t|`. By Cauchy-Schwarz `sum|a b| <= sqrt(|a|^2 |b|^2)`, so the dot
/// error moves the ratio by at most `(len + 4) * eps`. The two norm errors move
/// it by at most `(len + 4) * eps * |cos|`. The multiply, square root, divide
/// and subtract add about `5 * eps`, and the f64 reference adds fewer than
/// `20 * eps`. `(2 * len + 40) * eps` covers the sum with slack for second
/// order terms.
fn cosine_bound<F: TestFloat>(len: usize) -> f64 {
    (2.0 * len as f64 + 40.0) * F::EPS
}

fn cosine_reference<F: TestFloat>(a: &[F], b: &[F]) -> f64 {
    let (dot, _) = reference(a, b, dot_term);
    let (na, _) = reference(a, a, dot_term);
    let (nb, _) = reference(b, b, dot_term);
    let denom = (na * nb).sqrt();
    if denom == 0.0 { 0.0 } else { 1.0 - dot / denom }
}

fn check_cosine<F: TestFloat>(name: &str, cos: Cos<F>) {
    let mut rng = Lcg(13);
    for len in LENGTHS {
        let tol = cosine_bound::<F>(len);
        let zeros = vec![F::from_f64(0.0); len];
        for offset in [0usize, 1] {
            let a_buf = random_vec::<F>(&mut rng, len + offset, 1.0);
            let b_buf = random_vec::<F>(&mut rng, len + offset, 1.0);
            let (a, b) = (&a_buf[offset..], &b_buf[offset..]);
            let neg: Vec<F> = a.iter().map(|x| F::from_f64(-x.to_f64())).collect();
            let want = cosine_reference(a, b);
            for k in kernels() {
                let level = k.level();
                let got = (cos.pair)(&k, a, b).to_f64();
                assert!(
                    (got - want).abs() <= tol,
                    "{name}: level {level}, len {len}, offset {offset}: \
                     got {got:e}, reference {want:e}, bound {tol:e}"
                );
                let same = (cos.pair)(&k, a, a).to_f64();
                assert!(
                    same.abs() <= tol,
                    "{name}: identical, level {level}, len {len}: {same:e}"
                );
                if len > 0 {
                    let opposite = (cos.pair)(&k, a, &neg).to_f64();
                    assert!(
                        (opposite - 2.0).abs() <= tol,
                        "{name}: opposite, level {level}, len {len}: {opposite:e}"
                    );
                }
                assert_eq!((cos.pair)(&k, a, &zeros).to_f64(), 0.0, "{name}: zero b");
                assert_eq!((cos.pair)(&k, &zeros, b).to_f64(), 0.0, "{name}: zero a");
            }
        }
    }
    for d in [0usize, 1, 7, 33, 1537] {
        let query = random_vec::<F>(&mut rng, d, 1.0);
        let rows = random_vec::<F>(&mut rng, 4 * d, 1.0);
        for k in kernels() {
            let mut out = [F::from_f64(7.0); 4];
            (cos.many)(&k, &query, &rows, d, &mut out);
            for (i, got) in out.iter().enumerate() {
                let want = (cos.pair)(&k, &query, &rows[i * d..(i + 1) * d]);
                assert_eq!(got.bits(), want.bits(), "{name}: many, d {d}, row {i}");
            }
        }
    }
}

#[test]
fn cosine_every_level() {
    let f32_cos = Cos {
        pair: Kernels::cosine_distance_f32,
        many: Kernels::cosine_distance_many_f32,
    };
    let f64_cos = Cos {
        pair: Kernels::cosine_distance_f64,
        many: Kernels::cosine_distance_many_f64,
    };
    check_cosine("cosine_distance_f32", f32_cos);
    check_cosine("cosine_distance_f64", f64_cos);

    let a = [3.0f32, 4.0];
    let mut out = [9.0f32; 1];
    distance::cosine_distance_many_f32(&a, &a, 2, &mut out);
    assert_eq!(
        out[0].to_bits(),
        distance::cosine_distance_f32(&a, &a).to_bits()
    );
    let c = [3.0f64, 4.0];
    let mut out = [9.0f64; 1];
    distance::cosine_distance_many_f64(&c, &c, 2, &mut out);
    assert_eq!(
        out[0].to_bits(),
        distance::cosine_distance_f64(&c, &c).to_bits()
    );
}

// ----------------------------------------------------------------------------
// Extreme magnitudes
//
// The denominator is `sqrt(|a|^2) * sqrt(|b|^2)`. The old `sqrt(|a|^2 * |b|^2)`
// overflowed when the product of the squared norms left the float range, and
// flushed to zero when it fell below the smallest normal.
// ----------------------------------------------------------------------------

fn assert_near_f32(got: f32, want: f32, tol: f32, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e}"
    );
}

fn assert_near_f64(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e}"
    );
}

#[test]
fn cosine_f32_extreme_magnitudes() {
    let cases: [(f32, f32, f32); 4] = [
        (1e10, 1e10, 0.0),
        (1e10, -1e10, 2.0),
        (1e-12, -1e-12, 2.0),
        (1e-12, 1e-12, 0.0),
    ];
    for (x, y, want) in cases {
        let what = format!("free fn, {x:e} vs {y:e}");
        assert_near_f32(distance::cosine_distance_f32(&[x], &[y]), want, 1e-6, &what);
        for k in kernels() {
            let what = format!("level {}, {x:e} vs {y:e}", k.level());
            assert_near_f32(k.cosine_distance_f32(&[x], &[y]), want, 1e-6, &what);
        }
    }
}

#[test]
fn cosine_f32_long_vector_with_overflowing_norm_product() {
    // Each norm sum is 1.5e21 and the product is 2e42, which overflows f32.
    // Each square root is about 3.9e10, which does not.
    let a = vec![1e9f32; 1536];
    let b = vec![1e9f32; 1536];
    assert_near_f32(distance::cosine_distance_f32(&a, &b), 0.0, 1e-5, "free fn");
    for k in kernels() {
        let what = format!("level {}", k.level());
        assert_near_f32(k.cosine_distance_f32(&a, &b), 0.0, 1e-5, &what);
    }
}

#[test]
fn cosine_f64_extreme_magnitudes() {
    // The product of the two squared norms is 1e320 or 1e-400, outside the f64
    // range. Each squared norm and each square root stays in range.
    let cases: [(f64, f64, f64); 8] = [
        (1e80, 1e80, 0.0),
        (1e80, -1e80, 2.0),
        (1e-100, -1e-100, 2.0),
        (1e-100, 1e-100, 0.0),
        (1e80, 1e-100, 0.0),
        (1e-100, -1e80, 2.0),
        (3e80, 4e80, 0.0),
        (-1e-100, 1e-100, 2.0),
    ];
    for (x, y, want) in cases {
        let what = format!("free fn, {x:e} vs {y:e}");
        assert_near_f64(
            distance::cosine_distance_f64(&[x], &[y]),
            want,
            1e-12,
            &what,
        );
        for k in kernels() {
            let what = format!("level {}, {x:e} vs {y:e}", k.level());
            assert_near_f64(k.cosine_distance_f64(&[x], &[y]), want, 1e-12, &what);
        }
    }
}

#[test]
fn cosine_f64_long_vector_with_overflowing_norms() {
    for scale in [1e80f64, 1e-100] {
        let a = vec![scale; 1536];
        let b = vec![scale; 1536];
        let what = format!("free fn, scale {scale:e}");
        assert_near_f64(distance::cosine_distance_f64(&a, &b), 0.0, 1e-12, &what);
        for k in kernels() {
            let what = format!("level {}, scale {scale:e}", k.level());
            assert_near_f64(k.cosine_distance_f64(&a, &b), 0.0, 1e-12, &what);
        }
    }
}

#[test]
fn cosine_many_f32_extreme_rows() {
    // Rows are huge, tiny and zero. Both nonzero rows are parallel to the query.
    let d = 3;
    let query = [1.0f32, 1.0, 1.0];
    let rows = [
        1e10f32, 1e10, 1e10, // huge, parallel
        1e-12, 1e-12, 1e-12, // tiny, parallel
        0.0, 0.0, 0.0, // zero
    ];
    let mut out = [7.0f32; 3];
    distance::cosine_distance_many_f32(&query, &rows, d, &mut out);
    assert_near_f32(out[0], 0.0, 1e-6, "free fn, huge row");
    assert_near_f32(out[1], 0.0, 1e-6, "free fn, tiny row");
    assert_eq!(out[2], 0.0, "free fn, zero row");
    for k in kernels() {
        let mut out = [7.0f32; 3];
        k.cosine_distance_many_f32(&query, &rows, d, &mut out);
        let level = k.level();
        assert_near_f32(out[0], 0.0, 1e-6, &format!("level {level}, huge row"));
        assert_near_f32(out[1], 0.0, 1e-6, &format!("level {level}, tiny row"));
        assert_eq!(out[2], 0.0, "level {level}, zero row");
    }

    // Huge and tiny queries against the same rows.
    for (q, want) in [(1e10f32, [0.0f32, 0.0, 0.0]), (-1e-12, [2.0, 2.0, 0.0])] {
        let query = [q; 3];
        for k in kernels() {
            let mut out = [7.0f32; 3];
            k.cosine_distance_many_f32(&query, &rows, d, &mut out);
            for (i, (&got, &w)) in out.iter().zip(&want).enumerate() {
                let what = format!("level {}, query {q:e}, row {i}", k.level());
                if w == 0.0 && i == 2 {
                    assert_eq!(got, 0.0, "{what}");
                } else {
                    assert_near_f32(got, w, 1e-6, &what);
                }
            }
        }
    }
}

#[test]
fn cosine_many_f64_extreme_rows() {
    let d = 2;
    let query = [1.0f64, 1.0];
    let rows = [1e80f64, 1e80, 1e-100, 1e-100, 0.0, 0.0];
    for k in kernels() {
        let mut out = [7.0f64; 3];
        k.cosine_distance_many_f64(&query, &rows, d, &mut out);
        let level = k.level();
        assert_near_f64(out[0], 0.0, 1e-12, &format!("level {level}, huge row"));
        assert_near_f64(out[1], 0.0, 1e-12, &format!("level {level}, tiny row"));
        assert_eq!(out[2], 0.0, "level {level}, zero row");
    }
}
