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
