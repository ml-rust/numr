//! The i8 dot product at every level, against an exact i64 loop saturated to i32.

use super::common::{LENGTHS, Lcg, kernels};
use numr::distance::{self, Kernels};

fn exact_i8_dot(a: &[i8], b: &[i8]) -> i32 {
    let sum: i64 = a.iter().zip(b).map(|(&x, &y)| x as i64 * y as i64).sum();
    sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[test]
fn i8_dot_every_level() {
    let mut rng = Lcg(0xDEAD_BEEF);
    let scale = 0.037f32;
    for len in LENGTHS.iter().copied().chain([4095, 4096, 4097, 70_001]) {
        for offset in [0usize, 1] {
            let a_buf: Vec<i8> = (0..len + offset).map(|_| rng.next_i8()).collect();
            let b_buf: Vec<i8> = (0..len + offset).map(|_| rng.next_i8()).collect();
            let (a, b) = (&a_buf[offset..], &b_buf[offset..]);
            let want = exact_i8_dot(a, b);
            for k in kernels() {
                let level = k.level();
                assert_eq!(
                    k.dot_i8(a, b),
                    want,
                    "level {level}, len {len}, offset {offset}"
                );
                assert_eq!(
                    k.dot_i8_scaled(a, b, scale).to_bits(),
                    (want as f32 * scale).to_bits(),
                    "scaled: level {level}, len {len}, offset {offset}"
                );
            }
            assert_eq!(distance::dot_i8(a, b), want, "free: len {len}");
            assert_eq!(
                distance::dot_i8_scaled(a, b, scale).to_bits(),
                (want as f32 * scale).to_bits(),
                "free scaled: len {len}"
            );
        }
    }
}

#[test]
fn i8_dot_saturates_at_every_level() {
    // Longer than 8x the longest run an i32 accumulator holds exactly.
    let len = 8 * (i32::MAX as usize / (128 * 128));
    let pos = vec![127i8; len];
    let neg = vec![-127i8; len];
    for k in kernels() {
        assert_eq!(k.dot_i8(&pos, &pos), i32::MAX, "level {}", k.level());
        assert_eq!(k.dot_i8(&pos, &neg), i32::MIN, "level {}", k.level());
    }
}

#[test]
fn i8_dot_many_matches_per_row() {
    let mut rng = Lcg(99);
    for d in [0usize, 1, 31, 32, 33, 1537] {
        let query: Vec<i8> = (0..d).map(|_| rng.next_i8()).collect();
        let rows: Vec<i8> = (0..6 * d).map(|_| rng.next_i8()).collect();
        for k in kernels() {
            let mut out = [i32::MIN; 6];
            k.dot_i8_many(&query, &rows, d, &mut out);
            for (i, &got) in out.iter().enumerate() {
                let row = &rows[i * d..(i + 1) * d];
                assert_eq!(got, k.dot_i8(&query, row), "level {}, d {d}", k.level());
                assert_eq!(got, exact_i8_dot(&query, row), "exact, d {d}, row {i}");
            }
        }
        let mut free = [i32::MIN; 6];
        distance::dot_i8_many(&query, &rows, d, &mut free);
        let mut held = [i32::MIN; 6];
        Kernels::detect().dot_i8_many(&query, &rows, d, &mut held);
        assert_eq!(free, held, "free many, d {d}");
    }
}
