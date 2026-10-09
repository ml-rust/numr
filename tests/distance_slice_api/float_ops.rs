//! Summed float ops (dot, l2, Manhattan) at every level, against an f64 reference.
//!
//! Each op runs against a Neumaier-compensated f64 reference, within a bound
//! derived from the length. The bound mirrors the in-crate kernel tests
//! (`simd/distance/dispatch/test_support.rs`). The `*_many_*` calls must match
//! the per-pair call bit for bit.

use super::common::{
    LENGTHS, Lcg, TestFloat, bound, dot_term, kernels, l2_term, manhattan_term, random_vec,
    reference,
};
use numr::distance::{self, Kernels};

/// One summed op at one width.
struct Op<F> {
    name: &'static str,
    pair: fn(&Kernels, &[F], &[F]) -> F,
    many: fn(&Kernels, &[F], &[F], usize, &mut [F]),
    free: fn(&[F], &[F]) -> F,
    free_many: fn(&[F], &[F], usize, &mut [F]),
    term: fn(f64, f64) -> f64,
}

fn ops_f32() -> [Op<f32>; 3] {
    [
        Op {
            name: "dot_f32",
            pair: Kernels::dot_f32,
            many: Kernels::dot_many_f32,
            free: distance::dot_f32,
            free_many: distance::dot_many_f32,
            term: dot_term,
        },
        Op {
            name: "l2_squared_f32",
            pair: Kernels::l2_squared_f32,
            many: Kernels::l2_squared_many_f32,
            free: distance::l2_squared_f32,
            free_many: distance::l2_squared_many_f32,
            term: l2_term,
        },
        Op {
            name: "manhattan_f32",
            pair: Kernels::manhattan_f32,
            many: Kernels::manhattan_many_f32,
            free: distance::manhattan_f32,
            free_many: distance::manhattan_many_f32,
            term: manhattan_term,
        },
    ]
}

fn ops_f64() -> [Op<f64>; 3] {
    [
        Op {
            name: "dot_f64",
            pair: Kernels::dot_f64,
            many: Kernels::dot_many_f64,
            free: distance::dot_f64,
            free_many: distance::dot_many_f64,
            term: dot_term,
        },
        Op {
            name: "l2_squared_f64",
            pair: Kernels::l2_squared_f64,
            many: Kernels::l2_squared_many_f64,
            free: distance::l2_squared_f64,
            free_many: distance::l2_squared_many_f64,
            term: l2_term,
        },
        Op {
            name: "manhattan_f64",
            pair: Kernels::manhattan_f64,
            many: Kernels::manhattan_many_f64,
            free: distance::manhattan_f64,
            free_many: distance::manhattan_many_f64,
            term: manhattan_term,
        },
    ]
}

/// Every level matches the reference within [`bound`], at offsets 0 and 1.
/// The free function matches `Kernels::detect()` bit for bit.
fn check_reference<F: TestFloat>(op: &Op<F>, scale: f64) {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let detected = Kernels::detect();
    for len in LENGTHS {
        for offset in [0usize, 1] {
            let a_buf = random_vec::<F>(&mut rng, len + offset, scale);
            let b_buf = random_vec::<F>(&mut rng, len + offset, scale);
            let (a, b) = (&a_buf[offset..], &b_buf[offset..]);
            let (want, abs_sum) = reference(a, b, op.term);
            let tol = bound::<F>(len, abs_sum);
            for k in kernels() {
                let got = (op.pair)(&k, a, b).to_f64();
                assert!(
                    got.is_finite() && (got - want).abs() <= tol,
                    "{}: level {}, len {len}, offset {offset}: \
                     got {got:e}, reference {want:e}, bound {tol:e}",
                    op.name,
                    k.level()
                );
            }
            assert_eq!(
                (op.free)(a, b).bits(),
                (op.pair)(&detected, a, b).bits(),
                "{}: free function, len {len}, offset {offset}",
                op.name
            );
        }
    }
}

/// `many` matches the per-row pair call bit for bit, and `d == 0` fills zeros.
fn check_many<F: TestFloat>(op: &Op<F>) {
    const N_ROWS: usize = 5;
    let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
    for d in [0usize, 1, 7, 16, 33, 65, 1537] {
        let query = random_vec::<F>(&mut rng, d, 1.0);
        let rows = random_vec::<F>(&mut rng, N_ROWS * d, 1.0);
        for k in kernels() {
            let mut out = [F::from_f64(7.0); N_ROWS];
            (op.many)(&k, &query, &rows, d, &mut out);
            for (i, got) in out.iter().enumerate() {
                let want = (op.pair)(&k, &query, &rows[i * d..(i + 1) * d]);
                assert_eq!(
                    got.bits(),
                    want.bits(),
                    "{}: level {}, d {d}, row {i}",
                    op.name,
                    k.level()
                );
                if d == 0 {
                    assert_eq!(got.to_f64(), 0.0, "{}: d 0, row {i}", op.name);
                }
            }
        }
        let mut free = [F::from_f64(7.0); N_ROWS];
        let mut held = [F::from_f64(7.0); N_ROWS];
        (op.free_many)(&query, &rows, d, &mut free);
        (op.many)(&Kernels::detect(), &query, &rows, d, &mut held);
        let free_bits: Vec<u64> = free.iter().map(|v| v.bits()).collect();
        let held_bits: Vec<u64> = held.iter().map(|v| v.bits()).collect();
        assert_eq!(free_bits, held_bits, "{}: free many, d {d}", op.name);
    }
}

/// A NaN anywhere makes the result NaN. A `+inf` makes l2 and Manhattan `+inf`.
fn check_specials<F: TestFloat>(op: &Op<F>) {
    let len = 37;
    for pos in [0, len / 2, len - 1] {
        for k in kernels() {
            let mut rng = Lcg(77);
            let mut a = random_vec::<F>(&mut rng, len, 1.0);
            let b = random_vec::<F>(&mut rng, len, 1.0);
            a[pos] = F::from_f64(f64::NAN);
            let got = (op.pair)(&k, &a, &b).to_f64();
            assert!(
                got.is_nan(),
                "{}: level {}, NaN at {pos}",
                op.name,
                k.level()
            );
            if op.name.starts_with("dot") {
                continue;
            }
            a[pos] = F::from_f64(f64::INFINITY);
            let got = (op.pair)(&k, &a, &b).to_f64();
            assert!(
                got == f64::INFINITY,
                "{}: level {}, +inf at {pos}: got {got:e}",
                op.name,
                k.level()
            );
        }
    }
}

fn check_ops<F: TestFloat>(ops: &[Op<F>]) {
    for op in ops {
        check_reference(op, 1.0);
        check_reference(op, F::SUBNORMAL_SCALE);
        check_many(op);
        check_specials(op);
    }
}

#[test]
fn f32_ops_every_level() {
    check_ops(&ops_f32());
}

#[test]
fn f64_ops_every_level() {
    check_ops(&ops_f64());
}
