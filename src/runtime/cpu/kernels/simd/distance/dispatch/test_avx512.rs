//! Every-length checks for the AVX-512 distance kernels.
//!
//! The shared lengths in `test_support` sit on AVX2 loop boundaries. They miss
//! most AVX-512 masked-tail widths. These checks run every length up to a
//! limit, so every tail width runs after every loop exit.

use super::test_support::{Kernel, Lcg, Term, TestFloat, bound, random_vec, reference};
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// Lengths `0..=F32_MAX_LEN` cover the f32 main loop (64), one register (16)
/// and every tail width (0..16). The limit also reaches lengths that run the
/// main loop, the one-register loop and a tail in the same call.
pub const F32_MAX_LEN: usize = 160;

/// Lengths `0..=F64_MAX_LEN` cover the f64 main loop (32), one register (8)
/// and every tail width (0..8). The limit also reaches lengths that run the
/// main loop, the one-register loop and a tail in the same call.
pub const F64_MAX_LEN: usize = 80;

/// True when this CPU reports every feature `SimdLevel::Avx512` guarantees.
fn avx512_supported() -> bool {
    detect_simd() == SimdLevel::Avx512
}

/// `SimdLevel::Avx512` matches [`reference`] within [`bound`] for every length
/// in `0..=max_len`, at both offsets and at unit and subnormal scale.
///
/// It returns before any kernel call when [`avx512_supported`] is false, so no
/// test runs an unsupported instruction.
pub fn check_avx512_every_length<F: TestFloat>(
    op: &str,
    kernel: Kernel<F>,
    term: Term,
    max_len: usize,
) {
    if !avx512_supported() {
        return;
    }
    let mut rng = Lcg::new(0xD1B5_4A32_D192_ED03);
    for scale in [1.0, F::SUBNORMAL_SCALE] {
        for len in 0..=max_len {
            for offset in [0usize, 1] {
                let a_buf = random_vec::<F>(&mut rng, len + offset, scale);
                let b_buf = random_vec::<F>(&mut rng, len + offset, scale);
                let (a, b) = (&a_buf[offset..], &b_buf[offset..]);
                let (want, abs_sum) = reference(a, b, term);
                let tol = bound::<F>(len, abs_sum);
                let got = kernel(SimdLevel::Avx512, a.as_ptr(), b.as_ptr(), len).to_f64();
                assert!(
                    got.is_finite() && (got - want).abs() <= tol,
                    "{op}: level AVX-512, len {len}, offset {offset}, scale {scale:e}: \
                     got {got:e}, reference {want:e}, bound {tol:e}"
                );
            }
        }
    }
}
