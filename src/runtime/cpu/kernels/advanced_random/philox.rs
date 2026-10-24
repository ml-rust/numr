//! Philox4x32-10 PRNG kernel
//!
//! 10-round Feistel cipher from Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

use super::box_muller;
use crate::dtype::Element;

const PHILOX_M2X32_0: u32 = 0xD2511F53;
const PHILOX_M2X32_1: u32 = 0xCD9E8D57;
const PHILOX_W32_0: u32 = 0x9E3779B9;
const PHILOX_W32_1: u32 = 0xBB67AE85;

/// Philox4x32 round function
#[inline(always)]
fn philox_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let prod0 = (ctr[0] as u64).wrapping_mul(PHILOX_M2X32_0 as u64);
    let prod1 = (ctr[2] as u64).wrapping_mul(PHILOX_M2X32_1 as u64);

    [
        ((prod1 >> 32) as u32) ^ ctr[1] ^ key[0],
        prod1 as u32,
        ((prod0 >> 32) as u32) ^ ctr[3] ^ key[1],
        prod0 as u32,
    ]
}

/// Philox4x32-10: 10-round Feistel cipher
#[inline(always)]
fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;

    for _ in 0..10 {
        c = philox_round(c, k);
        k[0] = k[0].wrapping_add(PHILOX_W32_0);
        k[1] = k[1].wrapping_add(PHILOX_W32_1);
    }

    c
}

/// Convert u32 to uniform float in [0, 1): top 24 bits / 2^24.
///
/// The value is exact in f32 and f64. CUDA and WebGPU write the same value,
/// rounded to the output dtype.
#[inline(always)]
fn u32_to_uniform(u: u32) -> f64 {
    (u >> 8) as f64 / (1u64 << 24) as f64
}

/// Generate uniform random values in [0, 1) using Philox4x32-10
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn philox_uniform_kernel<T: Element>(
    out: *mut T,
    n: usize,
    key: u64,
    counter_base: u64,
) {
    let key_split = [(key & 0xFFFFFFFF) as u32, (key >> 32) as u32];
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    for i in (0..n).step_by(4) {
        let counter = counter_base.wrapping_add((i / 4) as u64);
        let ctr = [(counter & 0xFFFFFFFF) as u32, (counter >> 32) as u32, 0, 0];

        let random = philox4x32_10(ctr, key_split);

        for j in 0..4 {
            if i + j < n {
                let val = u32_to_uniform(random[j]);
                out_slice[i + j] =
                    crate::runtime::cpu::kernels::rng::clamp_uniform_sample(T::from_f64(val));
            }
        }
    }
}

/// Generate standard normal random values using Philox4x32-10 + Box-Muller
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn philox_randn_kernel<T: Element>(out: *mut T, n: usize, key: u64, counter_base: u64) {
    let key_split = [(key & 0xFFFFFFFF) as u32, (key >> 32) as u32];
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    let mut i = 0;
    while i < n {
        let counter = counter_base.wrapping_add((i / 4) as u64);
        let ctr = [(counter & 0xFFFFFFFF) as u32, (counter >> 32) as u32, 0, 0];

        let random = philox4x32_10(ctr, key_split);

        // Box-Muller: generate pairs of normal values from pairs of uniform
        for j in (0..4).step_by(2) {
            if i + j >= n {
                break;
            }

            let u1 = u32_to_uniform(random[j]);
            let u2 = u32_to_uniform(random[j + 1]);
            let (z0, z1) = box_muller(u1, u2);

            out_slice[i + j] = T::from_f64(z0);
            if i + j + 1 < n {
                out_slice[i + j + 1] = T::from_f64(z1);
            }
        }

        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_philox_uniform_range() {
        let mut out = vec![0.0f32; 1000];
        unsafe { philox_uniform_kernel(out.as_mut_ptr(), 1000, 42, 0) };
        assert!(out.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_philox_reproducibility() {
        let mut out1 = vec![0.0f64; 100];
        let mut out2 = vec![0.0f64; 100];
        unsafe {
            philox_uniform_kernel(out1.as_mut_ptr(), 100, 42, 0);
            philox_uniform_kernel(out2.as_mut_ptr(), 100, 42, 0);
        }
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_philox_different_counters() {
        let mut out1 = vec![0.0f64; 100];
        let mut out2 = vec![0.0f64; 100];
        unsafe {
            philox_uniform_kernel(out1.as_mut_ptr(), 100, 42, 0);
            philox_uniform_kernel(out2.as_mut_ptr(), 100, 42, 1);
        }
        assert_ne!(out1, out2);
    }

    /// Published Random123 known-answer vectors for Philox4x32-10.
    #[test]
    fn philox4x32_10_matches_random123_vectors() {
        let cases: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
            (
                [0, 0, 0, 0],
                [0, 0],
                [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
            ),
            (
                [0xffffffff; 4],
                [0xffffffff; 2],
                [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0],
                [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
            ),
        ];
        for (ctr, key, expected) in cases {
            assert_eq!(
                philox4x32_10(ctr, key),
                expected,
                "ctr={ctr:x?} key={key:x?}"
            );
        }
    }

    /// The u32 word to the f32 value every backend writes: top 24 bits / 2^24.
    fn word_to_f32(w: u32) -> f32 {
        (w >> 8) as f32 / 16_777_216.0
    }

    /// Counter `c` fills elements `4c..4c+4` with block `[c_lo, c_hi, 0, 0]`.
    /// A counter of `0xffffffff` carries into the hi word on the next block.
    #[test]
    fn uniform_layout_carries_counter_into_hi_word() {
        let key: u64 = 0x0123_4567_89ab_cdef;
        let words: [u32; 8] = [
            // Block [0xffffffff, 0, 0, 0].
            0x073402a4, 0x9775331e, 0x64c992ad, 0x325045ac, // Block [0, 1, 0, 0].
            0xd9412d6a, 0x59dd3f68, 0x9d46a012, 0x5d27ddd9,
        ];
        let mut out = [0.0f32; 8];
        unsafe { philox_uniform_kernel(out.as_mut_ptr(), 8, key, 0xffff_ffff) };
        let expected: Vec<u32> = words.iter().map(|&w| word_to_f32(w).to_bits()).collect();
        let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn uniform_key_and_counter_zero_match_kat_block() {
        let mut out = [0.0f32; 4];
        unsafe { philox_uniform_kernel(out.as_mut_ptr(), 4, 0, 0) };
        let words = [0x6627e8d5u32, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8];
        for (v, w) in out.iter().zip(words) {
            assert_eq!(v.to_bits(), word_to_f32(w).to_bits());
        }
    }

    /// Element `i` depends on `(key, counter, i)` only, never on the length.
    #[test]
    fn uniform_prefix_is_independent_of_length() {
        let key = 0xdead_beef_0000_0007u64;
        let counter = (5u64 << 32) | 0xffff_fffe;
        let mut long = vec![0.0f32; 103];
        unsafe { philox_uniform_kernel(long.as_mut_ptr(), long.len(), key, counter) };
        for n in [1usize, 2, 3, 5, 37, 64] {
            let mut short = vec![0.0f32; n];
            unsafe { philox_uniform_kernel(short.as_mut_ptr(), n, key, counter) };
            assert_eq!(short, long[..n], "n={n}");
        }
    }

    /// F64 output equals the F32 output widened: both hold the same 24-bit value.
    #[test]
    fn uniform_f64_equals_widened_f32() {
        let mut a = vec![0.0f32; 41];
        let mut b = vec![0.0f64; 41];
        unsafe {
            philox_uniform_kernel(a.as_mut_ptr(), 41, u64::MAX, 7);
            philox_uniform_kernel(b.as_mut_ptr(), 41, u64::MAX, 7);
        }
        let widened: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        assert_eq!(widened, b);
    }

    #[test]
    fn test_philox_randn_statistics() {
        let mut out = vec![0.0f64; 10000];
        unsafe { philox_randn_kernel(out.as_mut_ptr(), 10000, 42, 0) };

        let mean: f64 = out.iter().sum::<f64>() / 10000.0;
        let variance: f64 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 10000.0;

        assert!((mean - 0.0).abs() < 0.05, "mean = {}", mean);
        assert!((variance - 1.0).abs() < 0.1, "variance = {}", variance);
    }
}
