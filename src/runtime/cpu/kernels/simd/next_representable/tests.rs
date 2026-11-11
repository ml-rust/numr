//! Check available ISA paths, integer payloads, unaligned loads, and scalar tails.

use super::super::super::detect_simd;
use super::*;

fn available_levels() -> Vec<SimdLevel> {
    let detected = detect_simd();
    let mut levels = vec![SimdLevel::Scalar];
    match detected {
        SimdLevel::Avx512 => levels.extend([SimdLevel::Avx2Fma, SimdLevel::Avx512]),
        SimdLevel::Avx2Fma => levels.push(SimdLevel::Avx2Fma),
        SimdLevel::NeonFp16 => levels.extend([SimdLevel::Neon, SimdLevel::NeonFp16]),
        SimdLevel::Neon => levels.push(SimdLevel::Neon),
        _ => {}
    }
    levels
}

#[test]
fn binary32_isa_paths_and_tails_preserve_exact_bits() {
    let mut input = vec![0x1234_5678u32];
    input.extend([
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc0_0123,
        0xffc0_0123,
    ]);
    for exponent in 0..256u32 {
        for mantissa in [0, 1, 0x003f_ffff, 0x007f_ffff] {
            for sign in [0, 0x8000_0000] {
                input.push(sign | (exponent << 23) | mantissa);
            }
        }
    }
    let original = input.clone();
    let mut lengths: Vec<usize> = (0..=35).collect();
    lengths.extend([input.len() - 2, input.len() - 1]);
    for level in available_levels() {
        for up in [false, true] {
            for &len in &lengths {
                let mut output = vec![0x1357_2468u32; len + 2];
                unsafe {
                    next_f32(
                        input.as_ptr().add(1),
                        output.as_mut_ptr().add(1),
                        len,
                        up,
                        level,
                    );
                }
                assert_eq!(output[0], 0x1357_2468);
                assert_eq!(output[len + 1], 0x1357_2468);
                for index in 0..len {
                    let bits = input[index + 1];
                    let expected = if bits & 0x7fff_ffff > 0x7f80_0000 {
                        bits
                    } else if up {
                        f32::from_bits(bits).next_up().to_bits()
                    } else {
                        f32::from_bits(bits).next_down().to_bits()
                    };
                    assert_eq!(
                        output[index + 1],
                        expected,
                        "{level:?} up={up} len={len} index={index}"
                    );
                }
            }
        }
    }
    assert_eq!(input, original);
}

#[test]
fn binary64_isa_paths_and_tails_preserve_exact_bits() {
    let mut input = vec![0x1234_5678_1234_5678u64];
    input.extend([
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x800f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x8010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
        0x7ff8_1234_5678_1234,
        0xfff8_1234_5678_1234,
    ]);
    for exponent in 0..2048u64 {
        for mantissa in [0, 1, 0x0007_ffff_ffff_ffff, 0x000f_ffff_ffff_ffff] {
            for sign in [0, 0x8000_0000_0000_0000] {
                input.push(sign | (exponent << 52) | mantissa);
            }
        }
    }
    let original = input.clone();
    let mut lengths: Vec<usize> = (0..=19).collect();
    lengths.extend([input.len() - 2, input.len() - 1]);
    for level in available_levels() {
        for up in [false, true] {
            for &len in &lengths {
                let mut output = vec![0x1357_2468_1357_2468u64; len + 2];
                unsafe {
                    next_f64(
                        input.as_ptr().add(1),
                        output.as_mut_ptr().add(1),
                        len,
                        up,
                        level,
                    );
                }
                assert_eq!(output[0], 0x1357_2468_1357_2468);
                assert_eq!(output[len + 1], 0x1357_2468_1357_2468);
                for index in 0..len {
                    let bits = input[index + 1];
                    let expected = if bits & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000 {
                        bits
                    } else if up {
                        f64::from_bits(bits).next_up().to_bits()
                    } else {
                        f64::from_bits(bits).next_down().to_bits()
                    };
                    assert_eq!(
                        output[index + 1],
                        expected,
                        "{level:?} up={up} len={len} index={index}"
                    );
                }
            }
        }
    }
    assert_eq!(input, original);
}
