//! Exact references for each available unary square-root ISA and its scalar tails.

use crate::sqrt_f32_oracle as oracle;

use crate::ops::UnaryOp;

type Kernel = unsafe fn(UnaryOp, *const f32, *mut f32, usize);

fn check(kernel: Kernel, name: &str, bits: &[u32], expected: &[u32]) {
    let mut input = Vec::with_capacity(bits.len() + 2);
    input.push(23.0f32);
    input.extend(bits.iter().copied().map(f32::from_bits));
    input.push(29.0);
    let mut output = vec![31.0f32; input.len()];
    unsafe {
        kernel(
            UnaryOp::Sqrt,
            input.as_ptr().add(1),
            output.as_mut_ptr().add(1),
            bits.len(),
        );
    }
    assert_eq!(output[0].to_bits(), 31.0f32.to_bits());
    assert_eq!(output[output.len() - 1].to_bits(), 31.0f32.to_bits());
    for (index, (&value, &expected)) in output[1..output.len() - 1].iter().zip(expected).enumerate()
    {
        assert_eq!(
            value.to_bits(),
            expected,
            "{name},input={:#010x},index={index}",
            bits[index]
        );
    }
    assert_eq!(input[0].to_bits(), 23.0f32.to_bits());
    assert_eq!(input[input.len() - 1].to_bits(), 29.0f32.to_bits());
    for (index, &value) in input[1..input.len() - 1].iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            bits[index],
            "{name} input mutation,index={index}"
        );
    }
}

fn check_path(kernel: Kernel, name: &str) {
    let cases = oracle::cases();
    let bits: Vec<_> = cases.iter().map(|&(bits, _)| bits).collect();
    let expected: Vec<_> = cases.iter().map(|&(_, expected)| expected).collect();
    check(kernel, name, &bits, &expected);
    for len in [
        0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257,
    ] {
        for rotation in 0..oracle::WITNESSES.len() {
            let bits: Vec<_> = oracle::WITNESSES
                .iter()
                .copied()
                .cycle()
                .skip(rotation)
                .take(len)
                .collect();
            let expected: Vec<_> = bits
                .iter()
                .copied()
                .map(oracle::rounded_sqrt_bits)
                .collect();
            check(kernel, name, &bits, &expected);
        }
    }
}

#[test]
fn f32_square_root_matches_exact_references_in_scalar_and_available_isas() {
    check_path(super::unary_scalar_f32, "scalar");
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_path(super::x86_64::avx2::unary_f32, "AVX2");
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_path(super::x86_64::avx512::unary_f32, "AVX512");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            check_path(super::aarch64::neon::unary_f32, "NEON");
        }
    }
}
