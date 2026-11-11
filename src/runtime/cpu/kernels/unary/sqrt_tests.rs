//! Exact references for the portable unary fallback.

use crate::sqrt_f32_oracle as oracle;

use super::unary_op_scalar;
use crate::ops::UnaryOp;

#[test]
fn portable_f32_square_root_matches_integer_dyadic_references() {
    let cases = oracle::cases();
    let input: Vec<_> = cases
        .iter()
        .map(|&(bits, _)| f32::from_bits(bits))
        .collect();
    let mut output = vec![0.0f32; input.len()];
    unsafe {
        unary_op_scalar(
            UnaryOp::Sqrt,
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len(),
        );
    }
    for (index, (value, &(_, expected))) in output.iter().zip(cases).enumerate() {
        assert_eq!(
            value.to_bits(),
            expected,
            "portable input={:#010x},index={index}",
            cases[index].0
        );
    }
}
