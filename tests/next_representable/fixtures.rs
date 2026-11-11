//! Generated encodings and independent standard-library references.

macro_rules! fixtures {
    ($values:ident, $reference:ident, $float:ty, $bits:ty, $shift:expr, $max_exp:expr) => {
        pub fn $values() -> Vec<$bits> {
            let mantissa_mask = ((1 as $bits) << $shift) - 1;
            let sign = (1 as $bits) << (<$bits>::BITS - 1);
            let infinity = ($max_exp as $bits) << $shift;
            let mut values = Vec::new();
            for magnitude in [
                0,
                1,
                2,
                mantissa_mask - 1,
                mantissa_mask,
                mantissa_mask + 1,
                mantissa_mask + 2,
                infinity - 2,
                infinity - 1,
                infinity,
                infinity | 1,
                infinity | 0x123,
                infinity | (1 << ($shift - 1)),
                infinity | (1 << ($shift - 1)) | 0x456,
                infinity | mantissa_mask,
            ] {
                values.extend([magnitude, magnitude | sign]);
            }
            for exponent in 0..$max_exp {
                for mantissa in [0, 1, mantissa_mask / 2, mantissa_mask - 1, mantissa_mask] {
                    let magnitude = ((exponent as $bits) << $shift) | mantissa;
                    values.extend([magnitude, magnitude | sign]);
                }
            }
            values
        }

        pub fn $reference(bits: $bits, up: bool) -> $bits {
            let magnitude = bits & (<$bits>::MAX >> 1);
            if magnitude > (($max_exp as $bits) << $shift) {
                return bits;
            }
            let value = <$float>::from_bits(bits);
            if up {
                value.next_up().to_bits()
            } else {
                value.next_down().to_bits()
            }
        }
    };
}

fixtures!(f32_bits, reference_f32, f32, u32, 23, 255);
fixtures!(f64_bits, reference_f64, f64, u64, 52, 2047);
