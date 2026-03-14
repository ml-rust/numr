//! Half-precision float conversion utilities.

/// Convert f32 to half-precision bit representation.
///
/// If `is_f16` is true, converts to IEEE 754 half-precision (F16).
/// If false, converts to brain floating point (BF16).
///
/// This is a simple conversion for common cases. For full compliance,
/// enable the `f16` feature which uses the `half` crate.
pub fn half_from_f32_util(value: f32, is_f16: bool) -> u16 {
    #[cfg(feature = "f16")]
    {
        if is_f16 {
            half::f16::from_f32(value).to_bits()
        } else {
            half::bf16::from_f32(value).to_bits()
        }
    }
    #[cfg(not(feature = "f16"))]
    {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;

        if !is_f16 {
            // BF16: truncate mantissa
            ((bits >> 16) & 0xFFFF) as u16
        } else {
            // F16: IEEE 754 half precision
            if exp == 0 {
                (sign << 15) as u16
            } else if exp == 0xFF {
                ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16
            } else {
                let new_exp = exp - 127 + 15;
                if new_exp <= 0 {
                    (sign << 15) as u16
                } else if new_exp >= 31 {
                    ((sign << 15) | 0x7C00) as u16
                } else {
                    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
                }
            }
        }
    }
}
