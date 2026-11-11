//! Portable integer stepping and scalar tails.

macro_rules! scalar_kernel {
    ($step:ident, $kernel:ident, $bits:ty, $sign:expr, $inf:expr) => {
        #[inline]
        pub(crate) fn $step(bits: $bits, up: bool) -> $bits {
            let magnitude = bits & !$sign;
            if magnitude > $inf || bits == if up { $inf } else { $sign | $inf } {
                bits
            } else if magnitude == 0 {
                if up { 1 } else { $sign | 1 }
            } else if (bits & $sign == 0) == up {
                bits.wrapping_add(1)
            } else {
                bits.wrapping_sub(1)
            }
        }

        /// Step raw encodings without floating loads.
        ///
        /// # Safety
        /// Pointers must address `len` readable input and writable output elements.
        /// Output must not overlap unread input elements.
        pub unsafe fn $kernel(src: *const $bits, dst: *mut $bits, len: usize, up: bool) {
            for index in 0..len {
                unsafe {
                    dst.add(index).write($step(src.add(index).read(), up));
                }
            }
        }
    };
}

scalar_kernel!(step_u32, next_f32, u32, 0x8000_0000, 0x7f80_0000);
scalar_kernel!(
    step_u64,
    next_f64,
    u64,
    0x8000_0000_0000_0000,
    0x7ff0_0000_0000_0000
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceptional_encodings_preserve_exact_bits() {
        for (bits, up, down) in [
            (0, 1, 0x8000_0001),
            (0x8000_0000, 1, 0x8000_0001),
            (0x7f80_0000, 0x7f80_0000, 0x7f7f_ffff),
            (0xff80_0000, 0xff7f_ffff, 0xff80_0000),
            (1, 2, 0),
            (0x8000_0001, 0x8000_0000, 0x8000_0002),
        ] {
            assert_eq!(step_u32(bits, true), up);
            assert_eq!(step_u32(bits, false), down);
        }
        for bits in [0x7f80_0001, 0xff80_0001, 0x7fc0_1234, 0xffc0_4321] {
            assert_eq!(step_u32(bits, true), bits);
            assert_eq!(step_u32(bits, false), bits);
        }
        for bits in [
            0x7ff0_0000_0000_0001,
            0xfff0_0000_0000_0001,
            0x7ff8_1234_5678_1234,
            0xfff8_1234_5678_1234,
        ] {
            assert_eq!(step_u64(bits, true), bits);
            assert_eq!(step_u64(bits, false), bits);
        }
    }

    #[test]
    fn finite_encodings_match_standard_neighbors_at_every_exponent() {
        for exponent in 0..255u32 {
            for mantissa in [0, 1, 0x003f_ffff, 0x007f_ffff] {
                for sign in [0, 0x8000_0000] {
                    let bits = sign | (exponent << 23) | mantissa;
                    let value = f32::from_bits(bits);
                    assert_eq!(step_u32(bits, true), value.next_up().to_bits());
                    assert_eq!(step_u32(bits, false), value.next_down().to_bits());
                }
            }
        }
        for exponent in 0..2047u64 {
            for mantissa in [0, 1, 0x0007_ffff_ffff_ffff, 0x000f_ffff_ffff_ffff] {
                for sign in [0, 0x8000_0000_0000_0000] {
                    let bits = sign | (exponent << 52) | mantissa;
                    let value = f64::from_bits(bits);
                    assert_eq!(step_u64(bits, true), value.next_up().to_bits());
                    assert_eq!(step_u64(bits, false), value.next_down().to_bits());
                }
            }
        }
    }
}
