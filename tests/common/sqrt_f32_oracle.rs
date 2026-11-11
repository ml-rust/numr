//! Integer dyadic references for binary32 square roots.

use std::cmp::Ordering;
use std::sync::OnceLock;

const MAX_FINITE: u32 = 0x7f7f_ffff;
const INFINITY: u32 = 0x7f80_0000;

pub const WITNESSES: [u32; 20] = [
    0,
    0x8000_0000,
    1,
    2,
    3,
    0x007f_ffff,
    0x0080_0000,
    0x0080_0001,
    0x3f7f_ffff,
    0x3f80_0000,
    0x3f80_0001,
    0x3f80_0002,
    0x3fff_ffff,
    0x4000_0000,
    0x4000_0001,
    0x407f_ffff,
    0x4080_0001,
    0x7eff_ffff,
    MAX_FINITE,
    INFINITY,
];

#[derive(Clone, Copy)]
struct Dyadic {
    significand: u128,
    exponent: i32,
}
impl Dyadic {
    fn from_bits(bits: u32) -> Self {
        assert!(bits <= MAX_FINITE);
        let exponent = (bits >> 23) as i32;
        let fraction = u128::from(bits & 0x007f_ffff);
        if exponent == 0 {
            Self {
                significand: fraction,
                exponent: -149,
            }
        } else {
            Self {
                significand: (1 << 23) | fraction,
                exponent: exponent - 150,
            }
        }
    }

    fn square(self) -> Self {
        Self {
            significand: self.significand * self.significand,
            exponent: 2 * self.exponent,
        }
    }

    fn compare(self, other: Self) -> Ordering {
        if self.significand == 0 || other.significand == 0 {
            return self.significand.cmp(&other.significand);
        }
        let highest = 127 - self.significand.leading_zeros() as i32 + self.exponent;
        let other_highest = 127 - other.significand.leading_zeros() as i32 + other.exponent;
        if highest != other_highest {
            return highest.cmp(&other_highest);
        }
        // Equal highest bits limit alignment shifts to the significand-width difference.
        // Inputs use at most 24 bits; squared midpoints use at most 50 bits.
        let common = self.exponent.min(other.exponent);
        let left_shift = (self.exponent - common) as u32;
        let right_shift = (other.exponent - common) as u32;
        assert!(left_shift < 128 && right_shift < 128);
        (self.significand << left_shift).cmp(&(other.significand << right_shift))
    }
}

fn squared_midpoint(lower: u32, upper: u32) -> Dyadic {
    let left = Dyadic::from_bits(lower);
    let right = Dyadic::from_bits(upper);
    let exponent = left.exponent.min(right.exponent);
    let significand = (left.significand << (left.exponent - exponent) as u32)
        + (right.significand << (right.exponent - exponent) as u32);
    Dyadic {
        significand,
        exponent: exponent - 1,
    }
    .square()
}

fn floor_float_bits(value: Dyadic) -> u32 {
    let mut lower = 0;
    let mut upper = INFINITY;
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2;
        if Dyadic::from_bits(middle).compare(value) != Ordering::Greater {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    lower
}

/// Compute RN-even sqrt bits using exact comparisons, without floating arithmetic.
pub fn rounded_sqrt_bits(bits: u32) -> u32 {
    if bits == 0 || bits == 0x8000_0000 || bits == INFINITY {
        return bits;
    }
    assert!(bits <= MAX_FINITE);
    let value = Dyadic::from_bits(bits);
    let mut lower = 0;
    let mut upper = INFINITY;
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2;
        if Dyadic::from_bits(middle).square().compare(value) != Ordering::Greater {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    if Dyadic::from_bits(lower).square().compare(value) == Ordering::Equal {
        return lower;
    }
    match value.compare(squared_midpoint(lower, upper)) {
        Ordering::Less => lower,
        Ordering::Greater => upper,
        Ordering::Equal => {
            if lower & 1 == 0 {
                lower
            } else {
                upper
            }
        }
    }
}

/// Fixed 200,308 inputs cover every exponent, adjacent mantissas, and root-rounding boundaries.
pub fn cases() -> &'static [(u32, u32)] {
    static CASES: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
    CASES.get_or_init(|| {
        for (input, output) in [
            (0, 0),
            (0x8000_0000, 0x8000_0000),
            (1, 0x1a35_04f3),
            (2, 0x1a80_0000),
            (0x0080_0000, 0x2000_0000),
            (0x3f7f_ffff, 0x3f7f_ffff),
            (0x3f80_0000, 0x3f80_0000),
            (0x3f80_0001, 0x3f80_0000),
            (0x3f80_0002, 0x3f80_0001),
            (0x4000_0000, 0x3fb5_04f3),
            (0x4080_0000, 0x4000_0000),
            (0x4110_0000, 0x4040_0000),
            (MAX_FINITE, 0x5f7f_ffff),
            (INFINITY, INFINITY),
        ] {
            assert_eq!(
                rounded_sqrt_bits(input),
                output,
                "oracle spot input={input:#010x}"
            );
        }
        let mut inputs = Vec::with_capacity(200_308);
        inputs.extend(WITNESSES);
        for exponent in 0..255u32 {
            for fraction in 0..256u32 {
                let center = (exponent << 23) | (fraction << 15);
                for delta in [-1i64, 0, 1] {
                    inputs.push((i64::from(center) + delta).clamp(0, i64::from(MAX_FINITE)) as u32);
                }
            }
        }
        // These output binades span every possible finite binary32 square root.
        // Probe represented radicands next to their exact squared rounding boundaries.
        for exponent in 52..=190u32 {
            for fraction in [
                0, 1, 0x10_0000, 0x20_0000, 0x3f_ffff, 0x40_0000, 0x7f_fffe, 0x7f_ffff,
            ] {
                let lower = (exponent << 23) | fraction;
                let boundary = squared_midpoint(lower, lower + 1);
                let center = floor_float_bits(boundary);
                for delta in [-1i64, 0, 1, 2] {
                    inputs.push((i64::from(center) + delta).clamp(0, i64::from(MAX_FINITE)) as u32);
                }
            }
        }
        assert_eq!(inputs.len(), 200_308);
        inputs
            .into_iter()
            .map(|bits| (bits, rounded_sqrt_bits(bits)))
            .collect()
    })
}
