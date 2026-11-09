//! Exact original-F32 division references across shared dispatch routes.

#[derive(Clone, Copy)]
pub struct DivisionCase {
    pub id: &'static str,
    pub numerator: u32,
    pub denominator: u32,
    pub expected: Option<u32>,
}

impl DivisionCase {
    pub fn check(self, actual: f32, route: &str) {
        if let Some(expected) = self.expected {
            assert_eq!(
                actual.to_bits(),
                expected,
                "{} route={route},numerator={:08x},denominator={:08x}",
                self.id,
                self.numerator,
                self.denominator
            );
        } else {
            assert!(
                actual.is_nan(),
                "{} route={route},numerator={:08x},denominator={:08x}: expected NaN,actual={:08x}",
                self.id,
                self.numerator,
                self.denominator,
                actual.to_bits()
            );
        }
    }
}

pub const CASES: [DivisionCase; 32] = [
    DivisionCase {
        id: "positive_third",
        numerator: 0x3f800000,
        denominator: 0x40400000,
        expected: Some(0x3eaaaaab),
    },
    DivisionCase {
        id: "negative_third",
        numerator: 0xbf800000,
        denominator: 0x40400000,
        expected: Some(0xbeaaaaab),
    },
    DivisionCase {
        id: "ten_sevenths",
        numerator: 0x41200000,
        denominator: 0x40e00000,
        expected: Some(0x3fb6db6e),
    },
    DivisionCase {
        id: "adjacent_mantissas",
        numerator: 0x3f800001,
        denominator: 0x3f7fffff,
        expected: Some(0x3f800002),
    },
    DivisionCase {
        id: "normal_mantissas",
        numerator: 0x3f123456,
        denominator: 0x3f654321,
        expected: Some(0x3f23416d),
    },
    DivisionCase {
        id: "large_equal_operands",
        numerator: 0x7f000000,
        denominator: 0x7f000000,
        expected: Some(0x3f800000),
    },
    DivisionCase {
        id: "minimum_normal_halved",
        numerator: 0x00800000,
        denominator: 0x40000000,
        expected: Some(0x00400000),
    },
    DivisionCase {
        id: "minimum_subnormal_half_tie",
        numerator: 0x00000001,
        denominator: 0x40000000,
        expected: Some(0x00000000),
    },
    DivisionCase {
        id: "three_subnormals_half_tie",
        numerator: 0x00000003,
        denominator: 0x40000000,
        expected: Some(0x00000002),
    },
    DivisionCase {
        id: "negative_minimum_subnormal_half_tie",
        numerator: 0x80000001,
        denominator: 0x40000000,
        expected: Some(0x80000000),
    },
    DivisionCase {
        id: "negative_three_subnormals_half_tie",
        numerator: 0x80000003,
        denominator: 0x40000000,
        expected: Some(0x80000002),
    },
    DivisionCase {
        id: "equal_minimum_subnormals",
        numerator: 0x00000001,
        denominator: 0x00000001,
        expected: Some(0x3f800000),
    },
    DivisionCase {
        id: "minimum_normal_over_subnormal",
        numerator: 0x00800000,
        denominator: 0x00000001,
        expected: Some(0x4b000000),
    },
    DivisionCase {
        id: "positive_zero_over_one",
        numerator: 0x00000000,
        denominator: 0x3f800000,
        expected: Some(0x00000000),
    },
    DivisionCase {
        id: "negative_zero_over_one",
        numerator: 0x80000000,
        denominator: 0x3f800000,
        expected: Some(0x80000000),
    },
    DivisionCase {
        id: "positive_zero_over_negative_one",
        numerator: 0x00000000,
        denominator: 0xbf800000,
        expected: Some(0x80000000),
    },
    DivisionCase {
        id: "negative_zero_over_negative_one",
        numerator: 0x80000000,
        denominator: 0xbf800000,
        expected: Some(0x00000000),
    },
    DivisionCase {
        id: "one_over_positive_zero",
        numerator: 0x3f800000,
        denominator: 0x00000000,
        expected: Some(0x7f800000),
    },
    DivisionCase {
        id: "one_over_negative_zero",
        numerator: 0x3f800000,
        denominator: 0x80000000,
        expected: Some(0xff800000),
    },
    DivisionCase {
        id: "negative_one_over_negative_zero",
        numerator: 0xbf800000,
        denominator: 0x80000000,
        expected: Some(0x7f800000),
    },
    DivisionCase {
        id: "infinity_over_two",
        numerator: 0x7f800000,
        denominator: 0x40000000,
        expected: Some(0x7f800000),
    },
    DivisionCase {
        id: "negative_infinity_over_negative_two",
        numerator: 0xff800000,
        denominator: 0xc0000000,
        expected: Some(0x7f800000),
    },
    DivisionCase {
        id: "two_over_infinity",
        numerator: 0x40000000,
        denominator: 0x7f800000,
        expected: Some(0x00000000),
    },
    DivisionCase {
        id: "negative_two_over_infinity",
        numerator: 0xc0000000,
        denominator: 0x7f800000,
        expected: Some(0x80000000),
    },
    DivisionCase {
        id: "infinity_over_infinity",
        numerator: 0x7f800000,
        denominator: 0x7f800000,
        expected: None,
    },
    DivisionCase {
        id: "zero_over_zero",
        numerator: 0x00000000,
        denominator: 0x00000000,
        expected: None,
    },
    DivisionCase {
        id: "nan_over_one",
        numerator: 0x7fc00001,
        denominator: 0x3f800000,
        expected: None,
    },
    DivisionCase {
        id: "one_over_nan",
        numerator: 0x3f800000,
        denominator: 0x7fc00001,
        expected: None,
    },
    DivisionCase {
        id: "maximum_equal_operands",
        numerator: 0x7f7fffff,
        denominator: 0x7f7fffff,
        expected: Some(0x3f800000),
    },
    DivisionCase {
        id: "overflowing_quotient",
        numerator: 0x7f7fffff,
        denominator: 0x00800000,
        expected: Some(0x7f800000),
    },
    DivisionCase {
        id: "underflowing_quotient",
        numerator: 0x00800000,
        denominator: 0x7f7fffff,
        expected: Some(0x00000000),
    },
    DivisionCase {
        id: "large_operand_over_three",
        numerator: 0x7f000000,
        denominator: 0x40400000,
        expected: Some(0x7e2aaaab),
    },
];
