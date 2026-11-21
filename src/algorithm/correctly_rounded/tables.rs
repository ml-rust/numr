//! Constant tables of the correctly rounded kernels.
//!
//! `tables_data.rs` is the one source of every table. It holds flat `u64` arrays of
//! binary64 bit patterns, or of dint words `(hi, lo, ex, sgn)`. The CUDA build reads the
//! same file. Each table documents its layout.

include!("tables_data.rs");

/// Entry `index` of a table of binary64 bit patterns.
#[inline(always)]
pub fn f(table: &[u64], index: usize) -> f64 {
    f64::from_bits(table[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tables_hold_expected_anchor_values() {
        assert_eq!(f(&EXP_T0, 1), 1.0);
        assert_eq!(f(&EXP_T1, 1), 1.0);
        assert_eq!(f(&EXP_T0, 65), std::f64::consts::SQRT_2);
        assert_eq!(f(&LOG_INVERSE, 511 - 362), 1.0);
        assert_eq!(f(&LOG_INVERSE, 512 - 362), 1.0);
        assert_eq!(f(&LOG_FAST_POLY, 0), 1.0);
        assert_eq!(f(&LOG1P_RF, 37), 1.0);
        assert_eq!(f(&LOG1P_RT, 8), 1.0);
        assert_eq!(LOG_DINT_POLY.len(), 4 * 13);
        assert_eq!(LOG_DINT_INVERSE.len(), LOG_DINT_LOG_INV.len());
        assert_eq!(f(&ACOS_CC, 0), 1.0);
        assert_eq!(ACOS_CC[1], 0);
        assert_eq!(ACOS_CC.len(), 8 * 33);
        assert_eq!(f(&ACOS_S, 2 * 16 + 1), std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(f(&ACOS_S, 2 * 32 + 1), 1.0);
        assert_eq!(f(&ACOS_C, 0), 1.0);
        assert_eq!(f(&ACOS_C, 2), f(&ACOS_CC, 2));
        assert_eq!(ACOS_CT.len(), 3);
        assert_eq!(ACOS_EXCEPTIONS.len(), 3 * 5);
    }

    #[test]
    fn test_acos_sine_table_matches_libm_to_one_ulp() {
        for j in 1..32 {
            let want = (j as f64 * std::f64::consts::PI / 64.0).sin();
            let hi = f(&ACOS_S, 2 * j + 1);
            assert!((hi - want).abs() <= 2.0 * f64::EPSILON * want, "j {j}");
            assert!(f(&ACOS_S, 2 * j).abs() <= f64::EPSILON * hi, "j {j}");
        }
    }

    #[test]
    fn test_exception_inputs_are_sorted() {
        for pair in EXP_EXCEPTIONS.chunks(3).collect::<Vec<_>>().windows(2) {
            assert!(f64::from_bits(pair[0][0]) < f64::from_bits(pair[1][0]));
        }
    }
}
