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
    }

    #[test]
    fn test_exception_inputs_are_sorted() {
        for pair in EXP_EXCEPTIONS.chunks(3).collect::<Vec<_>>().windows(2) {
            assert!(f64::from_bits(pair[0][0]) < f64::from_bits(pair[1][0]));
        }
    }
}
