//! Sobol sequence direction number data.
//!
//! Contains Joe & Kuo (2008) primitive polynomial coefficients and initial
//! direction numbers for up to 21,201 dimensions.
//!
//! Data source: <https://web.maths.unsw.edu.au/~fkuo/sobol/>
//! License: Public domain

use rkyv::{Archive, Deserialize, Serialize};
use std::sync::OnceLock;

/// Embedded rkyv archive of Sobol polynomial data.
static SOBOL_DATA_BYTES: &[u8] = include_bytes!("sobol_data.bin");

/// Cached reference to the archived data.
static SOBOL_DATA: OnceLock<&'static ArchivedSobolData> = OnceLock::new();

/// Sobol primitive polynomial data for one dimension.
#[derive(Archive, Deserialize, Serialize, Debug)]
#[rkyv(crate = rkyv)]
pub struct SobolPolynomial {
    /// Polynomial degree (s)
    pub degree: u8,
    /// Polynomial coefficient (a)
    pub coeff: u32,
    /// Initial direction numbers (m_1, m_2, ..., m_s)
    pub m_values: Vec<u32>,
}

/// Complete Sobol direction number data.
#[derive(Archive, Deserialize, Serialize, Debug)]
#[rkyv(crate = rkyv)]
pub struct SobolData {
    /// Polynomials for dimensions 2, 3, 4, ... (dimension 1 is implicit)
    pub polynomials: Vec<SobolPolynomial>,
}

/// Get the archived Sobol data (zero-copy access).
///
/// Returns a reference to the archived data that can be accessed directly
/// without deserialization.
#[inline]
pub fn get_sobol_data() -> &'static ArchivedSobolData {
    SOBOL_DATA.get_or_init(|| {
        // SAFETY: The embedded bytes were serialized with the same rkyv version
        // and struct definitions. This is safe because we control both the
        // serialization (generate_sobol_rkyv tool) and deserialization.
        unsafe { rkyv::access_unchecked::<ArchivedSobolData>(SOBOL_DATA_BYTES) }
    })
}

/// Maximum supported dimension (21,201).
pub const MAX_SOBOL_DIMENSION: usize = 21201;

/// Get polynomial data for a specific dimension (2-indexed).
///
/// Dimension 1 uses implicit direction numbers (powers of 2).
/// Dimensions 2-21201 use the Joe & Kuo data.
///
/// Returns `None` if dimension is out of range.
#[inline]
pub fn get_polynomial(dimension: usize) -> Option<&'static ArchivedSobolPolynomial> {
    if dimension < 2 || dimension > MAX_SOBOL_DIMENSION {
        return None;
    }
    let data = get_sobol_data();
    data.polynomials.get(dimension - 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_sobol_data() {
        let data = get_sobol_data();
        assert_eq!(data.polynomials.len(), 21200);
    }

    #[test]
    fn test_dimension_2() {
        let poly = get_polynomial(2).unwrap();
        assert_eq!(poly.degree, 1);
        assert_eq!(poly.coeff, 0);
        assert_eq!(poly.m_values.len(), 1);
        assert_eq!(poly.m_values[0], 1);
    }

    #[test]
    fn test_dimension_3() {
        let poly = get_polynomial(3).unwrap();
        assert_eq!(poly.degree, 2);
        assert_eq!(poly.coeff, 1);
        assert_eq!(poly.m_values.len(), 2);
        assert_eq!(poly.m_values[0], 1);
        assert_eq!(poly.m_values[1], 3);
    }

    #[test]
    fn test_out_of_range() {
        assert!(get_polynomial(0).is_none());
        assert!(get_polynomial(1).is_none()); // Dimension 1 is implicit
        assert!(get_polynomial(21202).is_none());
    }
}
