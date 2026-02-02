//! Shared validation logic for quasi-random sequence operations.
//!
//! This module provides common parameter validation to ensure consistency
//! across CPU, CUDA, and WebGPU backends and eliminate code duplication.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Maximum number of dimensions supported for Sobol sequences with full direction numbers.
///
/// **Current implementation:** Only 6 dimensions have precomputed direction numbers.
/// Dimensions beyond 6 will use van der Corput fallback (different algorithm).
///
/// **TODO:** Expand to 1000 dimensions using Joe & Kuo (2008) direction numbers.
/// This would require including ~120KB of direction number data.
pub const SOBOL_MAX_DIMENSIONS: usize = 6;

/// Maximum number of dimensions supported for Halton sequences.
pub const HALTON_MAX_DIMENSIONS: usize = 100;

/// Validates common parameters for quasi-random sequence generation.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate (must be > 0)
/// * `dimension` - Dimensionality of the sequence (must be > 0 and <= max_dim)
/// * `max_dim` - Maximum dimension supported for this algorithm
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns `InvalidArgument` if:
/// - `n_points` is 0
/// - `dimension` is 0
/// - `dimension` exceeds `max_dim`
#[inline]
pub fn validate_basic_params(
    n_points: usize,
    dimension: usize,
    max_dim: usize,
    op: &'static str,
) -> Result<()> {
    if n_points == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_points",
            reason: format!("{} requires at least 1 point", op),
        });
    }

    if dimension == 0 {
        return Err(Error::InvalidArgument {
            arg: "dimension",
            reason: format!("{} requires at least 1 dimension", op),
        });
    }

    if dimension > max_dim {
        return Err(Error::InvalidArgument {
            arg: "dimension",
            reason: format!(
                "{} supports up to {} dimensions, got {}",
                op, max_dim, dimension
            ),
        });
    }

    Ok(())
}

/// Validates data type support for quasi-random operations.
///
/// # Arguments
///
/// * `dtype` - Data type to validate
/// * `supported_dtypes` - Slice of supported DType values
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns `UnsupportedDType` if the dtype is not in the supported list.
#[inline]
pub fn validate_dtype(dtype: DType, supported_dtypes: &[DType], op: &'static str) -> Result<()> {
    if !supported_dtypes.contains(&dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validates parameters for Sobol sequence generation.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `dimension` - Dimensionality of the sequence
/// * `dtype` - Data type
/// * `supported_dtypes` - Slice of supported DType values for this backend
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns error if basic validation or dtype validation fails.
#[inline]
pub fn validate_sobol_params(
    n_points: usize,
    dimension: usize,
    dtype: DType,
    supported_dtypes: &[DType],
    op: &'static str,
) -> Result<()> {
    validate_basic_params(n_points, dimension, SOBOL_MAX_DIMENSIONS, op)?;
    validate_dtype(dtype, supported_dtypes, op)?;
    Ok(())
}

/// Validates parameters for Halton sequence generation.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `dimension` - Dimensionality of the sequence
/// * `dtype` - Data type
/// * `supported_dtypes` - Slice of supported DType values for this backend
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns error if basic validation or dtype validation fails.
#[inline]
pub fn validate_halton_params(
    n_points: usize,
    dimension: usize,
    dtype: DType,
    supported_dtypes: &[DType],
    op: &'static str,
) -> Result<()> {
    validate_basic_params(n_points, dimension, HALTON_MAX_DIMENSIONS, op)?;
    validate_dtype(dtype, supported_dtypes, op)?;
    Ok(())
}

/// Validates parameters for Latin Hypercube Sampling.
///
/// # Arguments
///
/// * `n_samples` - Number of samples to generate
/// * `dimension` - Dimensionality of the samples
/// * `dtype` - Data type
/// * `supported_dtypes` - Slice of supported DType values for this backend
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns error if validation fails. Note: Latin Hypercube has no
/// dimension limit (other than memory constraints).
#[inline]
pub fn validate_latin_hypercube_params(
    n_samples: usize,
    dimension: usize,
    dtype: DType,
    supported_dtypes: &[DType],
    op: &'static str,
) -> Result<()> {
    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_samples",
            reason: format!("{} requires at least 1 sample", op),
        });
    }

    if dimension == 0 {
        return Err(Error::InvalidArgument {
            arg: "dimension",
            reason: format!("{} requires at least 1 dimension", op),
        });
    }

    validate_dtype(dtype, supported_dtypes, op)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_basic_params() {
        // Valid parameters
        assert!(validate_basic_params(10, 5, 100, "test").is_ok());

        // Zero points
        assert!(validate_basic_params(0, 5, 100, "test").is_err());

        // Zero dimension
        assert!(validate_basic_params(10, 0, 100, "test").is_err());

        // Dimension exceeds limit
        assert!(validate_basic_params(10, 101, 100, "test").is_err());
    }

    #[test]
    fn test_validate_dtype() {
        let supported = &[DType::F32, DType::F64];

        // Valid dtype
        assert!(validate_dtype(DType::F32, supported, "test").is_ok());
        assert!(validate_dtype(DType::F64, supported, "test").is_ok());

        // Unsupported dtype
        assert!(validate_dtype(DType::I32, supported, "test").is_err());
    }

    #[test]
    fn test_validate_sobol_params() {
        let supported = &[DType::F32, DType::F64];

        // Valid parameters
        assert!(validate_sobol_params(10, 5, DType::F32, supported, "sobol").is_ok());

        // Invalid dimension
        assert!(validate_sobol_params(10, 1001, DType::F32, supported, "sobol").is_err());

        // Unsupported dtype
        assert!(validate_sobol_params(10, 5, DType::I32, supported, "sobol").is_err());
    }

    #[test]
    fn test_validate_halton_params() {
        let supported = &[DType::F32, DType::F64];

        // Valid parameters
        assert!(validate_halton_params(10, 5, DType::F32, supported, "halton").is_ok());

        // Invalid dimension (Halton max is 100)
        assert!(validate_halton_params(10, 101, DType::F32, supported, "halton").is_err());
    }

    #[test]
    fn test_validate_latin_hypercube_params() {
        let supported = &[DType::F32, DType::F64];

        // Valid parameters
        assert!(
            validate_latin_hypercube_params(10, 5, DType::F32, supported, "latin_hypercube")
                .is_ok()
        );

        // Zero samples
        assert!(
            validate_latin_hypercube_params(0, 5, DType::F32, supported, "latin_hypercube")
                .is_err()
        );

        // Zero dimension
        assert!(
            validate_latin_hypercube_params(10, 0, DType::F32, supported, "latin_hypercube")
                .is_err()
        );
    }
}
