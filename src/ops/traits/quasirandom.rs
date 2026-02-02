//! Quasi-random sequence generation operations.
//!
//! This module defines the `QuasiRandomOps` trait for generating low-discrepancy
//! sequences used in quasi-Monte Carlo (QMC) methods.
//!
//! # Quasi-Random Sequences
//!
//! Unlike pseudo-random sequences, quasi-random (low-discrepancy) sequences are
//! designed to cover the sampling space more uniformly. This leads to faster
//! convergence in numerical integration and optimization compared to standard
//! Monte Carlo methods.
//!
//! # Common Applications
//!
//! - Quasi-Monte Carlo integration (QMC)
//! - Global optimization
//! - Sensitivity analysis
//! - Computational finance (option pricing)
//! - Computer graphics (ray tracing)
//!
//! # Backend Support
//!
//! ## Data Types
//!
//! - **CPU**: Supports F32, F64
//! - **CUDA**: Supports F32, F64
//! - **WebGPU**: F32 only (platform limitation)
//!
//! **Note on WebGPU F64 limitation:** WebGPU/WGSL has limited native F64 support,
//! requiring hardware extensions that are not universally available. This F32-only
//! constraint affects ALL WebGPU operations in numr, not just quasi-random sequences.
//! For F64 precision, use CPU or CUDA backends.
//!
//! All sequences generate points in the unit hypercube [0, 1)^d.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Quasi-random sequence generation operations.
///
/// Generates low-discrepancy sequences that provide better coverage of the
/// sampling space compared to pseudo-random sequences, leading to faster
/// convergence in numerical integration and optimization.
pub trait QuasiRandomOps<R: Runtime> {
    /// Generate Sobol sequence points.
    ///
    /// The Sobol sequence is a quasi-random low-discrepancy sequence that provides
    /// excellent uniform coverage of the unit hypercube. It's based on the
    /// Gray code and direction numbers.
    ///
    /// # Properties
    ///
    /// - Deterministic (same parameters always produce same sequence)
    /// - Low discrepancy (uniform coverage)
    /// - Base-2 construction (efficient)
    /// - Best performance when n_points is a power of 2
    ///
    /// # Current Implementation Limitation
    ///
    /// **Only 6 dimensions have true Sobol direction numbers.** Dimensions 7 and beyond
    /// fall back to van der Corput sequence, which has worse discrepancy properties.
    /// For high-quality quasi-random sequences in >6 dimensions, use Halton or consider
    /// contributing direction numbers from Joe & Kuo (2008).
    ///
    /// # Arguments
    ///
    /// * `n_points` - Number of points to generate
    /// * `dimension` - Dimensionality of the space (recommended ≤ 6 for true Sobol)
    /// * `skip` - Number of initial points to skip (for sequence continuation)
    /// * `dtype` - Output data type (F32 or F64 for CPU/CUDA; F32 only for WebGPU)
    ///
    /// # Returns
    ///
    /// Tensor with shape `(n_points, dimension)` containing points in [0, 1)^d
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `n_points` is 0
    /// - `dimension` is 0 or exceeds implementation limits (currently 6 for true Sobol)
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not supported by the backend.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate 1000 points in 10 dimensions for QMC integration
    /// let points = client.sobol(1000, 10, 0, DType::F64)?;
    /// assert_eq!(points.shape(), &[1000, 10]);
    /// ```
    fn sobol(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<R>>;

    /// Generate Halton sequence points.
    ///
    /// The Halton sequence is a quasi-random low-discrepancy sequence based on
    /// van der Corput sequences in different prime bases. Simple to compute but
    /// can suffer from correlation artifacts in higher dimensions.
    ///
    /// # Properties
    ///
    /// - Deterministic
    /// - Simple construction using prime bases
    /// - Good for low dimensions (typically ≤ 10-20)
    /// - Can have correlation issues in higher dimensions
    /// - Randomized variants (Halton-Braaten-Weller) reduce correlations
    ///
    /// # Arguments
    ///
    /// * `n_points` - Number of points to generate
    /// * `dimension` - Dimensionality of the space
    /// * `skip` - Number of initial points to skip
    /// * `dtype` - Output data type (F32 or F64)
    ///
    /// # Returns
    ///
    /// Tensor with shape `(n_points, dimension)` containing points in [0, 1)^d
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `n_points` is 0
    /// - `dimension` is 0 or exceeds available primes (typically ≤ 100)
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not F32 or F64.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate 500 points in 5 dimensions
    /// let points = client.halton(500, 5, 0, DType::F64)?;
    /// assert_eq!(points.shape(), &[500, 5]);
    /// ```
    fn halton(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<R>>;

    /// Generate Latin Hypercube Sampling points.
    ///
    /// Latin Hypercube Sampling (LHS) divides each dimension into n equal intervals
    /// and samples exactly once from each interval, ensuring good coverage.
    /// The randomized nature means different calls produce different samples.
    ///
    /// # Properties
    ///
    /// - Randomized (different calls produce different samples)
    /// - Excellent stratification in each dimension
    /// - Better space-filling than simple random sampling
    /// - Useful for design of experiments and sensitivity analysis
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of samples to generate
    /// * `dimension` - Dimensionality of the space
    /// * `dtype` - Output data type (F32 or F64)
    ///
    /// # Returns
    ///
    /// Tensor with shape `(n_samples, dimension)` containing points in [0, 1)^d
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `n_samples` is 0
    /// - `dimension` is 0
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not F32 or F64.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate 100 LHS samples in 8 dimensions
    /// let samples = client.latin_hypercube(100, 8, DType::F64)?;
    /// assert_eq!(samples.shape(), &[100, 8]);
    /// ```
    fn latin_hypercube(
        &self,
        n_samples: usize,
        dimension: usize,
        dtype: DType,
    ) -> Result<Tensor<R>>;
}
