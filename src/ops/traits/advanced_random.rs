//! Advanced PRNGs for reproducible parallel random number generation.
//!
//! Counter-based PRNGs enable reproducible parallel generation: same (key, counter) → same output.
//!
//! # Algorithms
//! - **Philox4x32-10**: JAX/TensorFlow default, excellent GPU performance
//! - **ThreeFry4x64-20**: Cryptographic quality, slower but secure
//! - **PCG64**: NumPy default, CPU-optimized
//! - **Xoshiro256++**: Rust `rand` default, excellent quality/speed
//!
//! # WebGPU Limitations
//! WGSL lacks native u64: uses 32-bit keys for Philox/ThreeFry, emulated u64 for PCG64/Xoshiro.
//! Only F32 supported on WebGPU.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Advanced PRNG operations for reproducible parallel random number generation.
///
/// # Reproducibility
/// Same (key, counter) or (seed, stream) → identical output.
/// Different counter/stream → independent sequences.
///
/// # Example
/// ```ignore
/// let samples = client.philox_randn(&[1000], 42, 0, DType::F32)?;
/// let same = client.philox_randn(&[1000], 42, 0, DType::F32)?;  // identical
/// let diff = client.philox_randn(&[1000], 42, 1, DType::F32)?;  // different
/// ```
pub trait AdvancedRandomOps<R: Runtime> {
    /// Generate N(0,1) samples using Philox4x32-10.
    /// Args: shape, key (seed), counter, dtype (F32/F64; WebGPU: F32 only)
    fn philox_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Generate uniform [0,1) samples using Philox4x32-10.
    fn philox_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Generate N(0,1) samples using ThreeFry4x64-20 (cryptographic quality).
    fn threefry_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Generate uniform [0,1) samples using ThreeFry4x64-20.
    fn threefry_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Generate N(0,1) samples using PCG64.
    /// Args: shape, seed, stream (for parallel generation), dtype
    fn pcg64_randn(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Generate uniform [0,1) samples using PCG64.
    fn pcg64_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Generate N(0,1) samples using Xoshiro256++.
    /// Args: shape, seed, dtype
    fn xoshiro256_randn(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Generate uniform [0,1) samples using Xoshiro256++.
    fn xoshiro256_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>>;
}
