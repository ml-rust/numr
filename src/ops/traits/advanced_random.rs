//! Advanced PRNGs for reproducible parallel random number generation.
//!
//! Counter-based PRNGs enable reproducible parallel generation: same (key, counter) → same output.
//!
//! # Algorithms
//! - **Philox4x32-10**: JAX/TensorFlow default, excellent GPU performance
//! - **ThreeFry4x64-20**: Counter-based, slower than Philox. Not a cryptographic RNG.
//! - **PCG64**: NumPy default, CPU-optimized
//! - **Xoshiro256++**: Rust `rand` default, excellent quality/speed
//!
//! # WebGPU Limitations
//! WGSL lacks native u64. Philox and ThreeFry carry 64-bit keys and counters as u32 pairs.
//! PCG64 and Xoshiro emulate u64. Only F32 is supported on WebGPU.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Advanced PRNG operations for reproducible parallel random number generation.
///
/// # Reproducibility
/// Same (key, counter) or (seed, stream) → identical output.
/// Different counter/stream → independent sequences.
///
/// # Example
/// ```
/// # use numr::prelude::*;
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// use numr::ops::AdvancedRandomOps;
///
/// let samples = client.philox_randn(&[1000], 42, 0, DType::F32)?;
/// let same = client.philox_randn(&[1000], 42, 0, DType::F32)?;  // identical
/// let diff = client.philox_randn(&[1000], 42, 1, DType::F32)?;  // different
/// # Ok::<(), numr::error::Error>(())
/// ```
pub trait AdvancedRandomOps<R: Runtime> {
    /// Generate N(0,1) samples using Philox4x32-10.
    ///
    /// Args: shape, key (seed), counter, dtype.
    /// Dtypes: CPU every float dtype, CUDA F32/F64/F16/BF16, WebGPU F32.
    ///
    /// Uses the block layout of [`philox_uniform`](Self::philox_uniform).
    /// Box-Muller maps words (0, 1) and (2, 3) of block `c` to elements `4c..4c+3`.
    /// Element `i` depends on `(key, counter, i)` only, never on the shape.
    /// Backends agree within floating-point tolerance, not bit for bit:
    /// `ln`, `sqrt`, `cos` and `sin` round differently per backend.
    fn philox_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, key, counter, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::philox_randn",
        })
    }

    /// Generate uniform [0,1) samples using Philox4x32-10.
    ///
    /// Dtypes: CPU every float dtype, CUDA F32/F64/F16/BF16, WebGPU F32.
    ///
    /// Contract, checked by known-answer and cross-backend tests:
    /// - Block `c` is Philox4x32-10 (Random123) of counter `[lo, hi, 0, 0]` and
    ///   key `[key_lo, key_hi]`, where `lo, hi` are the words of `counter + c` (u64 wrapping).
    /// - Element `4c + j` is word `j` of block `c`, as `(word >> 8) / 2^24`.
    ///   That value is exact in F32 and F64. F16 and BF16 round it to nearest even.
    ///   A value that rounds to 1.0 clamps to the dtype's largest value below 1.0.
    /// - Element `i` depends on `(key, counter, i)` only. A longer draw extends a shorter one.
    /// - F32 output is bit-identical on CPU, CUDA and WebGPU. F64, F16 and BF16 output is
    ///   bit-identical on CPU and CUDA. F64 equals the F32 output widened.
    fn philox_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, key, counter, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::philox_uniform",
        })
    }

    /// Generate N(0,1) samples using ThreeFry4x64-20 (counter-based, not cryptographic).
    fn threefry_randn(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, key, counter, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::threefry_randn",
        })
    }

    /// Generate uniform [0,1) samples using ThreeFry4x64-20.
    fn threefry_uniform(
        &self,
        shape: &[usize],
        key: u64,
        counter: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, key, counter, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::threefry_uniform",
        })
    }

    /// Generate N(0,1) samples using PCG64.
    /// Args: shape, seed, stream (for parallel generation), dtype
    fn pcg64_randn(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, seed, stream, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::pcg64_randn",
        })
    }

    /// Generate uniform [0,1) samples using PCG64.
    fn pcg64_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        stream: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, seed, stream, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::pcg64_uniform",
        })
    }

    /// Generate N(0,1) samples using Xoshiro256++.
    /// Args: shape, seed, dtype
    fn xoshiro256_randn(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, seed, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::xoshiro256_randn",
        })
    }

    /// Generate uniform [0,1) samples using Xoshiro256++.
    fn xoshiro256_uniform(
        &self,
        shape: &[usize],
        seed: u64,
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape, seed, dtype);
        Err(Error::NotImplemented {
            feature: "AdvancedRandomOps::xoshiro256_uniform",
        })
    }
}
