//! The [`Kernels`] handle: one SIMD level this CPU supports.

use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// A SIMD level for the distance kernels, checked against this CPU.
///
/// # Invariant
///
/// A `Kernels` value only ever holds a level this CPU can run. The three
/// constructors are the only way to build one, and each of them upholds this.
/// Every method calls an `unsafe` kernel at that level, and this invariant is
/// what makes those calls sound. A level the CPU lacks will fault with an
/// illegal instruction.
///
/// [`Kernels::detect`] is a cached read, so the free functions in
/// [`crate::distance`] cost one load more than a stored `Kernels`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Kernels {
    level: SimdLevel,
}

impl Kernels {
    /// The best level this CPU supports.
    ///
    /// The first call runs CPU feature detection. Later calls read the cached
    /// result.
    #[inline]
    pub fn detect() -> Self {
        Self {
            level: detect_simd(),
        }
    }

    /// The scalar level. Every CPU supports it.
    #[inline]
    pub fn scalar() -> Self {
        Self {
            level: SimdLevel::Scalar,
        }
    }

    /// `level`, if this CPU supports it, else `None`.
    ///
    /// Supported levels:
    /// - [`SimdLevel::Scalar`], always.
    /// - The detected level.
    /// - [`SimdLevel::Avx2Fma`] when the detected level is [`SimdLevel::Avx512`].
    /// - [`SimdLevel::Neon`] when the detected level is [`SimdLevel::NeonFp16`].
    pub fn with_level(level: SimdLevel) -> Option<Self> {
        let top = detect_simd();
        let supported = level == SimdLevel::Scalar
            || level == top
            || (top == SimdLevel::Avx512 && level == SimdLevel::Avx2Fma)
            || (top == SimdLevel::NeonFp16 && level == SimdLevel::Neon);
        supported.then_some(Self { level })
    }

    /// The SIMD level every method of this handle runs.
    #[inline]
    pub fn level(&self) -> SimdLevel {
        self.level
    }
}
