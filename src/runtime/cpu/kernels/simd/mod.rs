//! SIMD detection and dispatch for CPU kernels
//!
//! This module provides runtime CPU feature detection. Each operation type
//! (matmul, binary, unary, reduce) has its own submodule with SIMD kernels.
//!
//! # Module Structure
//!
//! ```text
//! simd/
//! ├── mod.rs              # This file: detection only
//! ├── matmul/             # Matrix multiplication kernels
//! │   ├── mod.rs          # Tiled algorithm + dispatch
//! │   ├── avx512.rs       # AVX-512 microkernels
//! │   └── avx2.rs         # AVX2 microkernels
//! ├── binary/             # Binary ops (add, sub, mul, div, max, min)
//! │   ├── mod.rs          # Dispatch logic
//! │   ├── avx512.rs       # AVX-512 kernels (16 f32s, 8 f64s)
//! │   └── avx2.rs         # AVX2 kernels (8 f32s, 4 f64s)
//! ├── unary/              # Unary ops (neg, abs, sqrt, relu, etc.)
//! │   ├── mod.rs          # Dispatch logic
//! │   ├── avx512.rs       # AVX-512 kernels
//! │   └── avx2.rs         # AVX2 kernels
//! ├── compare/            # Compare ops (eq, ne, lt, le, gt, ge)
//! │   ├── mod.rs          # Dispatch logic
//! │   ├── avx512.rs       # AVX-512 kernels
//! │   └── avx2.rs         # AVX2 kernels
//! │
//! Note: Scalar fallbacks live in kernels/{binary,unary,compare}.rs (single source of truth)
//! └── reduce/             # Reductions (sum, max, min, prod)
//!     ├── mod.rs          # Dispatch logic
//!     ├── avx512.rs       # AVX-512 kernels
//!     └── avx2.rs         # AVX2 kernels
//! ```
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Vector Width | Status |
//! |--------------|-----------------|--------------|--------|
//! | x86-64       | AVX-512F + FMA  | 512 bits     | Supported |
//! | x86-64       | AVX2 + FMA      | 256 bits     | Supported |
//! | x86-64       | Scalar          | N/A          | Fallback |
//! | ARM64        | NEON            | 128 bits     | Planned |

#[cfg(target_arch = "x86_64")]
pub mod binary;
#[cfg(target_arch = "x86_64")]
pub mod compare;
#[cfg(target_arch = "x86_64")]
pub mod matmul;
#[cfg(target_arch = "x86_64")]
pub mod norm;
#[cfg(target_arch = "x86_64")]
pub mod reduce;
#[cfg(target_arch = "x86_64")]
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod unary;

use std::sync::OnceLock;

/// SIMD capability level detected at runtime
///
/// Ordered from most capable to least capable.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// AVX-512F with FMA support (512-bit vectors, 16 f32s or 8 f64s)
    Avx512 = 3,
    /// AVX2 with FMA support (256-bit vectors, 8 f32s or 4 f64s)
    Avx2Fma = 2,
    /// Scalar fallback (no SIMD)
    Scalar = 0,
}

impl SimdLevel {
    /// Returns true if this level supports 512-bit operations
    #[inline]
    #[allow(dead_code)] // Reserved for future dispatch logic
    pub const fn has_avx512(self) -> bool {
        matches!(self, Self::Avx512)
    }

    /// Returns true if this level supports 256-bit operations
    #[inline]
    #[allow(dead_code)] // Reserved for future dispatch logic
    pub const fn has_avx2(self) -> bool {
        matches!(self, Self::Avx512 | Self::Avx2Fma)
    }

    /// Returns the number of f32 elements per vector register
    #[inline]
    #[allow(dead_code)] // Reserved for future tiling logic
    pub const fn f32_lanes(self) -> usize {
        match self {
            Self::Avx512 => 16,
            Self::Avx2Fma => 8,
            Self::Scalar => 1,
        }
    }

    /// Returns the number of f64 elements per vector register
    #[inline]
    #[allow(dead_code)] // Reserved for future tiling logic
    pub const fn f64_lanes(self) -> usize {
        match self {
            Self::Avx512 => 8,
            Self::Avx2Fma => 4,
            Self::Scalar => 1,
        }
    }
}

/// Cached SIMD level detection
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Detect the best available SIMD level for the current CPU
///
/// This function is cached - the first call performs detection,
/// subsequent calls return the cached result with ~1ns overhead.
#[inline]
pub fn detect_simd() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(detect_simd_uncached)
}

/// Perform actual CPU feature detection (called once)
#[cold]
fn detect_simd_uncached() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("fma")
        {
            return SimdLevel::Avx512;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdLevel::Avx2Fma;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON is baseline for AArch64, but we don't have NEON kernels yet
        let _ = std::arch::is_aarch64_feature_detected!("neon");
    }

    SimdLevel::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection_is_cached() {
        let level1 = detect_simd();
        let level2 = detect_simd();
        assert_eq!(level1, level2);
    }

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2Fma);
        assert!(SimdLevel::Avx2Fma > SimdLevel::Scalar);
    }

    #[test]
    fn test_simd_level_capabilities() {
        assert!(SimdLevel::Avx512.has_avx512());
        assert!(SimdLevel::Avx512.has_avx2());
        assert!(!SimdLevel::Avx2Fma.has_avx512());
        assert!(SimdLevel::Avx2Fma.has_avx2());
        assert!(!SimdLevel::Scalar.has_avx512());
        assert!(!SimdLevel::Scalar.has_avx2());
    }

    #[test]
    fn test_lane_counts() {
        assert_eq!(SimdLevel::Avx512.f32_lanes(), 16);
        assert_eq!(SimdLevel::Avx2Fma.f32_lanes(), 8);
        assert_eq!(SimdLevel::Scalar.f32_lanes(), 1);
        assert_eq!(SimdLevel::Avx512.f64_lanes(), 8);
        assert_eq!(SimdLevel::Avx2Fma.f64_lanes(), 4);
        assert_eq!(SimdLevel::Scalar.f64_lanes(), 1);
    }
}
