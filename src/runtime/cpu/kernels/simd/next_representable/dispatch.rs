//! Dispatch once per integer range using the caller's detected SIMD level.

use super::super::SimdLevel;
use super::scalar;

macro_rules! dispatch {
    ($name:ident, $bits:ty) => {
        /// Step raw IEEE encodings using the supplied SIMD level.
        ///
        /// # Safety
        /// Pointers must address `len` readable input and writable output elements.
        /// Output must not overlap unread input elements. The CPU must support `level`.
        pub unsafe fn $name(
            src: *const $bits,
            dst: *mut $bits,
            len: usize,
            up: bool,
            level: SimdLevel,
        ) {
            unsafe {
                match level {
                    #[cfg(target_arch = "x86_64")]
                    SimdLevel::Avx512 => super::avx512::$name(src, dst, len, up),
                    #[cfg(target_arch = "x86_64")]
                    SimdLevel::Avx2Fma => super::avx2::$name(src, dst, len, up),
                    #[cfg(target_arch = "aarch64")]
                    SimdLevel::Neon | SimdLevel::NeonFp16 => super::neon::$name(src, dst, len, up),
                    _ => scalar::$name(src, dst, len, up),
                }
            }
        }
    };
}

dispatch!(next_f32, u32);
dispatch!(next_f64, u64);

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
