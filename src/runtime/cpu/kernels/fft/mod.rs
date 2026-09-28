//! FFT kernels using Stockham autosort algorithm
//!
//! This module provides CPU implementations of FFT operations.
//! The Stockham algorithm is used for its:
//! - No bit-reversal permutation (Cooley-Tukey's main bottleneck)
//! - Sequential memory access patterns
//! - Natural double-buffering
//!
//! # Algorithm: Stockham Radix-2 FFT
//!
//! ```text
//! For each stage s = 0..log2(N):
//!     half_m = 2^s
//!     m = 2^(s+1)
//!     For each group g = 0..(N/m):
//!         For each butterfly b = 0..half_m:
//!             twiddle = exp(sign * 2πi * b / m)
//!             even = src[g * half_m + b]
//!             odd = src[N/2 + g * half_m + b] * twiddle
//!             dst[g * m + b] = even + odd
//!             dst[g * m + b + half_m] = even - odd
//!     swap(src, dst)
//! ```

mod batched;
mod bluestein;
mod dispatch;
#[cfg(test)]
mod real;
mod real_plan;
mod shift;
mod stockham;
#[cfg(test)]
mod test_support;
mod twiddles;

pub use batched::{stockham_fft_batched_c64, stockham_fft_batched_c128};
pub use real_plan::{IrfftPlanC64, IrfftPlanC128, RfftPlanC64, RfftPlanC128};
pub use shift::{fftshift_c64, fftshift_c128, ifftshift_c64, ifftshift_c128};
