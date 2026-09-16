//! AVX2 i32 matmul with a 64-bit accumulator, guarded by a magnitude prescan.
//!
//! See [`avx2`] for the guard and the kernel.

mod avx2;

pub use avx2::{matmul_i32_avx2, matmul_i32_fits_i64};
