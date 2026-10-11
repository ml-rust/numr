//! SIMD-optimized matrix multiplication.
//!
//! See [`dispatch`] for the public API and microkernel dispatch functions.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
// The float kernels below serve only the SIMD architectures. Other targets
// run the arch-neutral loops in `kernels::matmul`, which never call them.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod dispatch;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod gemv_bt;
// i32 matmul: AVX2 with an i64 accumulator when a magnitude prescan proves
// every partial sum fits, otherwise the exact i128 scalar path. An earlier AVX2
// kernel accumulated with `_mm256_add_epi32` and wrapped mid-dot-product; the
// guard is what makes a wider-but-still-finite accumulator safe.
#[cfg(target_arch = "x86_64")]
pub(crate) mod int32;
pub(crate) mod int8;
pub(crate) mod macros;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod packing;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod scalar;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod small;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod small_kernels;
pub(crate) mod tiled_path;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod tiling;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

// Both SIMD architectures: `matmul_kernel` routes F16/BF16 here on each, and
// the module itself is architecture-neutral — it converts to f32 and calls
// `matmul_f32`, which dispatches per architecture on its own.
#[cfg(all(feature = "f16", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) mod half_convert;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub use dispatch::{
    KC, MC, MR, NC, call_microkernel_2x_f32, call_microkernel_2x_f64, call_microkernel_f32,
    call_microkernel_f64, matmul_bias_f32, matmul_bias_f64, matmul_bt_f32, matmul_bt_f64,
    matmul_f32, matmul_f64,
};
pub use tiled_path::matmul_bt_is_tiled;
// Only the rayon column split reads this floor.
#[cfg(feature = "rayon")]
pub use tiled_path::min_tiled_columns;
