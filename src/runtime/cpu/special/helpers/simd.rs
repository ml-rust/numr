//! SIMD-optimized special function wrappers
//!
//! Provides tensor-level wrappers that dispatch to SIMD kernels when available,
//! falling back to scalar implementations for non-contiguous tensors or
//! unsupported architectures.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::cpu::{CpuDevice, CpuRuntime};
use crate::tensor::Tensor;

use super::scalar::apply_unary;

#[cfg(target_arch = "x86_64")]
use crate::runtime::cpu::kernels::simd::special as simd_special;

#[cfg(target_arch = "aarch64")]
use crate::runtime::cpu::kernels::simd::special as simd_special;

/// Macro to generate SIMD-optimized special function wrappers.
///
/// This eliminates duplication across erf, erfc, bessel_*, gamma, etc.
/// Each generated function:
/// 1. Checks if tensor is contiguous (required for SIMD)
/// 2. Dispatches to architecture-specific SIMD kernel if available
/// 3. Falls back to scalar implementation otherwise
macro_rules! impl_simd_special_fn {
    ($fn_name:ident, $simd_f32:ident, $simd_f64:ident, $scalar_fn:path) => {
        pub fn $fn_name(x: &Tensor<CpuRuntime>, device: &CpuDevice) -> Result<Tensor<CpuRuntime>> {
            // SIMD requires contiguous memory layout
            if !x.is_contiguous() {
                return apply_unary(x, device, $scalar_fn);
            }

            match x.dtype() {
                DType::F32 => {
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    {
                        let len = x.numel();
                        let mut result = vec![0.0f32; len];
                        let input_ptr = x.storage().ptr() as *const f32;
                        unsafe {
                            simd_special::$simd_f32(input_ptr, result.as_mut_ptr(), len);
                        }
                        return Ok(Tensor::from_slice(&result, x.shape(), device));
                    }

                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    apply_unary(x, device, $scalar_fn)
                }
                DType::F64 => {
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    {
                        let len = x.numel();
                        let mut result = vec![0.0f64; len];
                        let input_ptr = x.storage().ptr() as *const f64;
                        unsafe {
                            simd_special::$simd_f64(input_ptr, result.as_mut_ptr(), len);
                        }
                        return Ok(Tensor::from_slice(&result, x.shape(), device));
                    }

                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    apply_unary(x, device, $scalar_fn)
                }
                // F16/BF16/FP8: Convert to F32, compute, convert back
                DType::F16 | DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2 => {
                    apply_unary(x, device, $scalar_fn)
                }
                _ => unreachable!("dtype validated by caller"),
            }
        }
    };
}

// Generate all SIMD-optimized special function wrappers
impl_simd_special_fn!(
    apply_erf,
    erf_f32,
    erf_f64,
    crate::algorithm::special::scalar::erf_scalar
);

impl_simd_special_fn!(
    apply_erfc,
    erfc_f32,
    erfc_f64,
    crate::algorithm::special::scalar::erfc_scalar
);

impl_simd_special_fn!(
    apply_bessel_j0,
    bessel_j0_f32,
    bessel_j0_f64,
    crate::algorithm::special::scalar::bessel_j0_scalar
);

impl_simd_special_fn!(
    apply_bessel_j1,
    bessel_j1_f32,
    bessel_j1_f64,
    crate::algorithm::special::scalar::bessel_j1_scalar
);

impl_simd_special_fn!(
    apply_bessel_i0,
    bessel_i0_f32,
    bessel_i0_f64,
    crate::algorithm::special::scalar::bessel_i0_scalar
);

impl_simd_special_fn!(
    apply_bessel_i1,
    bessel_i1_f32,
    bessel_i1_f64,
    crate::algorithm::special::scalar::bessel_i1_scalar
);

impl_simd_special_fn!(
    apply_gamma,
    gamma_f32,
    gamma_f64,
    crate::algorithm::special::scalar::gamma_scalar
);

impl_simd_special_fn!(
    apply_lgamma,
    lgamma_f32,
    lgamma_f64,
    crate::algorithm::special::scalar::lgamma_scalar
);

impl_simd_special_fn!(
    apply_digamma,
    digamma_f32,
    digamma_f64,
    crate::algorithm::special::scalar::digamma_scalar
);
