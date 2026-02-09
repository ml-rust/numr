//! FFT algorithm contracts for backend consistency
//!
//! This module defines traits that ensure all backends implement the same
//! Fourier Transform algorithms. This guarantees numerical parity across
//! CPU, CUDA, WebGPU, and other backends.
//!
//! # Algorithm: Stockham Autosort FFT
//!
//! The Stockham algorithm is chosen over Cooley-Tukey because:
//!
//! 1. **No bit-reversal permutation** - Cooley-Tukey's main bottleneck avoided
//! 2. **Sequential memory access** - Natural access patterns for cache efficiency
//! 3. **Double-buffering** - GPU-friendly, no bank conflicts
//! 4. **Identical structure** - Same algorithm across CPU/CUDA/WGPU
//!
//! # Supported Operations
//!
//! - `fft` - Complex-to-complex FFT (forward/inverse)
//! - `rfft` - Real-to-complex FFT (exploits Hermitian symmetry)
//! - `irfft` - Complex-to-real inverse FFT
//!
//! # Algorithm Reference
//!
//! ```text
//! Stockham FFT (radix-2):
//!
//! Input: x[N] complex, where N = 2^m
//! Output: X[N] complex
//!
//! For each stage s = 0..m:
//!     half_m = 2^s
//!     m = 2^(s+1)
//!     For each group g = 0..(N/m):
//!         For each butterfly b = 0..half_m:
//!             twiddle = exp(-2πi * b / m)  // or +2πi for inverse
//!             even = src[g * half_m + b]
//!             odd = src[N/2 + g * half_m + b] * twiddle
//!             dst[g * m + b] = even + odd
//!             dst[g * m + b + half_m] = even - odd
//!     swap(src, dst)  // Double buffering
//!
//! For inverse: multiply final result by 1/N
//! ```

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// FFT Configuration
// ============================================================================

/// Direction of FFT computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward FFT: time domain → frequency domain
    /// Uses twiddle factor: e^(-2πi * k / N)
    Forward,
    /// Inverse FFT: frequency domain → time domain
    /// Uses twiddle factor: e^(+2πi * k / N)
    Inverse,
}

/// Normalization mode for FFT
///
/// # Default: `Backward`
///
/// The default normalization mode is `Backward`, which matches NumPy's `numpy.fft`
/// behavior. This means:
/// - Forward FFT: no normalization (factor = 1)
/// - Inverse FFT: divide by N (factor = 1/N)
///
/// This ensures that `ifft(fft(x)) == x` (roundtrip identity).
///
/// # Comparison Table
///
/// | Mode     | Forward Factor | Inverse Factor | Roundtrip         |
/// |----------|---------------|----------------|-------------------|
/// | None     | 1             | 1              | ifft(fft(x)) = N*x |
/// | Backward | 1             | 1/N            | ifft(fft(x)) = x  |
/// | Ortho    | 1/√N          | 1/√N           | ifft(fft(x)) = x  |
/// | Forward  | 1/N           | 1              | ifft(fft(x)) = x  |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FftNormalization {
    /// No normalization (forward: 1, inverse: 1)
    ///
    /// **Warning**: Roundtrip produces `N * x`. Use when you need unnormalized
    /// raw FFT output, such as for convolution or custom post-processing.
    None,
    /// Normalize inverse by 1/N (forward: 1, inverse: 1/N)
    ///
    /// **Default mode**. Matches `numpy.fft` behavior.
    /// Roundtrip: `ifft(fft(x)) == x`
    #[default]
    Backward,
    /// Normalize both by 1/sqrt(N) (forward: 1/√N, inverse: 1/√N)
    ///
    /// Makes FFT and IFFT symmetric. Preserves Parseval's theorem:
    /// `sum(|x|²) == sum(|X|²)` (energy conservation).
    /// Useful for signal processing where symmetry matters.
    Ortho,
    /// Normalize forward by 1/N (forward: 1/N, inverse: 1)
    ///
    /// Opposite of `Backward`. Roundtrip: `ifft(fft(x)) == x`
    Forward,
}

impl FftNormalization {
    /// Get the normalization factor for a given direction and size
    #[inline]
    pub fn factor(self, direction: FftDirection, n: usize) -> f64 {
        let n_f = n as f64;
        match (self, direction) {
            (Self::None, _) => 1.0,
            (Self::Backward, FftDirection::Forward) => 1.0,
            (Self::Backward, FftDirection::Inverse) => 1.0 / n_f,
            (Self::Ortho, _) => 1.0 / n_f.sqrt(),
            (Self::Forward, FftDirection::Forward) => 1.0 / n_f,
            (Self::Forward, FftDirection::Inverse) => 1.0,
        }
    }
}

// ============================================================================
// FFT Algorithm Trait
// ============================================================================

/// Algorithmic contract for FFT operations
///
/// All backends implementing FFT MUST implement this trait using the
/// EXACT SAME ALGORITHMS to ensure numerical parity.
///
/// # Algorithm: Stockham Autosort FFT
///
/// ```text
/// For each stage s = 0..log2(N):
///     half_m = 2^s
///     m = 2^(s+1)
///     For each group g = 0..(N/m):
///         For each butterfly b = 0..half_m:
///             twiddle = exp(sign * 2πi * b / m)
///             even = src[g * half_m + b]
///             odd = src[N/2 + g * half_m + b] * twiddle
///             dst[g * m + b] = even + odd
///             dst[g * m + b + half_m] = even - odd
///     swap(src, dst)
/// ```
///
/// # Implementation Requirements
///
/// Backends may differ in:
/// - Parallelization strategy (threads, SIMD, GPU blocks)
/// - Memory access patterns (vectorization)
/// - Small FFT optimizations (shared memory, unrolling)
///
/// Backends MUST match in:
/// - Mathematical formula (Stockham radix-2)
/// - Twiddle factor calculation (same precision)
/// - Normalization handling
/// - Input validation (power-of-2, dtype support)
///
/// # Backend-Specific DType Support
///
/// | Backend | Complex Types | Real Input |
/// |---------|---------------|------------|
/// | CPU     | C64, C128     | F32, F64   |
/// | CUDA    | C64, C128     | F32, F64   |
/// | WebGPU  | C64 only      | F32 only   |
pub trait FftAlgorithms<R: Runtime> {
    /// 1D FFT on complex input using Stockham autosort algorithm
    ///
    /// # Arguments
    ///
    /// * `input` - Complex tensor of shape [..., N] where N is power of 2
    /// * `direction` - Forward or Inverse FFT
    /// * `norm` - Normalization mode
    ///
    /// # Returns
    ///
    /// Complex tensor of same shape as input
    ///
    /// # Errors
    ///
    /// - `InvalidArgument` if N is not a power of 2
    /// - `UnsupportedDType` if input is not Complex64 or Complex128
    fn fft(
        &self,
        input: &Tensor<R>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<R>> {
        let _ = (input, direction, norm);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::fft",
        })
    }

    /// 1D FFT along a specific dimension
    ///
    /// # Arguments
    ///
    /// * `input` - Complex tensor
    /// * `dim` - Dimension along which to compute FFT
    /// * `direction` - Forward or Inverse FFT
    /// * `norm` - Normalization mode
    ///
    /// # Returns
    ///
    /// Complex tensor of same shape as input
    fn fft_dim(
        &self,
        input: &Tensor<R>,
        dim: isize,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<R>> {
        let _ = (input, dim, direction, norm);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::fft_dim",
        })
    }

    /// Real FFT: Real input → Complex output
    ///
    /// Exploits Hermitian symmetry: for real input, `X[k] = conj(X[N-k])`,
    /// so we only need to store the first `N/2 + 1` complex values.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Pack N real values into N/2 complex values:
    ///    z[k] = x[2k] + i*x[2k+1]
    /// 2. Compute FFT of z (size N/2)
    /// 3. Unpack to get first N/2+1 values of full FFT
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input` - Real tensor of shape [..., N] where N is power of 2
    /// * `norm` - Normalization mode
    ///
    /// # Returns
    ///
    /// Complex tensor of shape [..., N/2 + 1]
    ///
    /// # Errors
    ///
    /// - `InvalidArgument` if N is not a power of 2
    /// - `UnsupportedDType` if input is not F32 or F64
    fn rfft(&self, input: &Tensor<R>, norm: FftNormalization) -> Result<Tensor<R>> {
        let _ = (input, norm);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::rfft",
        })
    }

    /// Inverse real FFT: Complex input → Real output
    ///
    /// Takes Hermitian-symmetric complex input (from rfft) and produces real output.
    ///
    /// # Arguments
    ///
    /// * `input` - Complex tensor of shape [..., N/2 + 1]
    /// * `n` - Output size (must be even). If None, uses 2*(input_size - 1)
    /// * `norm` - Normalization mode
    ///
    /// # Returns
    ///
    /// Real tensor of shape [..., n]
    ///
    /// # Errors
    ///
    /// - `InvalidArgument` if n is not a power of 2 or doesn't match input
    /// - `UnsupportedDType` if input is not Complex64 or Complex128
    fn irfft(
        &self,
        input: &Tensor<R>,
        n: Option<usize>,
        norm: FftNormalization,
    ) -> Result<Tensor<R>> {
        let _ = (input, n, norm);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::irfft",
        })
    }

    /// 2D FFT on complex input
    ///
    /// Computes FFT along the last two dimensions.
    ///
    /// # Arguments
    ///
    /// * `input` - Complex tensor of shape `[..., M, N]` where `M`, `N` are powers of 2
    /// * `direction` - Forward or Inverse FFT
    /// * `norm` - Normalization mode
    ///
    /// # Returns
    ///
    /// Complex tensor of same shape as input
    fn fft2(
        &self,
        input: &Tensor<R>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<R>> {
        let _ = (input, direction, norm);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::fft2",
        })
    }

    /// 2D Real FFT
    ///
    /// Computes real FFT along the last two dimensions.
    ///
    /// # Arguments
    ///
    /// * `input` - Real tensor of shape `[..., M, N]`
    /// * `norm` - Normalization mode
    ///
    /// # Returns
    ///
    /// Complex tensor of shape `[..., M, N/2 + 1]`
    fn rfft2(&self, input: &Tensor<R>, norm: FftNormalization) -> Result<Tensor<R>> {
        let _ = (input, norm);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::rfft2",
        })
    }

    /// Inverse 2D Real FFT
    ///
    /// # Arguments
    ///
    /// * `input` - Complex tensor of shape `[..., M, N/2 + 1]`
    /// * `s` - Output shape `(M, N)`. If None, uses `(M, 2*(N-1))`
    /// * `norm` - Normalization mode
    ///
    /// # Returns
    ///
    /// Real tensor of shape `[..., M, N]`
    fn irfft2(
        &self,
        input: &Tensor<R>,
        s: Option<(usize, usize)>,
        norm: FftNormalization,
    ) -> Result<Tensor<R>> {
        let _ = (input, s, norm);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::irfft2",
        })
    }

    /// Frequency shift: shift zero-frequency component to center
    ///
    /// For a tensor of shape `[..., N]`, swaps the halves:
    /// `[0..N/2] <-> [N/2..N]`
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor (typically FFT output)
    ///
    /// # Returns
    ///
    /// Tensor with shifted frequencies
    fn fftshift(&self, input: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = input;
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::fftshift",
        })
    }

    /// Inverse frequency shift: undo fftshift
    fn ifftshift(&self, input: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = input;
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::ifftshift",
        })
    }

    /// Generate FFT sample frequencies
    ///
    /// Returns the sample frequencies for an N-point FFT.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of FFT points
    /// * `d` - Sample spacing (default 1.0)
    /// * `dtype` - Output dtype (F32 or F64)
    /// * `device` - Target device
    ///
    /// # Returns
    ///
    /// 1D tensor of shape `[N]` with frequencies:
    /// [0, 1, 2, ..., N/2-1, -N/2, ..., -1] / (d*N)
    fn fftfreq(&self, n: usize, d: f64, dtype: DType, device: &R::Device) -> Result<Tensor<R>> {
        let _ = (n, d, dtype, device);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::fftfreq",
        })
    }

    /// Generate non-negative FFT sample frequencies for rfft
    ///
    /// # Arguments
    ///
    /// * `n` - Number of FFT points (size of real input)
    /// * `d` - Sample spacing (default 1.0)
    /// * `dtype` - Output dtype (F32 or F64)
    /// * `device` - Target device
    ///
    /// # Returns
    ///
    /// 1D tensor of shape [N/2 + 1] with frequencies:
    /// [0, 1, 2, ..., N/2] / (d*N)
    fn rfftfreq(&self, n: usize, d: f64, dtype: DType, device: &R::Device) -> Result<Tensor<R>> {
        let _ = (n, d, dtype, device);
        Err(Error::NotImplemented {
            feature: "FftAlgorithms::rfftfreq",
        })
    }
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Check if a number is a power of 2
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Validate FFT size is power of 2
pub fn validate_fft_size(n: usize, op: &'static str) -> Result<()> {
    if !is_power_of_two(n) {
        let next = n.next_power_of_two();
        // Previous power of 2: only suggest if it's >= 2 and smaller than n
        let prev = next / 2;
        let suggestion = if prev >= 2 && prev < n {
            // Both options are valid: truncate to prev or pad to next
            format!(
                "{} requires power-of-2 size, got {}. \
                 Consider truncating to {} or padding to {}.",
                op, n, prev, next
            )
        } else {
            // Only next is valid (e.g., n=0 or small non-power-of-2)
            format!(
                "{} requires power-of-2 size, got {}. \
                 Consider padding to {}.",
                op, n, next
            )
        };
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: suggestion,
        });
    }
    Ok(())
}

/// Validate dtype is complex for FFT
pub fn validate_fft_complex_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if !dtype.is_complex() {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validate dtype is real float for rfft
pub fn validate_rfft_real_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Get the complex output dtype for a real input dtype
pub fn complex_dtype_for_real(real_dtype: DType) -> Result<DType> {
    match real_dtype {
        DType::F32 => Ok(DType::Complex64),
        DType::F64 => Ok(DType::Complex128),
        _ => Err(Error::UnsupportedDType {
            dtype: real_dtype,
            op: "rfft",
        }),
    }
}

/// Get the real output dtype for a complex input dtype
pub fn real_dtype_for_complex(complex_dtype: DType) -> Result<DType> {
    match complex_dtype {
        DType::Complex64 => Ok(DType::F32),
        DType::Complex128 => Ok(DType::F64),
        _ => Err(Error::UnsupportedDType {
            dtype: complex_dtype,
            op: "irfft",
        }),
    }
}

// ============================================================================
// Twiddle Factor Generation
// ============================================================================

use std::f64::consts::PI;

/// Generate twiddle factors for FFT stage
///
/// For radix-2 Stockham FFT, generates W_N^k = exp(-2πi * k / N)
/// for k = 0..N/2.
pub fn generate_twiddles_c64(n: usize, inverse: bool) -> Vec<crate::dtype::Complex64> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let n_f = n as f64;

    (0..n / 2)
        .map(|k| {
            let theta = sign * 2.0 * PI * (k as f64) / n_f;
            crate::dtype::Complex64::new(theta.cos() as f32, theta.sin() as f32)
        })
        .collect()
}

/// Generate twiddle factors for FFT stage (f64 precision)
pub fn generate_twiddles_c128(n: usize, inverse: bool) -> Vec<crate::dtype::Complex128> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let n_f = n as f64;

    (0..n / 2)
        .map(|k| {
            let theta = sign * 2.0 * PI * (k as f64) / n_f;
            crate::dtype::Complex128::new(theta.cos(), theta.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(7));
    }

    #[test]
    fn test_validate_fft_size() {
        assert!(validate_fft_size(4, "fft").is_ok());
        assert!(validate_fft_size(1024, "fft").is_ok());
        assert!(validate_fft_size(7, "fft").is_err());
    }

    #[test]
    fn test_normalization_factor() {
        let n = 8;

        // Backward (default): forward=1, inverse=1/N
        assert_eq!(
            FftNormalization::Backward.factor(FftDirection::Forward, n),
            1.0
        );
        assert_eq!(
            FftNormalization::Backward.factor(FftDirection::Inverse, n),
            0.125
        );

        // Ortho: both = 1/sqrt(N)
        let sqrt_inv = 1.0 / (n as f64).sqrt();
        assert!(
            (FftNormalization::Ortho.factor(FftDirection::Forward, n) - sqrt_inv).abs() < 1e-10
        );
        assert!(
            (FftNormalization::Ortho.factor(FftDirection::Inverse, n) - sqrt_inv).abs() < 1e-10
        );
    }

    #[test]
    fn test_twiddle_generation() {
        let twiddles = generate_twiddles_c64(8, false);
        assert_eq!(twiddles.len(), 4);

        // W_8^0 = 1 + 0i
        assert!((twiddles[0].re - 1.0).abs() < 1e-6);
        assert!(twiddles[0].im.abs() < 1e-6);

        // W_8^2 = exp(-2πi * 2/8) = exp(-πi/2) = -i
        assert!(twiddles[2].re.abs() < 1e-6);
        assert!((twiddles[2].im - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_complex_dtype_conversion() {
        assert_eq!(
            complex_dtype_for_real(DType::F32).unwrap(),
            DType::Complex64
        );
        assert_eq!(
            complex_dtype_for_real(DType::F64).unwrap(),
            DType::Complex128
        );
        assert!(complex_dtype_for_real(DType::I32).is_err());
    }
}

// ============================================================================
// Compile-Time Backend Verification
//
// These functions exist solely to verify at compile time that each enabled
// backend implements FftAlgorithms. If a backend is enabled but doesn't
// implement the trait, compilation fails with a clear error message.
// ============================================================================

/// Verify CPU backend implements FftAlgorithms (compile-time check)
#[allow(dead_code)]
fn _verify_cpu_fft_impl() {
    fn assert_fft_impl<T: FftAlgorithms<crate::runtime::cpu::CpuRuntime>>() {}
    assert_fft_impl::<crate::runtime::cpu::CpuClient>();
}

/// Verify CUDA backend implements FftAlgorithms (compile-time check)
#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn _verify_cuda_fft_impl() {
    fn assert_fft_impl<T: FftAlgorithms<crate::runtime::cuda::CudaRuntime>>() {}
    assert_fft_impl::<crate::runtime::cuda::CudaClient>();
}

/// Verify WebGPU backend implements FftAlgorithms (compile-time check)
#[cfg(feature = "wgpu")]
#[allow(dead_code)]
fn _verify_wgpu_fft_impl() {
    fn assert_fft_impl<T: FftAlgorithms<crate::runtime::wgpu::WgpuRuntime>>() {}
    assert_fft_impl::<crate::runtime::wgpu::WgpuClient>();
}
