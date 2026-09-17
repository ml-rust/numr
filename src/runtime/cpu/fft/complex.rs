//! FftAlgorithms trait dispatch for the CPU runtime.

use super::super::{CpuClient, CpuRuntime};
use super::{real, shift};
use crate::algorithm::fft::{
    FftAlgorithms, FftDirection, FftNormalization, validate_fft_complex_dtype,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl FftAlgorithms<CpuRuntime> for CpuClient {
    fn fft(
        &self,
        input: &Tensor<CpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "FFT requires at least 1D input".to_string(),
            });
        }
        self.fft_dim(input, -1, direction, norm)
    }

    fn fft_dim(
        &self,
        input: &Tensor<CpuRuntime>,
        dim: isize,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();
        validate_fft_complex_dtype(dtype, "fft")?;

        let ndim = input.ndim();
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        let n = input.shape()[dim];
        // The CPU backend handles any size: power-of-two via Stockham, everything
        // else via Bluestein. Only an empty transform axis is invalid.
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: format!("fft requires size >= 1 along dim {}, got 0", dim),
            });
        }

        if dim == ndim - 1 && input.is_contiguous() {
            return self.fft_last_dim_contiguous(input, direction, norm);
        }

        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(dim, ndim - 1);

        let transposed = input.permute(&perm)?;
        let transposed_contig = transposed.contiguous()?;

        let result = self.fft_last_dim_contiguous(&transposed_contig, direction, norm)?;
        result.permute(&perm)
    }

    fn rfft(
        &self,
        input: &Tensor<CpuRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        real::rfft_impl(self, input, norm)
    }

    fn irfft(
        &self,
        input: &Tensor<CpuRuntime>,
        n: Option<usize>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        real::irfft_impl(self, input, n, norm)
    }

    fn fft2(
        &self,
        input: &Tensor<CpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "fft2 requires at least 2D input".to_string(),
            });
        }

        let result = self.fft_dim(input, -1, direction, norm)?;
        self.fft_dim(&result, -2, direction, norm)
    }

    fn rfft2(
        &self,
        input: &Tensor<CpuRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "rfft2 requires at least 2D input".to_string(),
            });
        }

        let result = self.rfft(input, norm)?;
        self.fft_dim(&result, -2, FftDirection::Forward, FftNormalization::None)
    }

    fn irfft2(
        &self,
        input: &Tensor<CpuRuntime>,
        s: Option<(usize, usize)>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let ndim = input.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "irfft2 requires at least 2D input".to_string(),
            });
        }

        let input_shape = input.shape();
        let (m, n) = if let Some((m, n)) = s {
            (m, n)
        } else {
            (input_shape[ndim - 2], 2 * (input_shape[ndim - 1] - 1))
        };

        let result = self.fft_dim(input, -2, FftDirection::Inverse, FftNormalization::None)?;

        if result.shape()[ndim - 2] != m {
            return Err(Error::InvalidArgument {
                arg: "s",
                reason: format!(
                    "M dimension mismatch: expected {}, got {}",
                    m,
                    result.shape()[ndim - 2]
                ),
            });
        }

        self.irfft(&result, Some(n), norm)
    }

    fn fftshift(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        shift::fftshift_impl(self, input)
    }

    fn ifftshift(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        shift::ifftshift_impl(self, input)
    }

    fn fftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        shift::fftfreq_impl(n, d, dtype, device)
    }

    fn rfftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        shift::rfftfreq_impl(n, d, dtype, device)
    }
}
