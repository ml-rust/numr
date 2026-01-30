//! FFT algorithm implementation for CPU runtime
//!
//! This module implements the FftAlgorithms trait for CpuClient using
//! the Stockham autosort algorithm.

use super::{CpuClient, CpuRuntime, kernels};
use crate::algorithm::fft::{
    FftAlgorithms, FftDirection, FftNormalization, complex_dtype_for_real, real_dtype_for_complex,
    validate_fft_complex_dtype, validate_fft_size, validate_rfft_real_dtype,
};
use crate::dtype::{Complex64, Complex128, DType};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl FftAlgorithms<CpuRuntime> for CpuClient {
    fn fft(
        &self,
        input: &Tensor<CpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        // FFT along the last dimension
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
        validate_fft_size(n, "fft")?;

        // If FFT is along last dimension and tensor is contiguous, we can use optimized path
        if dim == ndim - 1 && input.is_contiguous() {
            return self.fft_last_dim_contiguous(input, direction, norm);
        }

        // For non-last dimension or non-contiguous, transpose, compute, transpose back
        // First, move the target dimension to the last position
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(dim, ndim - 1);

        let transposed = input.permute(&perm)?;
        let transposed_contig = transposed.contiguous();

        let result = self.fft_last_dim_contiguous(&transposed_contig, direction, norm)?;

        // Transpose back
        result.permute(&perm)
    }

    fn rfft(
        &self,
        input: &Tensor<CpuRuntime>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();
        validate_rfft_real_dtype(dtype, "rfft")?;

        let ndim = input.ndim();
        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "rfft requires at least 1D input".to_string(),
            });
        }

        let n = input.shape()[ndim - 1];
        validate_fft_size(n, "rfft")?;

        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()
        };

        let output_dtype = complex_dtype_for_real(dtype)?;
        let normalize_factor = norm.factor(FftDirection::Forward, n);

        // Compute output shape: [..., N/2 + 1]
        let mut out_shape = input_contig.shape().to_vec();
        out_shape[ndim - 1] = n / 2 + 1;

        let output = Tensor::<CpuRuntime>::empty(&out_shape, output_dtype, &self.device);

        // Compute batch size (product of all dims except last)
        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        let input_ptr = input_contig.storage().ptr();
        let output_ptr = output.storage().ptr();

        match dtype {
            DType::F32 => {
                let input_slice: &[f32] =
                    unsafe { std::slice::from_raw_parts(input_ptr as *const f32, batch_size * n) };
                let output_slice: &mut [Complex64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        output_ptr as *mut Complex64,
                        batch_size * (n / 2 + 1),
                    )
                };

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * n;
                    let out_start = batch_idx * (n / 2 + 1);

                    unsafe {
                        kernels::rfft_c64(
                            &input_slice[in_start..in_start + n],
                            &mut output_slice[out_start..out_start + n / 2 + 1],
                            normalize_factor as f32,
                        );
                    }
                }
            }
            DType::F64 => {
                let input_slice: &[f64] =
                    unsafe { std::slice::from_raw_parts(input_ptr as *const f64, batch_size * n) };
                let output_slice: &mut [Complex128] = unsafe {
                    std::slice::from_raw_parts_mut(
                        output_ptr as *mut Complex128,
                        batch_size * (n / 2 + 1),
                    )
                };

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * n;
                    let out_start = batch_idx * (n / 2 + 1);

                    unsafe {
                        kernels::rfft_c128(
                            &input_slice[in_start..in_start + n],
                            &mut output_slice[out_start..out_start + n / 2 + 1],
                            normalize_factor,
                        );
                    }
                }
            }
            _ => unreachable!(), // validate_rfft_real_dtype ensures F32 or F64
        }

        Ok(output)
    }

    fn irfft(
        &self,
        input: &Tensor<CpuRuntime>,
        n: Option<usize>,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();
        validate_fft_complex_dtype(dtype, "irfft")?;

        let ndim = input.ndim();
        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "irfft requires at least 1D input".to_string(),
            });
        }

        let input_n = input.shape()[ndim - 1];
        let output_n = n.unwrap_or(2 * (input_n - 1));

        if output_n / 2 + 1 != input_n {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: format!(
                    "For irfft with n={}, input must have size {}, got {}",
                    output_n,
                    output_n / 2 + 1,
                    input_n
                ),
            });
        }

        validate_fft_size(output_n, "irfft")?;

        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()
        };

        let output_dtype = real_dtype_for_complex(dtype)?;
        let normalize_factor = norm.factor(FftDirection::Inverse, output_n);

        // Compute output shape: [..., N]
        let mut out_shape = input_contig.shape().to_vec();
        out_shape[ndim - 1] = output_n;

        let output = Tensor::<CpuRuntime>::empty(&out_shape, output_dtype, &self.device);

        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        let input_ptr = input_contig.storage().ptr();
        let output_ptr = output.storage().ptr();

        match dtype {
            DType::Complex64 => {
                let input_slice: &[Complex64] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * input_n)
                };
                let output_slice: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut f32, batch_size * output_n)
                };

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * input_n;
                    let out_start = batch_idx * output_n;

                    unsafe {
                        kernels::irfft_c64(
                            &input_slice[in_start..in_start + input_n],
                            &mut output_slice[out_start..out_start + output_n],
                            normalize_factor as f32,
                        );
                    }
                }
            }
            DType::Complex128 => {
                let input_slice: &[Complex128] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex128, batch_size * input_n)
                };
                let output_slice: &mut [f64] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut f64, batch_size * output_n)
                };

                for batch_idx in 0..batch_size {
                    let in_start = batch_idx * input_n;
                    let out_start = batch_idx * output_n;

                    unsafe {
                        kernels::irfft_c128(
                            &input_slice[in_start..in_start + input_n],
                            &mut output_slice[out_start..out_start + output_n],
                            normalize_factor,
                        );
                    }
                }
            }
            _ => unreachable!(),
        }

        Ok(output)
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

        // FFT along last two dimensions (dim=-1, then dim=-2)
        // Apply normalization to both dimensions
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

        // rfft along last dimension, then fft along second-to-last
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

        // ifft along second-to-last, then irfft along last
        let result = self.fft_dim(input, -2, FftDirection::Inverse, FftNormalization::None)?;

        // Ensure M dimension matches if specified
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
        let dtype = input.dtype();
        let ndim = input.ndim();

        if ndim == 0 {
            return Ok(input.clone());
        }

        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()
        };

        let n = input_contig.shape()[ndim - 1];
        let output = Tensor::<CpuRuntime>::empty(input_contig.shape(), dtype, &self.device);

        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        let input_ptr = input_contig.storage().ptr();
        let output_ptr = output.storage().ptr();

        match dtype {
            DType::Complex64 => {
                let input_slice: &[Complex64] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * n)
                };
                let output_slice: &mut [Complex64] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex64, batch_size * n)
                };

                for batch_idx in 0..batch_size {
                    let start = batch_idx * n;
                    unsafe {
                        kernels::fftshift_c64(
                            &input_slice[start..start + n],
                            &mut output_slice[start..start + n],
                        );
                    }
                }
            }
            DType::Complex128 => {
                let input_slice: &[Complex128] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex128, batch_size * n)
                };
                let output_slice: &mut [Complex128] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex128, batch_size * n)
                };

                for batch_idx in 0..batch_size {
                    let start = batch_idx * n;
                    unsafe {
                        kernels::fftshift_c128(
                            &input_slice[start..start + n],
                            &mut output_slice[start..start + n],
                        );
                    }
                }
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "fftshift",
                });
            }
        }

        Ok(output)
    }

    fn ifftshift(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();
        let ndim = input.ndim();

        if ndim == 0 {
            return Ok(input.clone());
        }

        let input_contig = if input.is_contiguous() {
            input.clone()
        } else {
            input.contiguous()
        };

        let n = input_contig.shape()[ndim - 1];
        let output = Tensor::<CpuRuntime>::empty(input_contig.shape(), dtype, &self.device);

        let batch_size: usize = input_contig.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        let input_ptr = input_contig.storage().ptr();
        let output_ptr = output.storage().ptr();

        match dtype {
            DType::Complex64 => {
                let input_slice: &[Complex64] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * n)
                };
                let output_slice: &mut [Complex64] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex64, batch_size * n)
                };

                for batch_idx in 0..batch_size {
                    let start = batch_idx * n;
                    unsafe {
                        kernels::ifftshift_c64(
                            &input_slice[start..start + n],
                            &mut output_slice[start..start + n],
                        );
                    }
                }
            }
            DType::Complex128 => {
                let input_slice: &[Complex128] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex128, batch_size * n)
                };
                let output_slice: &mut [Complex128] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex128, batch_size * n)
                };

                for batch_idx in 0..batch_size {
                    let start = batch_idx * n;
                    unsafe {
                        kernels::ifftshift_c128(
                            &input_slice[start..start + n],
                            &mut output_slice[start..start + n],
                        );
                    }
                }
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "ifftshift",
                });
            }
        }

        Ok(output)
    }

    fn fftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: "n must be positive".to_string(),
            });
        }

        let output = Tensor::<CpuRuntime>::empty(&[n], dtype, device);
        let scale = 1.0 / (d * n as f64);
        let output_ptr = output.storage().ptr();

        // Frequencies: [0, 1, ..., N/2-1, -N/2, ..., -1] / (d*N)
        match dtype {
            DType::F32 => {
                let output_slice: &mut [f32] =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, n) };

                #[allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
                for i in 0..n {
                    let freq = if i < (n + 1) / 2 {
                        i as f64
                    } else {
                        (i as isize - n as isize) as f64
                    };
                    output_slice[i] = (freq * scale) as f32;
                }
            }
            DType::F64 => {
                let output_slice: &mut [f64] =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, n) };

                #[allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
                for i in 0..n {
                    let freq = if i < (n + 1) / 2 {
                        i as f64
                    } else {
                        (i as isize - n as isize) as f64
                    };
                    output_slice[i] = freq * scale;
                }
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "fftfreq",
                });
            }
        }

        Ok(output)
    }

    fn rfftfreq(
        &self,
        n: usize,
        d: f64,
        dtype: DType,
        device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: "n must be positive".to_string(),
            });
        }

        let output_len = n / 2 + 1;
        let output = Tensor::<CpuRuntime>::empty(&[output_len], dtype, device);
        let scale = 1.0 / (d * n as f64);
        let output_ptr = output.storage().ptr();

        // Frequencies: [0, 1, ..., N/2] / (d*N)
        match dtype {
            DType::F32 => {
                let output_slice: &mut [f32] =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_len) };

                #[allow(clippy::needless_range_loop)]
                for i in 0..output_len {
                    output_slice[i] = (i as f64 * scale) as f32;
                }
            }
            DType::F64 => {
                let output_slice: &mut [f64] =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_len) };

                #[allow(clippy::needless_range_loop)]
                for i in 0..output_len {
                    output_slice[i] = i as f64 * scale;
                }
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "rfftfreq",
                });
            }
        }

        Ok(output)
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

impl CpuClient {
    /// FFT along last dimension for contiguous tensor
    fn fft_last_dim_contiguous(
        &self,
        input: &Tensor<CpuRuntime>,
        direction: FftDirection,
        norm: FftNormalization,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();
        let ndim = input.ndim();
        let n = input.shape()[ndim - 1];

        let inverse = matches!(direction, FftDirection::Inverse);
        let normalize_factor = norm.factor(direction, n);

        let output = Tensor::<CpuRuntime>::empty(input.shape(), dtype, &self.device);

        let batch_size: usize = input.shape()[..ndim - 1].iter().product();
        let batch_size = batch_size.max(1);

        let input_ptr = input.storage().ptr();
        let output_ptr = output.storage().ptr();

        match dtype {
            DType::Complex64 => {
                let input_slice: &[Complex64] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * n)
                };
                let output_slice: &mut [Complex64] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex64, batch_size * n)
                };

                unsafe {
                    kernels::stockham_fft_batched_c64(
                        input_slice,
                        output_slice,
                        n,
                        batch_size,
                        inverse,
                        normalize_factor as f32,
                    );
                }
            }
            DType::Complex128 => {
                let input_slice: &[Complex128] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex128, batch_size * n)
                };
                let output_slice: &mut [Complex128] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex128, batch_size * n)
                };

                unsafe {
                    kernels::stockham_fft_batched_c128(
                        input_slice,
                        output_slice,
                        n,
                        batch_size,
                        inverse,
                        normalize_factor,
                    );
                }
            }
            _ => unreachable!(), // validate_fft_complex_dtype ensures Complex64 or Complex128
        }

        Ok(output)
    }
}
