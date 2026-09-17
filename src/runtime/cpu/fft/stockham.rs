//! Last-dimension contiguous FFT dispatch via the Stockham autosort kernels.

use super::super::{CpuClient, CpuRuntime, kernels};
use crate::algorithm::fft::{FftDirection, FftNormalization};
use crate::dtype::{Complex64, Complex128, DType};
use crate::error::Result;
use crate::tensor::Tensor;

impl CpuClient {
    /// FFT along last dimension for contiguous tensor
    pub(super) fn fft_last_dim_contiguous(
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

        let output = Tensor::<CpuRuntime>::empty(input.shape(), dtype, &self.device)?;

        // Unclamped: rank-1 already products to 1. Clamping a zero batch dim to 1
        // would build a `from_raw_parts` slice longer than the empty allocation.
        let batch_size: usize = input.shape()[..ndim - 1].iter().product();
        let min_len = self.chunk_size_hint();

        let input_ptr = input.ptr();
        let output_ptr = output.ptr();

        match dtype {
            DType::Complex64 => {
                let input_slice: &[Complex64] = unsafe {
                    std::slice::from_raw_parts(input_ptr as *const Complex64, batch_size * n)
                };
                let output_slice: &mut [Complex64] = unsafe {
                    std::slice::from_raw_parts_mut(output_ptr as *mut Complex64, batch_size * n)
                };

                if batch_size > 1 {
                    self.install_parallelism(|| unsafe {
                        kernels::stockham_fft_batched_c64(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor as f32,
                            min_len,
                        );
                    });
                } else {
                    unsafe {
                        kernels::stockham_fft_batched_c64(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor as f32,
                            min_len,
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

                if batch_size > 1 {
                    self.install_parallelism(|| unsafe {
                        kernels::stockham_fft_batched_c128(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor,
                            min_len,
                        );
                    });
                } else {
                    unsafe {
                        kernels::stockham_fft_batched_c128(
                            input_slice,
                            output_slice,
                            n,
                            batch_size,
                            inverse,
                            normalize_factor,
                            min_len,
                        );
                    }
                }
            }
            _ => unreachable!(),
        }

        Ok(output)
    }
}
