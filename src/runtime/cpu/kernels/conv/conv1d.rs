use super::super::wide_acc::FloatAcc;
use crate::dtype::{DType, Element};
use crate::ops::conv_common::Conv1dParams;

/// 1D convolution kernel with groups support.
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Arrays have sufficient size for the operation
/// - params contains valid dimensions
pub unsafe fn conv1d_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    bias: Option<*const T>,
    output: *mut T,
    params: Conv1dParams,
) {
    // Dispatch to SIMD for f16/bf16 on x86-64 and aarch64
    #[cfg(all(feature = "f16", any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        use super::super::simd::conv as simd_conv;

        match T::DTYPE {
            DType::F16 => {
                simd_conv::conv1d_f16(
                    input as *const half::f16,
                    weight as *const half::f16,
                    bias.map(|b| b as *const half::f16),
                    output as *mut half::f16,
                    params,
                );
                return;
            }
            DType::BF16 => {
                simd_conv::conv1d_bf16(
                    input as *const half::bf16,
                    weight as *const half::bf16,
                    bias.map(|b| b as *const half::bf16),
                    output as *mut half::bf16,
                    params,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    if T::DTYPE == DType::F64 {
        conv1d_kernel_acc::<T, f64>(input, weight, bias, output, params);
    } else {
        conv1d_kernel_acc::<T, f32>(input, weight, bias, output, params);
    }
}

/// `conv1d_kernel` with the accumulator type fixed by the caller.
///
/// # Safety
///
/// Same as [`conv1d_kernel`].
unsafe fn conv1d_kernel_acc<T: Element, A: FloatAcc>(
    input: *const T,
    weight: *const T,
    bias: Option<*const T>,
    output: *mut T,
    params: Conv1dParams,
) {
    let Conv1dParams {
        batch,
        c_in,
        length,
        c_out,
        kernel_size,
        stride,
        dilation,
        groups,
        pad_left,
        pad_right: _,
        output_length,
    } = params;

    let c_in_per_group = c_in / groups;
    let c_out_per_group = c_out / groups;

    // Input layout: (batch, c_in, length)
    // Weight layout: (c_out, c_in_per_group, kernel_size)
    // Output layout: (batch, c_out, output_length)

    for b in 0..batch {
        for g in 0..groups {
            let c_out_start = g * c_out_per_group;
            let c_in_start = g * c_in_per_group;

            for oc in 0..c_out_per_group {
                let c_out_idx = c_out_start + oc;

                for ox in 0..output_length {
                    let mut sum = A::ZERO;

                    for ic in 0..c_in_per_group {
                        let c_in_idx = c_in_start + ic;

                        for kx in 0..kernel_size {
                            let ix_signed = (ox * stride) as isize + (kx * dilation) as isize
                                - pad_left as isize;

                            if ix_signed >= 0 && (ix_signed as usize) < length {
                                let ix = ix_signed as usize;

                                // Input index: b * c_in * length + c_in_idx * length + ix
                                let input_idx = b * c_in * length + c_in_idx * length + ix;

                                // Weight index: c_out_idx * c_in_per_group * kernel_size + ic * kernel_size + kx
                                let weight_idx = c_out_idx * c_in_per_group * kernel_size
                                    + ic * kernel_size
                                    + kx;

                                let in_val = A::from_elem(*input.add(input_idx));
                                let w_val = A::from_elem(*weight.add(weight_idx));
                                sum = sum.acc_add(in_val.acc_mul(w_val));
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(bias_ptr) = bias {
                        sum = sum.acc_add(A::from_elem(*bias_ptr.add(c_out_idx)));
                    }

                    // Output index: b * c_out * output_length + c_out_idx * output_length + ox
                    let output_idx = b * c_out * output_length + c_out_idx * output_length + ox;
                    *output.add(output_idx) = sum.to_elem::<T>();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::PaddingMode;
    use crate::ops::conv_common::validate_conv1d;

    #[test]
    fn test_conv1d_no_padding() {
        // Input: (1, 1, 5) = [1, 2, 3, 4, 5]
        // Weight: (1, 1, 3) = [1, 1, 1]
        // Output: (1, 1, 3) = [6, 9, 12] (sum of sliding window)
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 3];

        let params = validate_conv1d(
            &[1, 1, 5],
            &[1, 1, 3],
            None,
            1,
            PaddingMode::Valid,
            1,
            1,
            crate::dtype::DType::F32,
            crate::dtype::DType::F32,
            None,
        )
        .unwrap();

        unsafe {
            conv1d_kernel(
                input.as_ptr(),
                weight.as_ptr(),
                None,
                output.as_mut_ptr(),
                params,
            );
        }

        assert!((output[0] - 6.0).abs() < 1e-5);
        assert!((output[1] - 9.0).abs() < 1e-5);
        assert!((output[2] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0f32, 1.0, 1.0];
        let bias = [10.0f32];
        let mut output = [0.0f32; 3];

        let params = validate_conv1d(
            &[1, 1, 5],
            &[1, 1, 3],
            Some(&[1]),
            1,
            PaddingMode::Valid,
            1,
            1,
            crate::dtype::DType::F32,
            crate::dtype::DType::F32,
            Some(crate::dtype::DType::F32),
        )
        .unwrap();

        unsafe {
            conv1d_kernel(
                input.as_ptr(),
                weight.as_ptr(),
                Some(bias.as_ptr()),
                output.as_mut_ptr(),
                params,
            );
        }

        assert!((output[0] - 16.0).abs() < 1e-5); // 6 + 10
        assert!((output[1] - 19.0).abs() < 1e-5); // 9 + 10
        assert!((output[2] - 22.0).abs() < 1e-5); // 12 + 10
    }
}
