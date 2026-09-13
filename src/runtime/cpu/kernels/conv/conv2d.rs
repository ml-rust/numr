use super::super::wide_acc::FloatAcc;
use crate::dtype::{DType, Element};
use crate::ops::conv_common::Conv2dParams;

/// 2D convolution kernel with groups support.
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Arrays have sufficient size for the operation
/// - params contains valid dimensions
pub unsafe fn conv2d_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    bias: Option<*const T>,
    output: *mut T,
    params: Conv2dParams,
) {
    // Dispatch to SIMD for f16/bf16 on x86-64 and aarch64
    #[cfg(all(feature = "f16", any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        use super::super::simd::conv as simd_conv;

        match T::DTYPE {
            DType::F16 => {
                simd_conv::conv2d_f16(
                    input as *const half::f16,
                    weight as *const half::f16,
                    bias.map(|b| b as *const half::f16),
                    output as *mut half::f16,
                    params,
                );
                return;
            }
            DType::BF16 => {
                simd_conv::conv2d_bf16(
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
        conv2d_kernel_acc::<T, f64>(input, weight, bias, output, params);
    } else {
        conv2d_kernel_acc::<T, f32>(input, weight, bias, output, params);
    }
}

/// `conv2d_kernel` with the accumulator type fixed by the caller.
///
/// # Safety
///
/// Same as [`conv2d_kernel`].
unsafe fn conv2d_kernel_acc<T: Element, A: FloatAcc>(
    input: *const T,
    weight: *const T,
    bias: Option<*const T>,
    output: *mut T,
    params: Conv2dParams,
) {
    let Conv2dParams {
        batch,
        c_in,
        height,
        width,
        c_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        groups,
        pad_top,
        pad_bottom: _,
        pad_left,
        pad_right: _,
        output_h,
        output_w,
    } = params;

    let c_in_per_group = c_in / groups;
    let c_out_per_group = c_out / groups;

    // Input layout: (batch, c_in, height, width)
    // Weight layout: (c_out, c_in_per_group, kernel_h, kernel_w)
    // Output layout: (batch, c_out, output_h, output_w)

    for b in 0..batch {
        for g in 0..groups {
            let c_out_start = g * c_out_per_group;
            let c_in_start = g * c_in_per_group;

            for oc in 0..c_out_per_group {
                let c_out_idx = c_out_start + oc;

                for oy in 0..output_h {
                    for ox in 0..output_w {
                        let mut sum = A::ZERO;

                        for ic in 0..c_in_per_group {
                            let c_in_idx = c_in_start + ic;

                            for ky in 0..kernel_h {
                                for kx in 0..kernel_w {
                                    let iy_signed = (oy * stride_h) as isize
                                        + (ky * dilation_h) as isize
                                        - pad_top as isize;
                                    let ix_signed = (ox * stride_w) as isize
                                        + (kx * dilation_w) as isize
                                        - pad_left as isize;

                                    if iy_signed >= 0
                                        && (iy_signed as usize) < height
                                        && ix_signed >= 0
                                        && (ix_signed as usize) < width
                                    {
                                        let iy = iy_signed as usize;
                                        let ix = ix_signed as usize;

                                        // Input index
                                        let input_idx = b * c_in * height * width
                                            + c_in_idx * height * width
                                            + iy * width
                                            + ix;

                                        // Weight index
                                        let weight_idx =
                                            c_out_idx * c_in_per_group * kernel_h * kernel_w
                                                + ic * kernel_h * kernel_w
                                                + ky * kernel_w
                                                + kx;

                                        let in_val = A::from_elem(*input.add(input_idx));
                                        let w_val = A::from_elem(*weight.add(weight_idx));
                                        sum = sum.acc_add(in_val.acc_mul(w_val));
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias_ptr) = bias {
                            sum = sum.acc_add(A::from_elem(*bias_ptr.add(c_out_idx)));
                        }

                        // Output index
                        let output_idx = b * c_out * output_h * output_w
                            + c_out_idx * output_h * output_w
                            + oy * output_w
                            + ox;
                        *output.add(output_idx) = sum.to_elem::<T>();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::PaddingMode;
    use crate::ops::conv_common::validate_conv2d;

    #[test]
    fn test_conv2d_no_padding() {
        // Input: (1, 1, 3, 3) = identity-like
        // Weight: (1, 1, 2, 2) = all ones
        // Output: (1, 1, 2, 2)
        #[rustfmt::skip]
        let input = [
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let weight = [1.0f32, 1.0, 1.0, 1.0];
        let mut output = [0.0f32; 4];

        let params = validate_conv2d(
            &[1, 1, 3, 3],
            &[1, 1, 2, 2],
            None,
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            1,
            crate::dtype::DType::F32,
            crate::dtype::DType::F32,
            None,
        )
        .unwrap();

        unsafe {
            conv2d_kernel(
                input.as_ptr(),
                weight.as_ptr(),
                None,
                output.as_mut_ptr(),
                params,
            );
        }

        // Top-left: 1+2+4+5 = 12
        // Top-right: 2+3+5+6 = 16
        // Bottom-left: 4+5+7+8 = 24
        // Bottom-right: 5+6+8+9 = 28
        assert!((output[0] - 12.0).abs() < 1e-5);
        assert!((output[1] - 16.0).abs() < 1e-5);
        assert!((output[2] - 24.0).abs() < 1e-5);
        assert!((output[3] - 28.0).abs() < 1e-5);
    }
}
