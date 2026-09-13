use super::super::wide_acc::FloatAcc;
use crate::dtype::{DType, Element};
use crate::ops::conv_common::Conv2dParams;

/// Depthwise 2D convolution kernel.
///
/// Optimized path for depthwise convolution where each channel is convolved independently.
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Arrays have sufficient size for the operation
/// - params contains valid dimensions (groups == c_in == c_out)
pub unsafe fn depthwise_conv2d_kernel<T: Element>(
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
                simd_conv::depthwise_conv2d_f16(
                    input as *const half::f16,
                    weight as *const half::f16,
                    bias.map(|b| b as *const half::f16),
                    output as *mut half::f16,
                    params,
                );
                return;
            }
            DType::BF16 => {
                simd_conv::depthwise_conv2d_bf16(
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
        depthwise_conv2d_kernel_acc::<T, f64>(input, weight, bias, output, params);
    } else {
        depthwise_conv2d_kernel_acc::<T, f32>(input, weight, bias, output, params);
    }
}

/// `depthwise_conv2d_kernel` with the accumulator type fixed by the caller.
///
/// # Safety
///
/// Same as [`depthwise_conv2d_kernel`].
unsafe fn depthwise_conv2d_kernel_acc<T: Element, A: FloatAcc>(
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
        c_out: _,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        groups: _,
        pad_top,
        pad_bottom: _,
        pad_left,
        pad_right: _,
        output_h,
        output_w,
    } = params;

    // For depthwise: c_in == c_out, groups == c_in
    // Weight layout: (c_in, 1, kernel_h, kernel_w)
    // Each channel has its own kernel

    for b in 0..batch {
        for c in 0..c_in {
            for oy in 0..output_h {
                for ox in 0..output_w {
                    let mut sum = A::ZERO;

                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            let iy_signed = (oy * stride_h) as isize + (ky * dilation_h) as isize
                                - pad_top as isize;
                            let ix_signed = (ox * stride_w) as isize + (kx * dilation_w) as isize
                                - pad_left as isize;

                            if iy_signed >= 0
                                && (iy_signed as usize) < height
                                && ix_signed >= 0
                                && (ix_signed as usize) < width
                            {
                                let iy = iy_signed as usize;
                                let ix = ix_signed as usize;

                                // Input index: (b, c, iy, ix)
                                let input_idx = b * c_in * height * width
                                    + c * height * width
                                    + iy * width
                                    + ix;

                                // Weight index: (c, 0, ky, kx) = c * kernel_h * kernel_w + ky * kernel_w + kx
                                let weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;

                                let in_val = A::from_elem(*input.add(input_idx));
                                let w_val = A::from_elem(*weight.add(weight_idx));
                                sum = sum.acc_add(in_val.acc_mul(w_val));
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(bias_ptr) = bias {
                        sum = sum.acc_add(A::from_elem(*bias_ptr.add(c)));
                    }

                    // Output index: (b, c, oy, ox)
                    let output_idx = b * c_in * output_h * output_w
                        + c * output_h * output_w
                        + oy * output_w
                        + ox;
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
    use crate::ops::conv_common::validate_depthwise_conv2d;

    #[test]
    fn test_depthwise_conv2d() {
        // Input: (1, 2, 3, 3) - 2 channels
        // Weight: (2, 1, 2, 2) - one 2x2 kernel per channel
        #[rustfmt::skip]
        let input = [
            // Channel 0
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            // Channel 1
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ];
        // Channel 0 kernel: all 1s
        // Channel 1 kernel: all 2s
        let weight = [
            1.0f32, 1.0, 1.0, 1.0, // channel 0
            2.0, 2.0, 2.0, 2.0, // channel 1
        ];
        let mut output = [0.0f32; 8]; // (1, 2, 2, 2)

        let params = validate_depthwise_conv2d(
            &[1, 2, 3, 3],
            &[2, 1, 2, 2],
            None,
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            crate::dtype::DType::F32,
            crate::dtype::DType::F32,
            None,
        )
        .unwrap();

        unsafe {
            depthwise_conv2d_kernel(
                input.as_ptr(),
                weight.as_ptr(),
                None,
                output.as_mut_ptr(),
                params,
            );
        }

        // Channel 0: same as test_conv2d_no_padding
        assert!((output[0] - 12.0).abs() < 1e-5);
        assert!((output[1] - 16.0).abs() < 1e-5);
        assert!((output[2] - 24.0).abs() < 1e-5);
        assert!((output[3] - 28.0).abs() < 1e-5);

        // Channel 1: (9+8+6+5)*2 = 56, etc.
        assert!((output[4] - 56.0).abs() < 1e-5); // (9+8+6+5)*2
        assert!((output[5] - 48.0).abs() < 1e-5); // (8+7+5+4)*2
        assert!((output[6] - 32.0).abs() < 1e-5); // (6+5+3+2)*2
        assert!((output[7] - 24.0).abs() < 1e-5); // (5+4+2+1)*2
    }
}
