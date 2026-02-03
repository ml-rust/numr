//! CPU convolution kernels.
//!
//! Direct convolution implementations without im2col transformation.

use crate::dtype::Element;
use crate::ops::conv_common::{Conv1dParams, Conv2dParams};

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
                    let mut sum = T::zero();

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

                                let in_val = *input.add(input_idx);
                                let w_val = *weight.add(weight_idx);
                                sum = sum + in_val * w_val;
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(bias_ptr) = bias {
                        sum = sum + *bias_ptr.add(c_out_idx);
                    }

                    // Output index: b * c_out * output_length + c_out_idx * output_length + ox
                    let output_idx = b * c_out * output_length + c_out_idx * output_length + ox;
                    *output.add(output_idx) = sum;
                }
            }
        }
    }
}

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
                        let mut sum = T::zero();

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

                                        let in_val = *input.add(input_idx);
                                        let w_val = *weight.add(weight_idx);
                                        sum = sum + in_val * w_val;
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias_ptr) = bias {
                            sum = sum + *bias_ptr.add(c_out_idx);
                        }

                        // Output index
                        let output_idx = b * c_out * output_h * output_w
                            + c_out_idx * output_h * output_w
                            + oy * output_w
                            + ox;
                        *output.add(output_idx) = sum;
                    }
                }
            }
        }
    }
}

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
                    let mut sum = T::zero();

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

                                let in_val = *input.add(input_idx);
                                let w_val = *weight.add(weight_idx);
                                sum = sum + in_val * w_val;
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(bias_ptr) = bias {
                        sum = sum + *bias_ptr.add(c);
                    }

                    // Output index: (b, c, oy, ox)
                    let output_idx = b * c_in * output_h * output_w
                        + c * output_h * output_w
                        + oy * output_w
                        + ox;
                    *output.add(output_idx) = sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::PaddingMode;
    use crate::ops::conv_common::{validate_conv1d, validate_conv2d, validate_depthwise_conv2d};

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
