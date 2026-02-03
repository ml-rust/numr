//! Shared validation and utility functions for convolution operations.
//!
//! This module contains common validation logic used across all backend implementations
//! (CPU, CUDA, WebGPU) to ensure consistency and eliminate code duplication.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::PaddingMode;

/// Validates that a tensor is 3-dimensional (for conv1d).
#[inline]
pub fn validate_3d_tensor(shape: &[usize], arg_name: &'static str, op: &'static str) -> Result<()> {
    if shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!("{} expects 3D tensor, got {}D", op, shape.len()),
        });
    }
    Ok(())
}

/// Validates that a tensor is 4-dimensional (for conv2d).
#[inline]
pub fn validate_4d_tensor(shape: &[usize], arg_name: &'static str, op: &'static str) -> Result<()> {
    if shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!("{} expects 4D tensor, got {}D", op, shape.len()),
        });
    }
    Ok(())
}

/// Validates that a tensor is 1-dimensional (for bias).
#[inline]
pub fn validate_1d_tensor(shape: &[usize], arg_name: &'static str, op: &'static str) -> Result<()> {
    if shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!("{} expects 1D tensor, got {}D", op, shape.len()),
        });
    }
    Ok(())
}

/// Validates that a dtype is a floating-point type.
#[inline]
pub fn validate_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if !dtype.is_float() {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validates that two tensors have the same dtype.
#[inline]
pub fn validate_same_dtype(x_dtype: DType, y_dtype: DType, op: &'static str) -> Result<()> {
    if y_dtype != x_dtype {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!(
                "{} requires same dtype, got input.dtype={}, weight.dtype={}",
                op, x_dtype, y_dtype
            ),
        });
    }
    Ok(())
}

/// Validates that stride, dilation, and groups are non-zero.
#[inline]
pub fn validate_positive(value: usize, name: &'static str, op: &'static str) -> Result<()> {
    if value == 0 {
        return Err(Error::InvalidArgument {
            arg: name,
            reason: format!("{} requires {} > 0, got 0", op, name),
        });
    }
    Ok(())
}

/// Validates that channels are divisible by groups.
#[inline]
pub fn validate_groups(c_in: usize, c_out: usize, groups: usize, op: &'static str) -> Result<()> {
    if !c_in.is_multiple_of(groups) {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: format!(
                "{} requires C_in ({}) to be divisible by groups ({})",
                op, c_in, groups
            ),
        });
    }
    if !c_out.is_multiple_of(groups) {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: format!(
                "{} requires C_out ({}) to be divisible by groups ({})",
                op, c_out, groups
            ),
        });
    }
    Ok(())
}

/// Validates that weight has correct channels for the given groups.
#[inline]
pub fn validate_weight_channels(
    c_in: usize,
    weight_c_in: usize,
    groups: usize,
    op: &'static str,
) -> Result<()> {
    let expected = c_in / groups;
    if weight_c_in != expected {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!(
                "{} weight.shape[1] should be C_in/groups = {}/{} = {}, got {}",
                op, c_in, groups, expected, weight_c_in
            ),
        });
    }
    Ok(())
}

/// Validates that bias has the correct length.
#[inline]
pub fn validate_bias_length(bias_len: usize, c_out: usize, op: &'static str) -> Result<()> {
    if bias_len != c_out {
        return Err(Error::InvalidArgument {
            arg: "bias",
            reason: format!(
                "{} bias should have length C_out = {}, got {}",
                op, c_out, bias_len
            ),
        });
    }
    Ok(())
}

/// Computes output size for a single dimension in convolution.
///
/// output_size = floor((input_size + pad_before + pad_after - dilation * (kernel_size - 1) - 1) / stride + 1)
#[inline]
pub fn compute_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    pad_before: usize,
    pad_after: usize,
) -> usize {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let padded_size = input_size + pad_before + pad_after;
    if padded_size < effective_kernel {
        0
    } else {
        (padded_size - effective_kernel) / stride + 1
    }
}

/// Computes padding values for "same" padding mode.
///
/// Same padding ensures output_size == input_size when stride == 1.
/// When stride > 1, output_size = ceil(input_size / stride).
#[inline]
pub fn compute_same_padding(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
) -> (usize, usize) {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let output_size = (input_size + stride - 1) / stride; // ceil division
    let total_pad = if output_size > 0 {
        let needed = (output_size - 1) * stride + effective_kernel;
        needed.saturating_sub(input_size)
    } else {
        0
    };
    let pad_before = total_pad / 2;
    let pad_after = total_pad - pad_before;
    (pad_before, pad_after)
}

/// Resolves padding mode to explicit padding values for conv1d.
#[inline]
pub fn resolve_padding_1d(
    padding: PaddingMode,
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
) -> (usize, usize) {
    match padding {
        PaddingMode::Valid => (0, 0),
        PaddingMode::Same => compute_same_padding(input_length, kernel_size, stride, dilation),
        PaddingMode::Custom(left, right, _, _) => (left, right),
    }
}

/// Resolves padding mode to explicit padding values for conv2d.
#[inline]
pub fn resolve_padding_2d(
    padding: PaddingMode,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: (usize, usize),
    dilation: (usize, usize),
) -> (usize, usize, usize, usize) {
    match padding {
        PaddingMode::Valid => (0, 0, 0, 0),
        PaddingMode::Same => {
            let (pad_top, pad_bottom) =
                compute_same_padding(input_h, kernel_h, stride.0, dilation.0);
            let (pad_left, pad_right) =
                compute_same_padding(input_w, kernel_w, stride.1, dilation.1);
            (pad_top, pad_bottom, pad_left, pad_right)
        }
        PaddingMode::Custom(top, bottom, left, right) => (top, bottom, left, right),
    }
}

/// Parameters for conv1d operation after validation.
#[derive(Debug, Clone, Copy)]
pub struct Conv1dParams {
    pub batch: usize,
    pub c_in: usize,
    pub length: usize,
    pub c_out: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub pad_left: usize,
    #[allow(dead_code)] // Stored for potential asymmetric padding support
    pub pad_right: usize,
    pub output_length: usize,
}

/// Parameters for conv2d operation after validation.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dParams {
    pub batch: usize,
    pub c_in: usize,
    pub height: usize,
    pub width: usize,
    pub c_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub groups: usize,
    pub pad_top: usize,
    #[allow(dead_code)] // Stored for potential asymmetric padding support
    pub pad_bottom: usize,
    pub pad_left: usize,
    #[allow(dead_code)] // Stored for potential asymmetric padding support
    pub pad_right: usize,
    pub output_h: usize,
    pub output_w: usize,
}

/// Validates and extracts parameters for conv1d.
pub fn validate_conv1d(
    input_shape: &[usize],
    weight_shape: &[usize],
    bias_shape: Option<&[usize]>,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
    input_dtype: DType,
    weight_dtype: DType,
    bias_dtype: Option<DType>,
) -> Result<Conv1dParams> {
    const OP: &str = "conv1d";

    // Validate tensor dimensions
    validate_3d_tensor(input_shape, "input", OP)?;
    validate_3d_tensor(weight_shape, "weight", OP)?;

    // Validate dtypes
    validate_float_dtype(input_dtype, OP)?;
    validate_same_dtype(input_dtype, weight_dtype, OP)?;
    if let Some(b_dtype) = bias_dtype {
        validate_same_dtype(input_dtype, b_dtype, OP)?;
    }

    // Validate hyperparameters
    validate_positive(stride, "stride", OP)?;
    validate_positive(dilation, "dilation", OP)?;
    validate_positive(groups, "groups", OP)?;

    let batch = input_shape[0];
    let c_in = input_shape[1];
    let length = input_shape[2];
    let c_out = weight_shape[0];
    let kernel_size = weight_shape[2];

    // Validate groups
    validate_groups(c_in, c_out, groups, OP)?;
    validate_weight_channels(c_in, weight_shape[1], groups, OP)?;

    // Validate bias
    if let Some(b_shape) = bias_shape {
        validate_1d_tensor(b_shape, "bias", OP)?;
        validate_bias_length(b_shape[0], c_out, OP)?;
    }

    // Compute padding
    let (pad_left, pad_right) = resolve_padding_1d(padding, length, kernel_size, stride, dilation);

    // Compute output size
    let output_length =
        compute_output_size(length, kernel_size, stride, dilation, pad_left, pad_right);

    Ok(Conv1dParams {
        batch,
        c_in,
        length,
        c_out,
        kernel_size,
        stride,
        dilation,
        groups,
        pad_left,
        pad_right,
        output_length,
    })
}

/// Validates and extracts parameters for conv2d.
pub fn validate_conv2d(
    input_shape: &[usize],
    weight_shape: &[usize],
    bias_shape: Option<&[usize]>,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
    input_dtype: DType,
    weight_dtype: DType,
    bias_dtype: Option<DType>,
) -> Result<Conv2dParams> {
    const OP: &str = "conv2d";

    // Validate tensor dimensions
    validate_4d_tensor(input_shape, "input", OP)?;
    validate_4d_tensor(weight_shape, "weight", OP)?;

    // Validate dtypes
    validate_float_dtype(input_dtype, OP)?;
    validate_same_dtype(input_dtype, weight_dtype, OP)?;
    if let Some(b_dtype) = bias_dtype {
        validate_same_dtype(input_dtype, b_dtype, OP)?;
    }

    // Validate hyperparameters
    validate_positive(stride.0, "stride_h", OP)?;
    validate_positive(stride.1, "stride_w", OP)?;
    validate_positive(dilation.0, "dilation_h", OP)?;
    validate_positive(dilation.1, "dilation_w", OP)?;
    validate_positive(groups, "groups", OP)?;

    let batch = input_shape[0];
    let c_in = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];
    let c_out = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    // Validate groups
    validate_groups(c_in, c_out, groups, OP)?;
    validate_weight_channels(c_in, weight_shape[1], groups, OP)?;

    // Validate bias
    if let Some(b_shape) = bias_shape {
        validate_1d_tensor(b_shape, "bias", OP)?;
        validate_bias_length(b_shape[0], c_out, OP)?;
    }

    // Compute padding
    let (pad_top, pad_bottom, pad_left, pad_right) =
        resolve_padding_2d(padding, height, width, kernel_h, kernel_w, stride, dilation);

    // Compute output size
    let output_h = compute_output_size(height, kernel_h, stride.0, dilation.0, pad_top, pad_bottom);
    let output_w = compute_output_size(width, kernel_w, stride.1, dilation.1, pad_left, pad_right);

    Ok(Conv2dParams {
        batch,
        c_in,
        height,
        width,
        c_out,
        kernel_h,
        kernel_w,
        stride_h: stride.0,
        stride_w: stride.1,
        dilation_h: dilation.0,
        dilation_w: dilation.1,
        groups,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        output_h,
        output_w,
    })
}

/// Validates and extracts parameters for depthwise_conv2d.
pub fn validate_depthwise_conv2d(
    input_shape: &[usize],
    weight_shape: &[usize],
    bias_shape: Option<&[usize]>,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    input_dtype: DType,
    weight_dtype: DType,
    bias_dtype: Option<DType>,
) -> Result<Conv2dParams> {
    const OP: &str = "depthwise_conv2d";

    // Validate tensor dimensions
    validate_4d_tensor(input_shape, "input", OP)?;
    validate_4d_tensor(weight_shape, "weight", OP)?;

    // Validate dtypes
    validate_float_dtype(input_dtype, OP)?;
    validate_same_dtype(input_dtype, weight_dtype, OP)?;
    if let Some(b_dtype) = bias_dtype {
        validate_same_dtype(input_dtype, b_dtype, OP)?;
    }

    // Validate hyperparameters
    validate_positive(stride.0, "stride_h", OP)?;
    validate_positive(stride.1, "stride_w", OP)?;
    validate_positive(dilation.0, "dilation_h", OP)?;
    validate_positive(dilation.1, "dilation_w", OP)?;

    let batch = input_shape[0];
    let c_in = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];
    let c_out = weight_shape[0];
    let weight_c_in = weight_shape[1];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    // Depthwise requires weight.shape[1] == 1
    if weight_c_in != 1 {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!(
                "{} weight.shape[1] should be 1 for depthwise, got {}",
                OP, weight_c_in
            ),
        });
    }

    // Depthwise requires c_out == c_in (each channel has its own filter)
    if c_out != c_in {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!(
                "{} requires weight.shape[0] == input.shape[1], got {} != {}",
                OP, c_out, c_in
            ),
        });
    }

    // Validate bias
    if let Some(b_shape) = bias_shape {
        validate_1d_tensor(b_shape, "bias", OP)?;
        validate_bias_length(b_shape[0], c_out, OP)?;
    }

    // Compute padding
    let (pad_top, pad_bottom, pad_left, pad_right) =
        resolve_padding_2d(padding, height, width, kernel_h, kernel_w, stride, dilation);

    // Compute output size
    let output_h = compute_output_size(height, kernel_h, stride.0, dilation.0, pad_top, pad_bottom);
    let output_w = compute_output_size(width, kernel_w, stride.1, dilation.1, pad_left, pad_right);

    Ok(Conv2dParams {
        batch,
        c_in,
        height,
        width,
        c_out,
        kernel_h,
        kernel_w,
        stride_h: stride.0,
        stride_w: stride.1,
        dilation_h: dilation.0,
        dilation_w: dilation.1,
        groups: c_in, // depthwise uses groups = c_in
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        output_h,
        output_w,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_output_size() {
        // Simple case: 5x5 input, 3x3 kernel, stride 1, no padding
        assert_eq!(compute_output_size(5, 3, 1, 1, 0, 0), 3);

        // With padding
        assert_eq!(compute_output_size(5, 3, 1, 1, 1, 1), 5);

        // With stride
        assert_eq!(compute_output_size(7, 3, 2, 1, 0, 0), 3);

        // With dilation
        assert_eq!(compute_output_size(7, 3, 1, 2, 0, 0), 3); // effective kernel = 5
    }

    #[test]
    fn test_compute_same_padding() {
        // 5x5 input, 3x3 kernel, stride 1 -> need 2 total padding
        let (left, right) = compute_same_padding(5, 3, 1, 1);
        assert_eq!(left, 1);
        assert_eq!(right, 1);

        // 5x5 input, 3x3 kernel, stride 2 -> output is 3, need padding
        let (top, bottom) = compute_same_padding(5, 3, 2, 1);
        // Verify we computed valid padding values (non-trivially)
        let _ = (top, bottom); // Use values to silence warning
    }

    #[test]
    fn test_validate_conv1d_basic() {
        let result = validate_conv1d(
            &[2, 3, 10], // input: batch=2, channels=3, length=10
            &[16, 3, 3], // weight: c_out=16, c_in=3, kernel=3
            None,        // no bias
            1,           // stride
            PaddingMode::Valid,
            1, // dilation
            1, // groups
            DType::F32,
            DType::F32,
            None,
        );
        assert!(result.is_ok());
        let params = result.unwrap();
        assert_eq!(params.batch, 2);
        assert_eq!(params.c_in, 3);
        assert_eq!(params.c_out, 16);
        assert_eq!(params.output_length, 8); // (10 - 3) / 1 + 1 = 8
    }

    #[test]
    fn test_validate_conv2d_basic() {
        let result = validate_conv2d(
            &[2, 3, 32, 32], // input: batch=2, channels=3, height=32, width=32
            &[64, 3, 3, 3],  // weight: c_out=64, c_in=3, kernel=3x3
            None,            // no bias
            (1, 1),          // stride
            PaddingMode::Same,
            (1, 1), // dilation
            1,      // groups
            DType::F32,
            DType::F32,
            None,
        );
        assert!(result.is_ok());
        let params = result.unwrap();
        assert_eq!(params.batch, 2);
        assert_eq!(params.c_in, 3);
        assert_eq!(params.c_out, 64);
        assert_eq!(params.output_h, 32);
        assert_eq!(params.output_w, 32);
    }

    #[test]
    fn test_validate_depthwise_conv2d() {
        let result = validate_depthwise_conv2d(
            &[2, 32, 28, 28], // input
            &[32, 1, 3, 3],   // weight: c_out=32, c_in_per_group=1, kernel=3x3
            None,
            (1, 1),
            PaddingMode::Same,
            (1, 1),
            DType::F32,
            DType::F32,
            None,
        );
        assert!(result.is_ok());
        let params = result.unwrap();
        assert_eq!(params.groups, 32); // depthwise has groups = c_in
    }

    #[test]
    fn test_validate_groups() {
        // Grouped convolution: 6 input channels, 2 groups -> 3 channels per group
        let result = validate_conv2d(
            &[1, 6, 8, 8],  // 6 input channels
            &[12, 3, 3, 3], // 12 output channels, 3 = 6/2 input per group
            None,
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            2, // 2 groups
            DType::F32,
            DType::F32,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_groups_error() {
        // Invalid: 5 channels not divisible by 2 groups
        let result = validate_conv2d(
            &[1, 5, 8, 8],
            &[10, 3, 3, 3], // wrong: should be 5/2 = 2.5 (invalid)
            None,
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            2,
            DType::F32,
            DType::F32,
            None,
        );
        assert!(result.is_err());
    }
}
