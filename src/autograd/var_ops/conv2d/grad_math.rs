//! Gradient math for conv2d backward: input and weight gradient computation.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, PaddingMode, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::Runtime;

use super::super::conv_common::compute_padding_2d;

/// Compute conv2d backward for input using tensor operations.
///
/// d_input[n, c_in, h, w] = sum over c_out, kh, kw of:
///   weight[c_out, c_in, kh, kw] * grad_output[n, c_out, h*sh - pad_top + kh*dh, w*sw - pad_left + kw*dw]
pub(super) fn conv2d_input_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    weight: &crate::tensor::Tensor<R>,
    input_shape: &[usize],
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let batch = input_shape[0];
    let _c_in = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];
    let c_out = weight.shape()[0];
    let c_in_per_group = weight.shape()[1];
    let kernel_h = weight.shape()[2];
    let kernel_w = weight.shape()[3];
    let output_h = grad_output.shape()[2];
    let output_w = grad_output.shape()[3];
    let c_out_per_group = c_out / groups;

    let (pad_top, _pad_bottom, pad_left, _pad_right) =
        compute_padding_2d(padding, kernel_h, kernel_w, dilation.0, dilation.1);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_input = crate::tensor::Tensor::<R>::zeros(input_shape, dtype, device)?;

    for kh in 0..kernel_h {
        for kw in 0..kernel_w {
            // Extract weight slice at [kh, kw]: weight[:, :, kh, kw] → [c_out, c_in_per_group]
            let weight_kh = weight.narrow(2, kh, 1)?;
            let weight_khkw = weight_kh.narrow(3, kw, 1)?;
            let weight_2d = weight_khkw.squeeze(Some(3)).squeeze(Some(2));

            for oh in 0..output_h {
                let ih_pos = oh * stride.0 + kh * dilation.0;
                if ih_pos < pad_top || ih_pos >= pad_top + input_h {
                    continue;
                }
                let ih = ih_pos - pad_top;

                for ow in 0..output_w {
                    let iw_pos = ow * stride.1 + kw * dilation.1;
                    if iw_pos < pad_left || iw_pos >= pad_left + input_w {
                        continue;
                    }
                    let iw = iw_pos - pad_left;

                    // grad_output[:, :, oh, ow] → [batch, c_out]
                    let grad_o = grad_output.narrow(2, oh, 1)?.narrow(3, ow, 1)?;
                    let grad_o_2d = grad_o.squeeze(Some(3)).squeeze(Some(2));

                    for g in 0..groups {
                        let c_in_start = g * c_in_per_group;
                        let c_out_start = g * c_out_per_group;

                        let grad_g = grad_o_2d.narrow(1, c_out_start, c_out_per_group)?;
                        let weight_g = weight_2d.narrow(0, c_out_start, c_out_per_group)?;

                        // d_input contracts over c_out, so weight_g is used as-is:
                        // [batch, c_out_per_group] @ [c_out_per_group, c_in_per_group]
                        // = [batch, c_in_per_group]. Transposing weight_g here is only
                        // shape-valid when c_in_per_group == c_out_per_group, which is
                        // why 1x1-channel tests never caught it.
                        let contrib_g = client.matmul(&grad_g, &weight_g)?;

                        // Reshape to [batch, c_in_per_group, 1, 1]
                        let contrib_4d = contrib_g.reshape(&[batch, c_in_per_group, 1, 1])?;

                        // Get the slice at position (ih, iw) in the full d_input
                        let mut d_input_at = d_input.narrow(2, ih, 1)?.narrow(3, iw, 1)?;

                        // Get the group slice
                        let d_input_group = d_input_at.narrow(1, c_in_start, c_in_per_group)?;

                        // Add contribution
                        let updated_group = client.add(&d_input_group, &contrib_4d)?;

                        // Put back along dim 1
                        d_input_at =
                            client.slice_assign(&d_input_at, &updated_group, 1, c_in_start)?;

                        // Put back into d_input: first along dim 3 (width), then dim 2 (height)
                        let mut d_input_h = d_input.narrow(2, ih, 1)?;
                        d_input_h = client.slice_assign(&d_input_h, &d_input_at, 3, iw)?;
                        d_input = client.slice_assign(&d_input, &d_input_h, 2, ih)?;
                    }
                }
            }
        }
    }

    Ok(d_input)
}

/// Compute conv2d backward for weight using tensor operations.
///
/// d_weight[c_out, c_in, kh, kw] = sum over n, oh, ow of:
///   input[n, c_in, oh*sh - pad_top + kh*dh, ow*sw - pad_left + kw*dw] * grad_output[n, c_out, oh, ow]
pub(super) fn conv2d_weight_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    input: &crate::tensor::Tensor<R>,
    weight_shape: &[usize],
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let _batch = input.shape()[0];
    let _c_in = input.shape()[1];
    let input_h = input.shape()[2];
    let input_w = input.shape()[3];
    let c_out = weight_shape[0];
    let c_in_per_group = weight_shape[1];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];
    let output_h = grad_output.shape()[2];
    let output_w = grad_output.shape()[3];
    let c_out_per_group = c_out / groups;

    let (pad_top, _pad_bottom, pad_left, _pad_right) =
        compute_padding_2d(padding, kernel_h, kernel_w, dilation.0, dilation.1);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_weight = crate::tensor::Tensor::<R>::zeros(weight_shape, dtype, device)?;

    for oh in 0..output_h {
        for ow in 0..output_w {
            // grad_output[:, :, oh, ow] → [batch, c_out]
            let grad_o = grad_output.narrow(2, oh, 1)?.narrow(3, ow, 1)?;
            let grad_o_2d = grad_o.squeeze(Some(3)).squeeze(Some(2));

            for kh in 0..kernel_h {
                let ih_pos = oh * stride.0 + kh * dilation.0;
                if ih_pos < pad_top || ih_pos >= pad_top + input_h {
                    continue;
                }
                let ih = ih_pos - pad_top;

                for kw in 0..kernel_w {
                    let iw_pos = ow * stride.1 + kw * dilation.1;
                    if iw_pos < pad_left || iw_pos >= pad_left + input_w {
                        continue;
                    }
                    let iw = iw_pos - pad_left;

                    // input[:, :, ih, iw] → [batch, c_in]
                    let input_hw = input.narrow(2, ih, 1)?.narrow(3, iw, 1)?;
                    let input_2d = input_hw.squeeze(Some(3)).squeeze(Some(2));

                    for g in 0..groups {
                        let c_in_start = g * c_in_per_group;
                        let c_out_start = g * c_out_per_group;

                        let input_g = input_2d.narrow(1, c_in_start, c_in_per_group)?;
                        let grad_g = grad_o_2d.narrow(1, c_out_start, c_out_per_group)?;

                        // [c_out_per_group, batch] @ [batch, c_in_per_group]
                        // = [c_out_per_group, c_in_per_group]
                        let contrib_2d = client.matmul(&grad_g.transpose(0, 1)?, &input_g)?;

                        // Reshape to [c_out_per_group, c_in_per_group, 1, 1]
                        let contrib_4d =
                            contrib_2d.reshape(&[c_out_per_group, c_in_per_group, 1, 1])?;

                        // Get the weight slice at kernel position (kh, kw)
                        let mut d_weight_at = d_weight.narrow(2, kh, 1)?.narrow(3, kw, 1)?;

                        // Get the group slice
                        let d_weight_group = d_weight_at.narrow(0, c_out_start, c_out_per_group)?;

                        // Add contribution
                        let updated_group = client.add(&d_weight_group, &contrib_4d)?;

                        // Put back along dim 0
                        d_weight_at =
                            client.slice_assign(&d_weight_at, &updated_group, 0, c_out_start)?;

                        // Put back into d_weight: first along dim 3, then dim 2
                        let mut d_weight_kh = d_weight.narrow(2, kh, 1)?;
                        d_weight_kh = client.slice_assign(&d_weight_kh, &d_weight_at, 3, kw)?;
                        d_weight = client.slice_assign(&d_weight, &d_weight_kh, 2, kh)?;
                    }
                }
            }
        }
    }

    Ok(d_weight)
}
