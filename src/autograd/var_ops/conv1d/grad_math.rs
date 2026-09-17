//! Gradient math for conv1d backward: input and weight gradient computation.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, PaddingMode, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::Runtime;

use super::super::conv_common::compute_padding;

/// Compute conv1d backward for input using tensor operations.
///
/// d_input[n, c_in, l] = sum over c_out, k of:
///   weight[c_out, c_in, k] * grad_output[n, c_out, l*stride - pad + k*dilation]
///
/// This is equivalent to a transposed convolution (conv_transpose1d).
///
/// IMPLEMENTATION NOTE: Uses tensor operations (no to_vec/to_cpu). All computation
/// is performed through the client, which works on any backend. The Rust loop
/// structures the iteration, but actual mathematical operations (matmul, add)
/// happen on the device via the client.
pub(super) fn conv1d_input_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    weight: &crate::tensor::Tensor<R>,
    input_shape: &[usize],
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let batch = input_shape[0];
    let _c_in = input_shape[1];
    let input_len = input_shape[2];
    let c_out = weight.shape()[0];
    let c_in_per_group = weight.shape()[1];
    let kernel_size = weight.shape()[2];
    let output_len = grad_output.shape()[2];
    let c_out_per_group = c_out / groups;

    let (pad_left, _pad_right) = compute_padding(padding, kernel_size, dilation);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_input = crate::tensor::Tensor::<R>::zeros(input_shape, dtype, device)?;

    // Accumulate contributions by iterating and accumulating tensor operations
    for k in 0..kernel_size {
        let weight_k = weight.narrow(2, k, 1)?;
        let weight_k = weight_k.squeeze(Some(2));

        for o in 0..output_len {
            let i_pos = o * stride + k * dilation;

            if i_pos >= pad_left && i_pos < pad_left + input_len {
                let i = i_pos - pad_left;

                let grad_o = grad_output.narrow(2, o, 1)?;
                let grad_o = grad_o.squeeze(Some(2));

                for g in 0..groups {
                    let c_in_start = g * c_in_per_group;
                    let c_out_start = g * c_out_per_group;

                    let grad_g = grad_o.narrow(1, c_out_start, c_out_per_group)?;
                    let weight_g = weight_k.narrow(0, c_out_start, c_out_per_group)?;

                    // d_input contracts over c_out, so weight_g is used as-is:
                    // [batch, c_out_per_group] @ [c_out_per_group, c_in_per_group]
                    // = [batch, c_in_per_group]. Transposing weight_g here is only
                    // shape-valid when c_in_per_group == c_out_per_group, which is
                    // why 1x1-channel tests never caught it.
                    let contrib_g = client.matmul(&grad_g, &weight_g)?;

                    // Reshape to [batch, c_in_per_group, 1]
                    let contrib_g_3d = contrib_g.reshape(&[batch, c_in_per_group, 1])?;

                    // Get the slice at position i in the full d_input
                    let mut d_input_at_i = d_input.narrow(2, i, 1)?; // [batch, c_in, 1]

                    // Get the group slice
                    let d_input_group = d_input_at_i.narrow(1, c_in_start, c_in_per_group)?; // [batch, c_in_per_group, 1]

                    // Add contribution
                    let updated_group = client.add(&d_input_group, &contrib_g_3d)?;

                    // Now put it back. We need to use slice_assign correctly.
                    // The challenge is that we have a [batch, c_in_per_group, 1] but
                    // we need to update a specific region of a [batch, c_in, 1].
                    // slice_assign along dim 1 requires src to have the same dimension count
                    // and the same size on all dims except dim.
                    // So src should be [batch, c_in_per_group, 1] and we use dim=1, start=c_in_start
                    d_input_at_i =
                        client.slice_assign(&d_input_at_i, &updated_group, 1, c_in_start)?;

                    // Now put d_input_at_i back into d_input at position i
                    d_input = client.slice_assign(&d_input, &d_input_at_i, 2, i)?;
                }
            }
        }
    }

    Ok(d_input)
}

/// Compute conv1d backward for weight using tensor operations.
///
/// d_weight[c_out, c_in, k] = sum over n, o of:
///   input[n, c_in, o*stride - pad + k*dilation] * grad_output[n, c_out, o]
///
/// This function uses only tensor operations (no to_vec/to_cpu). All computation
/// is performed through the client, which works on any backend.
pub(in crate::autograd::var_ops) fn conv1d_weight_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    input: &crate::tensor::Tensor<R>,
    weight_shape: &[usize],
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let _batch = input.shape()[0];
    let _c_in = input.shape()[1];
    let input_len = input.shape()[2];
    let c_out = weight_shape[0];
    let c_in_per_group = weight_shape[1];
    let kernel_size = weight_shape[2];
    let output_len = grad_output.shape()[2];
    let c_out_per_group = c_out / groups;

    let (pad_left, _pad_right) = compute_padding(padding, kernel_size, dilation);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_weight = crate::tensor::Tensor::<R>::zeros(weight_shape, dtype, device)?;

    // Accumulate contributions by iterating and accumulating tensor operations
    for o in 0..output_len {
        for k in 0..kernel_size {
            let i_pos = o * stride + k * dilation;

            if i_pos >= pad_left && i_pos < pad_left + input_len {
                let i = i_pos - pad_left;

                let input_i = input.narrow(2, i, 1)?;
                let input_i = input_i.squeeze(Some(2));

                let grad_o = grad_output.narrow(2, o, 1)?;
                let grad_o = grad_o.squeeze(Some(2));

                for g in 0..groups {
                    let c_in_start = g * c_in_per_group;
                    let c_out_start = g * c_out_per_group;

                    let input_g = input_i.narrow(1, c_in_start, c_in_per_group)?;
                    let grad_g = grad_o.narrow(1, c_out_start, c_out_per_group)?;

                    // Compute: [c_out_per_group, batch] @ [batch, c_in_per_group]
                    //        = [c_out_per_group, c_in_per_group]
                    let contrib_2d = client.matmul(&grad_g.transpose(0, 1)?, &input_g)?;

                    // Reshape to [c_out_per_group, c_in_per_group, 1]
                    let contrib_3d = contrib_2d.reshape(&[c_out_per_group, c_in_per_group, 1])?;

                    // Get the weight slice at kernel position k
                    let mut d_weight_at_k = d_weight.narrow(2, k, 1)?; // [c_out, c_in_per_group, 1]

                    // Get the group slice
                    let d_weight_group = d_weight_at_k.narrow(0, c_out_start, c_out_per_group)?; // [c_out_per_group, c_in_per_group, 1]

                    // Add contribution
                    let updated_group = client.add(&d_weight_group, &contrib_3d)?;

                    // Put back along dimension 0
                    d_weight_at_k =
                        client.slice_assign(&d_weight_at_k, &updated_group, 0, c_out_start)?;

                    // Put back into d_weight along dimension 2
                    d_weight = client.slice_assign(&d_weight, &d_weight_at_k, 2, k)?;
                }
            }
        }
    }

    Ok(d_weight)
}
