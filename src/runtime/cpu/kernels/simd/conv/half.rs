//! f16/bf16 convolution wrappers via bulk f32 conversion
//!
//! Convolutions need random access across the entire input (sliding window),
//! so block-convert is not feasible. Instead we pre-convert all inputs to f32
//! using a single allocation (partitioned into input/weight/output/bias regions)
//! to minimize allocator overhead.

use super::super::half_convert_utils::*;
use super::*;
use crate::ops::conv_common::{Conv1dParams, Conv2dParams};

/// Generate f16 and bf16 conv wrappers that pre-convert to f32 via a single allocation.
macro_rules! half_conv_wrapper {
    (
        $fn_f16:ident, $fn_bf16:ident, $f32_fn:path, $params_ty:ty,
        sizes: |$p:ident| ($in_expr:expr, $w_expr:expr, $out_expr:expr, $bias_expr:expr)
    ) => {
        #[cfg(feature = "f16")]
        pub unsafe fn $fn_f16(
            input: *const ::half::f16,
            weight: *const ::half::f16,
            bias: Option<*const ::half::f16>,
            output: *mut ::half::f16,
            $p: $params_ty,
        ) {
            let (input_len, weight_len, output_len, bias_len) =
                ($in_expr, $w_expr, $out_expr, $bias_expr);
            let total =
                input_len + weight_len + output_len + if bias.is_some() { bias_len } else { 0 };
            let mut buf = vec![0.0f32; total];
            let (input_f32, rest) = buf.split_at_mut(input_len);
            let (weight_f32, rest) = rest.split_at_mut(weight_len);
            let (output_f32, bias_f32) = rest.split_at_mut(output_len);

            convert_f16_to_f32(input as *const u16, input_f32.as_mut_ptr(), input_len);
            convert_f16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), weight_len);

            let bias_ptr = if let Some(b) = bias {
                convert_f16_to_f32(b as *const u16, bias_f32.as_mut_ptr(), bias_len);
                Some(bias_f32.as_ptr())
            } else {
                None
            };

            $f32_fn(
                input_f32.as_ptr(),
                weight_f32.as_ptr(),
                bias_ptr,
                output_f32.as_mut_ptr(),
                $p,
            );
            convert_f32_to_f16(output_f32.as_ptr(), output as *mut u16, output_len);
        }

        #[cfg(feature = "f16")]
        pub unsafe fn $fn_bf16(
            input: *const ::half::bf16,
            weight: *const ::half::bf16,
            bias: Option<*const ::half::bf16>,
            output: *mut ::half::bf16,
            $p: $params_ty,
        ) {
            let (input_len, weight_len, output_len, bias_len) =
                ($in_expr, $w_expr, $out_expr, $bias_expr);
            let total =
                input_len + weight_len + output_len + if bias.is_some() { bias_len } else { 0 };
            let mut buf = vec![0.0f32; total];
            let (input_f32, rest) = buf.split_at_mut(input_len);
            let (weight_f32, rest) = rest.split_at_mut(weight_len);
            let (output_f32, bias_f32) = rest.split_at_mut(output_len);

            convert_bf16_to_f32(input as *const u16, input_f32.as_mut_ptr(), input_len);
            convert_bf16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), weight_len);

            let bias_ptr = if let Some(b) = bias {
                convert_bf16_to_f32(b as *const u16, bias_f32.as_mut_ptr(), bias_len);
                Some(bias_f32.as_ptr())
            } else {
                None
            };

            $f32_fn(
                input_f32.as_ptr(),
                weight_f32.as_ptr(),
                bias_ptr,
                output_f32.as_mut_ptr(),
                $p,
            );
            convert_f32_to_bf16(output_f32.as_ptr(), output as *mut u16, output_len);
        }
    };
}

half_conv_wrapper!(
    conv1d_f16, conv1d_bf16, conv1d_f32, Conv1dParams,
    sizes: |params| (
        params.batch * params.c_in * params.length,
        params.c_out * (params.c_in / params.groups) * params.kernel_size,
        params.batch * params.c_out * params.output_length,
        params.c_out
    )
);

half_conv_wrapper!(
    conv2d_f16, conv2d_bf16, conv2d_f32, Conv2dParams,
    sizes: |params| (
        params.batch * params.c_in * params.height * params.width,
        params.c_out * (params.c_in / params.groups) * params.kernel_h * params.kernel_w,
        params.batch * params.c_out * params.output_h * params.output_w,
        params.c_out
    )
);

half_conv_wrapper!(
    depthwise_conv2d_f16, depthwise_conv2d_bf16, depthwise_conv2d_f32, Conv2dParams,
    sizes: |params| (
        params.batch * params.c_in * params.height * params.width,
        params.c_in * params.kernel_h * params.kernel_w,
        params.batch * params.c_out * params.output_h * params.output_w,
        params.c_out
    )
);
