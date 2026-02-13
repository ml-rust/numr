//! CPU implementation of convolution operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::conv_common::{validate_conv1d, validate_conv2d, validate_depthwise_conv2d};
use crate::ops::{ConvOps, PaddingMode};
#[cfg(target_arch = "x86_64")]
use crate::runtime::cpu::kernels::simd::conv as simd_conv;
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::ensure_contiguous, kernels};
use crate::tensor::Tensor;

/// Dispatch to convolution kernel for float types only
macro_rules! dispatch_float_dtype {
    ($dtype:expr, $T:ident => $body:block, $op:expr) => {
        match $dtype {
            DType::F32 => {
                type $T = f32;
                $body
            }
            DType::F64 => {
                type $T = f64;
                $body
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                type $T = half::f16;
                $body
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => {
                type $T = crate::dtype::FP8E4M3;
                $body
            }
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => {
                type $T = crate::dtype::FP8E5M2;
                $body
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $op,
                })
            }
        }
    };
}

/// Dispatch to SIMD kernels for F32/F64 on x86_64, scalar for other types/platforms.
///
/// This macro eliminates duplication across conv1d, conv2d, and depthwise_conv2d dispatch blocks.
macro_rules! dispatch_conv {
    ($dtype:expr, $conv_name:ident, $input_ptr:expr, $weight_ptr:expr, $bias_ptr:expr, $output_ptr:expr, $params:expr) => {
        paste::paste! {
            match $dtype {
                #[cfg(target_arch = "x86_64")]
                DType::F32 => unsafe {
                    simd_conv::[<$conv_name _f32>](
                        $input_ptr as *const f32,
                        $weight_ptr as *const f32,
                        $bias_ptr.map(|p| p as *const f32),
                        $output_ptr as *mut f32,
                        $params,
                    );
                },
                #[cfg(target_arch = "x86_64")]
                DType::F64 => unsafe {
                    simd_conv::[<$conv_name _f64>](
                        $input_ptr as *const f64,
                        $weight_ptr as *const f64,
                        $bias_ptr.map(|p| p as *const f64),
                        $output_ptr as *mut f64,
                        $params,
                    );
                },
                _ => {
                    dispatch_float_dtype!($dtype, T => {
                        unsafe {
                            kernels::[<$conv_name _kernel>]::<T>(
                                $input_ptr as *const T,
                                $weight_ptr as *const T,
                                $bias_ptr.map(|p| p as *const T),
                                $output_ptr as *mut T,
                                $params,
                            );
                        }
                    }, stringify!($conv_name));
                }
            }
        }
    };
}

impl ConvOps<CpuRuntime> for CpuClient {
    fn conv1d(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

        // Validate all inputs and get computed parameters
        let params = validate_conv1d(
            input.shape(),
            weight.shape(),
            bias.map(|b| b.shape()),
            stride,
            padding,
            dilation,
            groups,
            dtype,
            weight.dtype(),
            bias.map(|b| b.dtype()),
        )?;

        // Handle empty output
        if params.output_length == 0 || params.batch == 0 {
            return Ok(Tensor::<CpuRuntime>::empty(
                &[params.batch, params.c_out, params.output_length],
                dtype,
                &self.device,
            ));
        }

        // Ensure contiguous
        let input = ensure_contiguous(input);
        let weight = ensure_contiguous(weight);
        let bias = bias.map(ensure_contiguous);

        // Allocate output
        let output = Tensor::<CpuRuntime>::empty(
            &[params.batch, params.c_out, params.output_length],
            dtype,
            &self.device,
        );

        let input_ptr = input.storage().ptr();
        let weight_ptr = weight.storage().ptr();
        let bias_ptr = bias.as_ref().map(|b| b.storage().ptr());
        let output_ptr = output.storage().ptr();

        dispatch_conv!(
            dtype, conv1d, input_ptr, weight_ptr, bias_ptr, output_ptr, params
        );

        Ok(output)
    }

    fn conv2d(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

        // Validate all inputs and get computed parameters
        let params = validate_conv2d(
            input.shape(),
            weight.shape(),
            bias.map(|b| b.shape()),
            stride,
            padding,
            dilation,
            groups,
            dtype,
            weight.dtype(),
            bias.map(|b| b.dtype()),
        )?;

        // Handle empty output
        if params.output_h == 0 || params.output_w == 0 || params.batch == 0 {
            return Ok(Tensor::<CpuRuntime>::empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                &self.device,
            ));
        }

        // Ensure contiguous
        let input = ensure_contiguous(input);
        let weight = ensure_contiguous(weight);
        let bias = bias.map(ensure_contiguous);

        // Allocate output
        let output = Tensor::<CpuRuntime>::empty(
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
            &self.device,
        );

        let input_ptr = input.storage().ptr();
        let weight_ptr = weight.storage().ptr();
        let bias_ptr = bias.as_ref().map(|b| b.storage().ptr());
        let output_ptr = output.storage().ptr();

        dispatch_conv!(
            dtype, conv2d, input_ptr, weight_ptr, bias_ptr, output_ptr, params
        );

        Ok(output)
    }

    fn depthwise_conv2d(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = input.dtype();

        // Validate all inputs and get computed parameters
        let params = validate_depthwise_conv2d(
            input.shape(),
            weight.shape(),
            bias.map(|b| b.shape()),
            stride,
            padding,
            dilation,
            dtype,
            weight.dtype(),
            bias.map(|b| b.dtype()),
        )?;

        // Handle empty output
        if params.output_h == 0 || params.output_w == 0 || params.batch == 0 {
            return Ok(Tensor::<CpuRuntime>::empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                &self.device,
            ));
        }

        // Ensure contiguous
        let input = ensure_contiguous(input);
        let weight = ensure_contiguous(weight);
        let bias = bias.map(ensure_contiguous);

        // Allocate output
        let output = Tensor::<CpuRuntime>::empty(
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
            &self.device,
        );

        let input_ptr = input.storage().ptr();
        let weight_ptr = weight.storage().ptr();
        let bias_ptr = bias.as_ref().map(|b| b.storage().ptr());
        let output_ptr = output.storage().ptr();

        dispatch_conv!(
            dtype,
            depthwise_conv2d,
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            params
        );

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::RandomOps;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_conv1d_basic() {
        let (device, client) = setup();

        // Input: (1, 1, 5) = [1, 2, 3, 4, 5]
        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);

        // Weight: (1, 1, 3) = [1, 1, 1] (moving average kernel)
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

        let output = client
            .conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1)
            .unwrap();

        assert_eq!(output.shape(), &[1, 1, 3]);
        let data: Vec<f32> = output.to_vec();
        assert!((data[0] - 6.0).abs() < 1e-5); // 1+2+3
        assert!((data[1] - 9.0).abs() < 1e-5); // 2+3+4
        assert!((data[2] - 12.0).abs() < 1e-5); // 3+4+5
    }

    #[test]
    fn test_conv1d_same_padding() {
        let (device, client) = setup();

        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

        let output = client
            .conv1d(&input, &weight, None, 1, PaddingMode::Same, 1, 1)
            .unwrap();

        // With same padding and stride 1, output length == input length
        assert_eq!(output.shape(), &[1, 1, 5]);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let (device, client) = setup();

        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device);

        let output = client
            .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 1)
            .unwrap();

        let data: Vec<f32> = output.to_vec();
        assert!((data[0] - 16.0).abs() < 1e-5); // 6 + 10
        assert!((data[1] - 19.0).abs() < 1e-5); // 9 + 10
        assert!((data[2] - 22.0).abs() < 1e-5); // 12 + 10
    }

    #[test]
    fn test_conv1d_stride() {
        let (device, client) = setup();

        let input = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[1, 1, 7],
            &device,
        );
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

        let output = client
            .conv1d(&input, &weight, None, 2, PaddingMode::Valid, 1, 1)
            .unwrap();

        // Output length = (7 - 3) / 2 + 1 = 3
        assert_eq!(output.shape(), &[1, 1, 3]);
    }

    #[test]
    fn test_conv2d_basic() {
        let (device, client) = setup();

        #[rustfmt::skip]
        let input_data = [
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 1, 3, 3], &device);

        // 2x2 kernel of all ones
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &device);

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        let data: Vec<f32> = output.to_vec();
        assert!((data[0] - 12.0).abs() < 1e-5); // 1+2+4+5
        assert!((data[1] - 16.0).abs() < 1e-5); // 2+3+5+6
        assert!((data[2] - 24.0).abs() < 1e-5); // 4+5+7+8
        assert!((data[3] - 28.0).abs() < 1e-5); // 5+6+8+9
    }

    #[test]
    fn test_conv2d_same_padding() {
        let (device, client) = setup();

        #[rustfmt::skip]
        let input_data = [
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 1, 3, 3], &device);
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &device);

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Same, (1, 1), 1)
            .unwrap();

        // With same padding and stride 1, output size == input size
        assert_eq!(output.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_multi_channel() {
        let (_device, client) = setup();

        // 2 input channels, 4x4 images
        let input = client.randn(&[1, 2, 4, 4], DType::F32).unwrap();

        // 3 output channels, reading from 2 input channels, 3x3 kernels
        let weight = client.randn(&[3, 2, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        // Output: (1, 3, 2, 2) -> (4-3)/1+1 = 2
        assert_eq!(output.shape(), &[1, 3, 2, 2]);
    }

    #[test]
    fn test_conv2d_grouped() {
        let (_device, client) = setup();

        // 4 input channels, split into 2 groups
        let input = client.randn(&[1, 4, 4, 4], DType::F32).unwrap();

        // 6 output channels (3 per group), 2 input channels per group
        let weight = client.randn(&[6, 2, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(
                &input,
                &weight,
                None,
                (1, 1),
                PaddingMode::Valid,
                (1, 1),
                2, // 2 groups
            )
            .unwrap();

        assert_eq!(output.shape(), &[1, 6, 2, 2]);
    }

    #[test]
    fn test_depthwise_conv2d() {
        let (device, client) = setup();

        // 2 channels, 3x3 image
        #[rustfmt::skip]
        let input_data = [
            // Channel 0
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            // Channel 1
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ];
        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 2, 3, 3], &device);

        // Depthwise: 2 output channels, 1 input per channel, 2x2 kernel
        let weight = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 1.0, 1.0, 1.0, // channel 0: all 1s
                2.0, 2.0, 2.0, 2.0, // channel 1: all 2s
            ],
            &[2, 1, 2, 2],
            &device,
        );

        let output = client
            .depthwise_conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1))
            .unwrap();

        assert_eq!(output.shape(), &[1, 2, 2, 2]);
        let data: Vec<f32> = output.to_vec();

        // Channel 0: 1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
        assert!((data[0] - 12.0).abs() < 1e-5);
        assert!((data[1] - 16.0).abs() < 1e-5);
        assert!((data[2] - 24.0).abs() < 1e-5);
        assert!((data[3] - 28.0).abs() < 1e-5);

        // Channel 1: (9+8+6+5)*2=56, (8+7+5+4)*2=48, (6+5+3+2)*2=32, (5+4+2+1)*2=24
        assert!((data[4] - 56.0).abs() < 1e-5);
        assert!((data[5] - 48.0).abs() < 1e-5);
        assert!((data[6] - 32.0).abs() < 1e-5);
        assert!((data[7] - 24.0).abs() < 1e-5);
    }

    #[test]
    fn test_depthwise_conv2d_same_padding() {
        let (_device, client) = setup();

        let input = client.randn(&[2, 32, 28, 28], DType::F32).unwrap();
        let weight = client.randn(&[32, 1, 3, 3], DType::F32).unwrap();

        let output = client
            .depthwise_conv2d(&input, &weight, None, (1, 1), PaddingMode::Same, (1, 1))
            .unwrap();

        // Same padding preserves spatial dimensions
        assert_eq!(output.shape(), &[2, 32, 28, 28]);
    }

    #[test]
    fn test_conv2d_dilation() {
        let (_device, client) = setup();

        // 5x5 input
        let input = client.randn(&[1, 1, 5, 5], DType::F32).unwrap();

        // 3x3 kernel with dilation 2 -> effective kernel size is 5x5
        let weight = client.randn(&[1, 1, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(
                &input,
                &weight,
                None,
                (1, 1),
                PaddingMode::Valid,
                (2, 2), // dilation 2
                1,
            )
            .unwrap();

        // Output: (5 - 2*(3-1) - 1) / 1 + 1 = 1
        assert_eq!(output.shape(), &[1, 1, 1, 1]);
    }

    #[test]
    fn test_conv1d_invalid_dimensions() {
        let (_device, client) = setup();

        // 2D tensor instead of 3D
        let input = client.randn(&[3, 10], DType::F32).unwrap();
        let weight = client.randn(&[16, 3, 3], DType::F32).unwrap();

        let result = client.conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_invalid_groups() {
        let (_device, client) = setup();

        // 5 channels not divisible by 2 groups
        let input = client.randn(&[1, 5, 8, 8], DType::F32).unwrap();
        let weight = client.randn(&[10, 2, 3, 3], DType::F32).unwrap();

        let result = client.conv2d(
            &input,
            &weight,
            None,
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            2, // 5 not divisible by 2
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_batch() {
        let (_device, client) = setup();

        // Batch of 4 images
        let input = client.randn(&[4, 3, 8, 8], DType::F32).unwrap();
        let weight = client.randn(&[16, 3, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        assert_eq!(output.shape(), &[4, 16, 6, 6]);
    }

    #[test]
    fn test_conv2d_f64() {
        let (_device, client) = setup();

        let input = client.randn(&[1, 1, 4, 4], DType::F64).unwrap();
        let weight = client.randn(&[1, 1, 2, 2], DType::F64).unwrap();

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        assert_eq!(output.shape(), &[1, 1, 3, 3]);
        assert_eq!(output.dtype(), DType::F64);
    }
}
