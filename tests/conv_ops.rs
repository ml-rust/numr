//! Integration tests for convolution operations.

use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode, RandomOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn setup() -> (CpuDevice, <CpuRuntime as Runtime>::Client) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (device, client)
}

// =============================================================================
// Conv1d Tests
// =============================================================================

#[test]
fn test_conv1d_moving_average() {
    let (device, client) = setup();

    // Input: [1, 2, 3, 4, 5]
    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);

    // Moving average kernel
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
fn test_conv1d_edge_detection() {
    let (device, client) = setup();

    // Input: constant signal
    let input =
        Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0, 5.0, 5.0, 5.0], &[1, 1, 5], &device);

    // Edge detection kernel: [-1, 0, 1]
    let weight = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 1.0], &[1, 1, 3], &device);

    let output = client
        .conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1)
        .unwrap();

    let data: Vec<f32> = output.to_vec();
    // Constant signal should have zero edges
    for val in data {
        assert!(val.abs() < 1e-5);
    }
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

    // Same padding preserves length
    assert_eq!(output.shape(), &[1, 1, 5]);
}

#[test]
fn test_conv1d_custom_padding() {
    let (device, client) = setup();

    let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device);
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

    // Pad 2 on left, 1 on right
    let output = client
        .conv1d(&input, &weight, None, 1, PaddingMode::conv1d(2, 1), 1, 1)
        .unwrap();

    // Output length = (3 + 2 + 1 - 3) / 1 + 1 = 4
    assert_eq!(output.shape(), &[1, 1, 4]);
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

    // Output: (7-3)/2+1 = 3
    assert_eq!(output.shape(), &[1, 1, 3]);
    let data: Vec<f32> = output.to_vec();
    assert!((data[0] - 6.0).abs() < 1e-5); // 1+2+3
    assert!((data[1] - 12.0).abs() < 1e-5); // 3+4+5
    assert!((data[2] - 18.0).abs() < 1e-5); // 5+6+7
}

#[test]
fn test_conv1d_dilation() {
    let (device, client) = setup();

    // With dilation=2, kernel [1,1,1] looks at positions 0,2,4
    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 2.0, 0.0, 3.0], &[1, 1, 5], &device);
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

    let output = client
        .conv1d(&input, &weight, None, 1, PaddingMode::Valid, 2, 1)
        .unwrap();

    // Effective kernel size = 2*(3-1)+1 = 5, so output length = 1
    assert_eq!(output.shape(), &[1, 1, 1]);
    let data: Vec<f32> = output.to_vec();
    assert!((data[0] - 6.0).abs() < 1e-5); // 1+2+3
}

#[test]
fn test_conv1d_with_bias() {
    let (device, client) = setup();

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[100.0f32], &[1], &device);

    let output = client
        .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 1)
        .unwrap();

    let data: Vec<f32> = output.to_vec();
    assert!((data[0] - 106.0).abs() < 1e-5); // 6 + 100
    assert!((data[1] - 109.0).abs() < 1e-5); // 9 + 100
    assert!((data[2] - 112.0).abs() < 1e-5); // 12 + 100
}

#[test]
fn test_conv1d_multi_channel() {
    let (_device, client) = setup();

    // 2 input channels
    let input = client.randn(&[1, 2, 10], DType::F32).unwrap();
    // 4 output channels
    let weight = client.randn(&[4, 2, 3], DType::F32).unwrap();

    let output = client
        .conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1)
        .unwrap();

    assert_eq!(output.shape(), &[1, 4, 8]);
}

#[test]
fn test_conv1d_batch() {
    let (_device, client) = setup();

    // Batch of 3
    let input = client.randn(&[3, 2, 10], DType::F32).unwrap();
    let weight = client.randn(&[4, 2, 3], DType::F32).unwrap();

    let output = client
        .conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1)
        .unwrap();

    assert_eq!(output.shape(), &[3, 4, 8]);
}

// =============================================================================
// Conv2d Tests
// =============================================================================

#[test]
fn test_conv2d_box_blur() {
    let (device, client) = setup();

    #[rustfmt::skip]
    let input_data = [
        1.0f32, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 1, 3, 3], &device);

    // 2x2 box blur (all ones)
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 1, 2, 2], &device);

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
fn test_conv2d_edge_detection() {
    let (device, client) = setup();

    // Constant image
    let input = Tensor::<CpuRuntime>::full_scalar(&[1, 1, 5, 5], DType::F32, 5.0, &device);

    // Sobel-like edge detector
    #[rustfmt::skip]
    let weight_data = [
        -1.0f32, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];
    let weight = Tensor::<CpuRuntime>::from_slice(&weight_data, &[1, 1, 3, 3], &device);

    let output = client
        .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
        .unwrap();

    // Constant image -> zero edges
    let data: Vec<f32> = output.to_vec();
    for val in data {
        assert!(val.abs() < 1e-5);
    }
}

#[test]
fn test_conv2d_same_padding() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 1, 5, 5], DType::F32).unwrap();
    let weight = client.randn(&[1, 1, 3, 3], DType::F32).unwrap();

    let output = client
        .conv2d(&input, &weight, None, (1, 1), PaddingMode::Same, (1, 1), 1)
        .unwrap();

    // Same padding preserves spatial dimensions
    assert_eq!(output.shape(), &[1, 1, 5, 5]);
}

#[test]
fn test_conv2d_stride() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 1, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[1, 1, 3, 3], DType::F32).unwrap();

    let output = client
        .conv2d(&input, &weight, None, (2, 2), PaddingMode::Valid, (1, 1), 1)
        .unwrap();

    // Output: (8-3)/2+1 = 3
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
}

#[test]
fn test_conv2d_dilation() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 1, 7, 7], DType::F32).unwrap();
    let weight = client.randn(&[1, 1, 3, 3], DType::F32).unwrap();

    // Dilation 2 -> effective kernel is 5x5
    let output = client
        .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (2, 2), 1)
        .unwrap();

    // Output: (7-5)/1+1 = 3
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
}

#[test]
fn test_conv2d_multi_channel_in_out() {
    let (_device, client) = setup();

    // 3 input channels (RGB-like)
    let input = client.randn(&[1, 3, 8, 8], DType::F32).unwrap();
    // 16 output channels
    let weight = client.randn(&[16, 3, 3, 3], DType::F32).unwrap();

    let output = client
        .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
        .unwrap();

    assert_eq!(output.shape(), &[1, 16, 6, 6]);
}

#[test]
fn test_conv2d_grouped() {
    let (_device, client) = setup();

    // 4 input channels, 2 groups -> 2 channels per group
    let input = client.randn(&[1, 4, 6, 6], DType::F32).unwrap();
    // 8 output channels (4 per group), weight needs 2 channels per group
    let weight = client.randn(&[8, 2, 3, 3], DType::F32).unwrap();

    let output = client
        .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 2)
        .unwrap();

    assert_eq!(output.shape(), &[1, 8, 4, 4]);
}

#[test]
fn test_conv2d_with_bias() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 3, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[16, 3, 3, 3], DType::F32).unwrap();
    let bias = client.randn(&[16], DType::F32).unwrap();

    let output = client
        .conv2d(
            &input,
            &weight,
            Some(&bias),
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            1,
        )
        .unwrap();

    assert_eq!(output.shape(), &[1, 16, 6, 6]);
}

#[test]
fn test_conv2d_batch() {
    let (_device, client) = setup();

    // Batch of 4 images
    let input = client.randn(&[4, 3, 32, 32], DType::F32).unwrap();
    let weight = client.randn(&[64, 3, 3, 3], DType::F32).unwrap();

    let output = client
        .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
        .unwrap();

    assert_eq!(output.shape(), &[4, 64, 30, 30]);
}

// =============================================================================
// Depthwise Conv2d Tests
// =============================================================================

#[test]
fn test_depthwise_conv2d_basic() {
    let (device, client) = setup();

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

    // Depthwise: each channel has its own kernel
    let weight = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 1.0, 1.0, 1.0, // channel 0
            2.0, 2.0, 2.0, 2.0, // channel 1
        ],
        &[2, 1, 2, 2],
        &device,
    );

    let output = client
        .depthwise_conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1))
        .unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 2]);
    let data: Vec<f32> = output.to_vec();

    // Channel 0
    assert!((data[0] - 12.0).abs() < 1e-5);
    assert!((data[1] - 16.0).abs() < 1e-5);
    assert!((data[2] - 24.0).abs() < 1e-5);
    assert!((data[3] - 28.0).abs() < 1e-5);

    // Channel 1
    assert!((data[4] - 56.0).abs() < 1e-5); // (9+8+6+5)*2
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

    assert_eq!(output.shape(), &[2, 32, 28, 28]);
}

#[test]
fn test_depthwise_conv2d_stride() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 8, 16, 16], DType::F32).unwrap();
    let weight = client.randn(&[8, 1, 3, 3], DType::F32).unwrap();

    let output = client
        .depthwise_conv2d(&input, &weight, None, (2, 2), PaddingMode::Valid, (1, 1))
        .unwrap();

    // Output: (16-3)/2+1 = 7
    assert_eq!(output.shape(), &[1, 8, 7, 7]);
}

#[test]
fn test_depthwise_conv2d_with_bias() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 16, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[16, 1, 3, 3], DType::F32).unwrap();
    let bias = client.randn(&[16], DType::F32).unwrap();

    let output = client
        .depthwise_conv2d(
            &input,
            &weight,
            Some(&bias),
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
        )
        .unwrap();

    assert_eq!(output.shape(), &[1, 16, 6, 6]);
}

// =============================================================================
// Error Cases
// =============================================================================

#[test]
fn test_conv1d_invalid_input_dim() {
    let (_device, client) = setup();

    // 2D instead of 3D
    let input = client.randn(&[3, 10], DType::F32).unwrap();
    let weight = client.randn(&[16, 3, 3], DType::F32).unwrap();

    let result = client.conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_invalid_input_dim() {
    let (_device, client) = setup();

    // 3D instead of 4D
    let input = client.randn(&[3, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[16, 3, 3, 3], DType::F32).unwrap();

    let result = client.conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_invalid_groups() {
    let (_device, client) = setup();

    // 5 channels not divisible by 2 groups
    let input = client.randn(&[1, 5, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[10, 3, 3, 3], DType::F32).unwrap();

    let result = client.conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 2);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_zero_stride() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 3, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[16, 3, 3, 3], DType::F32).unwrap();

    let result = client.conv2d(
        &input,
        &weight,
        None,
        (0, 1), // zero stride
        PaddingMode::Valid,
        (1, 1),
        1,
    );
    assert!(result.is_err());
}

#[test]
fn test_depthwise_conv2d_invalid_weight_channels() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 4, 8, 8], DType::F32).unwrap();
    // Weight should have shape (4, 1, k, k) but has (4, 2, 3, 3)
    let weight = client.randn(&[4, 2, 3, 3], DType::F32).unwrap();

    let result = client.depthwise_conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1));
    assert!(result.is_err());
}

#[test]
fn test_conv2d_dtype_mismatch() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 3, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[16, 3, 3, 3], DType::F64).unwrap();

    let result = client.conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_bias_length_mismatch() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 3, 8, 8], DType::F32).unwrap();
    let weight = client.randn(&[16, 3, 3, 3], DType::F32).unwrap();
    // Bias length 8 doesn't match c_out=16
    let bias = client.randn(&[8], DType::F32).unwrap();

    let result = client.conv2d(
        &input,
        &weight,
        Some(&bias),
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
    );
    assert!(result.is_err());
}

// =============================================================================
// F64 Tests
// =============================================================================

#[test]
fn test_conv1d_f64() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 2, 10], DType::F64).unwrap();
    let weight = client.randn(&[4, 2, 3], DType::F64).unwrap();

    let output = client
        .conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1)
        .unwrap();

    assert_eq!(output.shape(), &[1, 4, 8]);
    assert_eq!(output.dtype(), DType::F64);
}

#[test]
fn test_conv2d_f64() {
    let (_device, client) = setup();

    let input = client.randn(&[1, 3, 8, 8], DType::F64).unwrap();
    let weight = client.randn(&[16, 3, 3, 3], DType::F64).unwrap();

    let output = client
        .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
        .unwrap();

    assert_eq!(output.shape(), &[1, 16, 6, 6]);
    assert_eq!(output.dtype(), DType::F64);
}
