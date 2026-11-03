#![cfg(feature = "wgpu")]
// ImageOps::resize on WebGPU, F32, against PyTorch CPU references.
//
// WGSL division is not correctly rounded, and an implementation can fuse or
// split a multiply-add, so WebGPU gets twice the CPU bound. The source-index
// term of that bound covers a fused or split index for both plain filters.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{ConvLayout, ImageOps, ResizeConfig, ResizeFilter};
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

use super::resize_suite::{check_resize_golden, check_resize_srvgg};
use super::resize_suite_views::{
    check_resize_equal_sizes, check_resize_errors, check_resize_into_channel_slice,
    check_resize_layouts_agree, check_resize_strided_input,
};
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;

#[test]
fn test_resize_wgpu_f32_matches_torch() {
    with_wgpu_backend_or_skip(|client, device| {
        check_resize_golden::<WgpuRuntime, _>(&client, &device, 2.0);
        check_resize_srvgg::<WgpuRuntime, _>(&client, &device, 2.0);
    });
}

#[test]
fn test_resize_wgpu_views_equal_sizes_and_errors() {
    with_wgpu_backend_or_skip(|client, device| {
        check_resize_strided_input::<WgpuRuntime, _>(&client, &device);
        check_resize_into_channel_slice::<WgpuRuntime, _>(&client, &device);
        check_resize_layouts_agree::<WgpuRuntime, _>(&client, &device);
        check_resize_equal_sizes::<WgpuRuntime, _>(&client, &device);
        check_resize_errors::<WgpuRuntime, _>(&client, &device);
    });
}

#[test]
fn test_resize_wgpu_f64_is_unsupported_dtype() {
    with_wgpu_backend_or_skip(|client, device| {
        let x = Tensor::<WgpuRuntime>::from_slice(&[0.0f64; 16], &[1, 1, 4, 4], &device)
            .expect("WGPU F64 tensor creation");
        let cfg = ResizeConfig::new((2, 2), ResizeFilter::Bicubic);
        match client.resize(&x, &cfg, ConvLayout::Nchw) {
            Err(Error::UnsupportedDType { dtype, op }) => {
                assert_eq!(dtype, DType::F64);
                assert_eq!(op, "resize");
            }
            other => panic!("resize: expected UnsupportedDType, got {other:?}"),
        }
    });
}
