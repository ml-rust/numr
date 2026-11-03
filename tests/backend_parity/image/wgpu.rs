#![cfg(feature = "wgpu")]
// ImageOps on WebGPU against the host reference, for F32, I32 and U32.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{ConvLayout, ImageOps, NearestSize};
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

use super::suite::{check_dtype, check_nhwc_matches_nchw, check_roundtrip, check_strided_input};
use super::suite_into::{
    check_empty_batch, check_errors, check_into_channel_slice, check_into_shared_storage,
    check_nearest_golden, check_repeat_interleave,
};
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;

/// The dtypes WebGPU stores.
const WGPU_DTYPES: [DType; 3] = [DType::F32, DType::I32, DType::U32];

#[test]
fn test_image_ops_wgpu_every_dtype_matches_reference() {
    with_wgpu_backend_or_skip(|client, device| {
        for dtype in WGPU_DTYPES {
            check_dtype::<WgpuRuntime, _>(&client, &device, dtype);
            check_roundtrip::<WgpuRuntime, _>(&client, &device, dtype);
        }
    });
}

#[test]
fn test_image_ops_wgpu_nhwc_strided_and_into_views() {
    with_wgpu_backend_or_skip(|client, device| {
        for dtype in WGPU_DTYPES {
            check_nhwc_matches_nchw::<WgpuRuntime, _>(&client, &device, dtype);
            check_strided_input::<WgpuRuntime, _>(&client, &device, dtype);
            check_into_channel_slice::<WgpuRuntime, _>(&client, &device, dtype);
            check_into_shared_storage::<WgpuRuntime, _>(&client, &device, dtype);
        }
    });
}

#[test]
fn test_upsample_nearest2d_wgpu_golden_tables_and_repeat_interleave() {
    with_wgpu_backend_or_skip(|client, device| {
        check_nearest_golden::<WgpuRuntime, _>(&client, &device);
        check_repeat_interleave::<WgpuRuntime, _>(&client, &device);
    });
}

#[test]
fn test_image_ops_wgpu_errors_and_empty_batch() {
    with_wgpu_backend_or_skip(|client, device| {
        check_errors::<WgpuRuntime, _>(&client, &device);
        check_empty_batch::<WgpuRuntime, _>(&client, &device);
    });
}

#[test]
fn test_image_ops_wgpu_f64_is_unsupported_dtype() {
    with_wgpu_backend_or_skip(|client, device| {
        let x = Tensor::<WgpuRuntime>::from_slice(&[0.0f64; 16], &[1, 4, 2, 2], &device)
            .expect("WGPU F64 tensor creation");
        let results = [
            (
                "pixel_shuffle",
                client.pixel_shuffle(&x, 2, ConvLayout::Nchw),
            ),
            (
                "pixel_unshuffle",
                client.pixel_unshuffle(&x, 2, ConvLayout::Nchw),
            ),
            (
                "upsample_nearest2d",
                client.upsample_nearest2d(&x, NearestSize::Size(4, 4), ConvLayout::Nchw),
            ),
        ];
        for (name, result) in results {
            match result {
                Err(Error::UnsupportedDType { dtype, op }) => {
                    assert_eq!(dtype, DType::F64);
                    assert_eq!(op, name);
                }
                other => panic!("{name}: expected UnsupportedDType, got {other:?}"),
            }
        }
    });
}
