#![cfg(feature = "wgpu")]
// ColorOps on WebGPU through packed U32 planes and packed RGB24: the golden,
// round-trip, RGB24 and error suites, bit parity with CPU on the U8 and U16
// form, and the refusals of the forms WebGPU does not address.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, YuvPlanes, YuvSpec};
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

use super::golden::Format;
use super::packed::{upload_packed, with_packed};
use super::suite_golden::{check_decode_golden, check_decode_into_slice, check_encode_golden};
use super::suite_packed::{check_packed_ffmpeg_bytes, check_packed_matches_cpu};
use super::suite_packed_errors::{check_packed_rgb24_errors, check_packed_yuv_errors};
use super::suite_packed_rgb24::{
    check_packed_rgb24, check_packed_rgb24_ties_and_clamps, check_packed_rgb24_views,
};
use super::suite_roundtrip::{check_rgb_yuv_rgb, check_yuv_rgb_yuv};
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;

#[test]
fn test_color_wgpu_matches_golden() {
    with_wgpu_backend_or_skip(|client, device| {
        with_packed(|| {
            check_decode_golden::<WgpuRuntime, _>(&client, &device);
            check_decode_into_slice::<WgpuRuntime, _>(&client, &device);
            check_encode_golden::<WgpuRuntime, _>(&client, &device);
        });
    });
}

#[test]
fn test_color_wgpu_round_trips() {
    with_wgpu_backend_or_skip(|client, device| {
        with_packed(|| {
            check_rgb_yuv_rgb::<WgpuRuntime, _>(&client, &device);
            check_yuv_rgb_yuv::<WgpuRuntime, _>(&client, &device);
        });
    });
}

#[test]
fn test_color_wgpu_matches_cpu_bit_for_bit() {
    with_wgpu_backend_or_skip(|client, device| {
        check_packed_matches_cpu::<WgpuRuntime, _>(&client, &device);
        check_packed_ffmpeg_bytes::<WgpuRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_wgpu_rgb24() {
    with_wgpu_backend_or_skip(|client, device| {
        check_packed_rgb24::<WgpuRuntime, _>(&client, &device);
        check_packed_rgb24_views::<WgpuRuntime, _>(&client, &device);
        check_packed_rgb24_ties_and_clamps::<WgpuRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_wgpu_errors() {
    with_wgpu_backend_or_skip(|client, device| {
        check_packed_yuv_errors::<WgpuRuntime, _>(&client, &device);
        check_packed_rgb24_errors::<WgpuRuntime, _>(&client, &device);
    });
}

fn limitation<T: std::fmt::Debug>(result: numr::error::Result<T>, text: &str) {
    match result {
        Err(Error::BackendLimitation { reason, .. }) => {
            assert!(reason.contains(text), "reason {reason:?} lacks {text:?}");
        }
        other => panic!("expected BackendLimitation ({text}), got {other:?}"),
    }
}

/// U8 and U16 planes and U8 RGB24 return `BackendLimitation` that names the
/// packed form, and float dtypes other than F32 return `UnsupportedDType`.
#[test]
fn test_color_wgpu_refuses_unpacked_forms_and_wide_floats() {
    with_wgpu_backend_or_skip(|client, device| {
        let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
        let zeros = |shape: &[usize], dtype| {
            Tensor::<WgpuRuntime>::zeros(shape, dtype, &device).expect("zeros")
        };
        let (y, uv) = (zeros(&[4, 6], DType::U8), zeros(&[2, 6], DType::U8));
        let planes = YuvPlanes::Nv12 { y: &y, uv: &uv };
        limitation(
            client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw),
            "Nv12Packed",
        );
        let rgb = zeros(&[1, 3, 4, 6], DType::F32);
        limitation(
            client.rgb_to_yuv_into(&rgb, planes, &spec, ConvLayout::Nchw),
            "Nv12Packed",
        );
        let (y16, uv16) = (zeros(&[4, 6], DType::U16), zeros(&[2, 6], DType::U16));
        let planes = YuvPlanes::P010 { y: &y16, uv: &uv16 };
        limitation(
            client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw),
            "P010Packed",
        );
        let rgb24 = zeros(&[2, 2, 3], DType::U8);
        limitation(
            client.rgb24_to_float(&rgb24, DType::F32, ConvLayout::Nchw),
            "packed_rgb24_to_float",
        );
        limitation(
            client.float_to_rgb24(&zeros(&[1, 3, 2, 2], DType::F32), ConvLayout::Nchw),
            "float_to_packed_rgb24",
        );

        let zero_planes: Vec<Vec<u16>> = Format::Nv12
            .plane_shapes(4, 6)
            .iter()
            .map(|[r, c]| vec![0; r * c])
            .collect();
        let set = upload_packed::<WgpuRuntime>(Format::Nv12, 4, 6, &zero_planes, false, &device);
        assert!(matches!(
            client.yuv_to_rgb(set.yuv(), &spec, DType::F64, ConvLayout::Nchw),
            Err(Error::UnsupportedDType { .. })
        ));
        let wide = zeros(&[1, 3, 4, 6], DType::F64);
        assert!(matches!(
            client.rgb_to_yuv_into(&wide, set.yuv(), &spec, ConvLayout::Nchw),
            Err(Error::UnsupportedDType { .. })
        ));
        assert!(matches!(
            client.float_to_packed_rgb24(&wide, ConvLayout::Nchw),
            Err(Error::UnsupportedDType { .. })
        ));
    });
}
