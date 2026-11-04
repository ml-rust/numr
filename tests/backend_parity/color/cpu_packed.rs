// ColorOps on CPU through packed U32 planes and packed RGB24: the golden,
// round-trip, float dtype, RGB24 and error suites, and bit parity with the
// U8 and U16 form.

use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;

use super::packed::with_packed;
use super::suite_golden::{
    check_decode_golden, check_decode_into_slice, check_encode_golden, check_float_dtypes,
};
use super::suite_packed::{check_packed_ffmpeg_bytes, check_packed_matches_cpu};
use super::suite_packed_errors::{check_packed_rgb24_errors, check_packed_yuv_errors};
use super::suite_packed_pad::{
    check_pad_packed_luma, check_pad_packed_luma_errors, check_padded_luma_decodes,
};
use super::suite_packed_rgb24::{
    check_packed_rgb24, check_packed_rgb24_ties_and_clamps, check_packed_rgb24_views,
};
use super::suite_roundtrip::{check_rgb_yuv_rgb, check_yuv_rgb_yuv};
use crate::common::create_cpu_client;

#[test]
fn test_color_cpu_packed_matches_golden() {
    let (client, device) = create_cpu_client();
    with_packed(|| {
        check_decode_golden::<CpuRuntime, _>(&client, &device);
        check_decode_into_slice::<CpuRuntime, _>(&client, &device);
        check_encode_golden::<CpuRuntime, _>(&client, &device);
        check_float_dtypes::<CpuRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_cpu_packed_round_trips() {
    let (client, device) = create_cpu_client();
    with_packed(|| {
        check_rgb_yuv_rgb::<CpuRuntime, _>(&client, &device);
        check_yuv_rgb_yuv::<CpuRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_cpu_packed_matches_unpacked() {
    let (client, device) = create_cpu_client();
    check_packed_matches_cpu::<CpuRuntime, _>(&client, &device);
    check_packed_ffmpeg_bytes::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_packed_rgb24() {
    let (client, device) = create_cpu_client();
    check_packed_rgb24::<CpuRuntime, _>(&client, &device);
    check_packed_rgb24_views::<CpuRuntime, _>(&client, &device);
    check_packed_rgb24_ties_and_clamps::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_packed_errors() {
    let (client, device) = create_cpu_client();
    check_packed_yuv_errors::<CpuRuntime, _>(&client, &device);
    check_packed_rgb24_errors::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_pad_packed_luma() {
    let (client, device) = create_cpu_client();
    check_pad_packed_luma::<CpuRuntime, _>(&client, &device);
    check_padded_luma_decodes::<CpuRuntime, _>(&client, &device);
    check_pad_packed_luma_errors::<CpuRuntime, _>(&client, &device);
}

/// CPU kernels take the U8 and U16 forms, so generic code need not pack.
#[test]
fn test_color_cpu_does_not_require_packed_planes() {
    assert!(!CpuRuntime::packed_color_planes());
}
