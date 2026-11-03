// ColorOps on CPU against the numpy references, round trips, ffmpeg and
// argument errors.

use numr::runtime::cpu::CpuRuntime;

use super::suite_errors::{check_rgb24_errors, check_yuv_errors};
use super::suite_golden::{
    check_decode_golden, check_decode_into_slice, check_encode_golden, check_float_dtypes,
};
use super::suite_rgb24::{
    check_rgb24_dtypes, check_rgb24_round_trip, check_rgb24_ties_and_clamps, check_rgb24_views,
};
use super::suite_roundtrip::{check_ffmpeg_psnr, check_rgb_yuv_rgb, check_yuv_rgb_yuv};
use crate::common::create_cpu_client;

#[test]
fn test_color_cpu_decode_matches_golden() {
    let (client, device) = create_cpu_client();
    check_decode_golden::<CpuRuntime, _>(&client, &device);
    check_decode_into_slice::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_encode_matches_golden() {
    let (client, device) = create_cpu_client();
    check_encode_golden::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_float_dtypes() {
    let (client, device) = create_cpu_client();
    check_float_dtypes::<CpuRuntime, _>(&client, &device);
    check_rgb24_dtypes::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_round_trips() {
    let (client, device) = create_cpu_client();
    check_rgb_yuv_rgb::<CpuRuntime, _>(&client, &device);
    check_yuv_rgb_yuv::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_nv12_matches_ffmpeg_psnr() {
    let (client, device) = create_cpu_client();
    check_ffmpeg_psnr::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_rgb24() {
    let (client, device) = create_cpu_client();
    check_rgb24_round_trip::<CpuRuntime, _>(&client, &device);
    check_rgb24_ties_and_clamps::<CpuRuntime, _>(&client, &device);
    check_rgb24_views::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_color_cpu_errors() {
    let (client, device) = create_cpu_client();
    check_yuv_errors::<CpuRuntime, _>(&client, &device);
    check_rgb24_errors::<CpuRuntime, _>(&client, &device);
}
