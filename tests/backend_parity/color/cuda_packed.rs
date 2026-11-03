#![cfg(feature = "cuda")]
// ColorOps on CUDA through packed U32 planes and packed RGB24: the golden,
// round-trip, float dtype, RGB24 and error suites, and bit parity with CPU
// on the U8 and U16 form.

use numr::runtime::cuda::CudaRuntime;

use super::packed::with_packed;
use super::suite_golden::{
    check_decode_golden, check_decode_into_slice, check_encode_golden, check_float_dtypes,
};
use super::suite_packed::{check_packed_ffmpeg_bytes, check_packed_matches_cpu};
use super::suite_packed_errors::{check_packed_rgb24_errors, check_packed_yuv_errors};
use super::suite_packed_rgb24::{
    check_packed_rgb24, check_packed_rgb24_ties_and_clamps, check_packed_rgb24_views,
};
use super::suite_roundtrip::{check_rgb_yuv_rgb, check_yuv_rgb_yuv};
use crate::backend_parity::helpers::with_cuda_backend;

#[test]
fn test_color_cuda_packed_matches_golden_and_round_trips() {
    with_cuda_backend(|client, device| {
        with_packed(|| {
            check_decode_golden::<CudaRuntime, _>(&client, &device);
            check_decode_into_slice::<CudaRuntime, _>(&client, &device);
            check_encode_golden::<CudaRuntime, _>(&client, &device);
            check_float_dtypes::<CudaRuntime, _>(&client, &device);
            check_rgb_yuv_rgb::<CudaRuntime, _>(&client, &device);
            check_yuv_rgb_yuv::<CudaRuntime, _>(&client, &device);
        });
    });
}

#[test]
fn test_color_cuda_packed_matches_cpu() {
    with_cuda_backend(|client, device| {
        check_packed_matches_cpu::<CudaRuntime, _>(&client, &device);
        check_packed_ffmpeg_bytes::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_cuda_packed_rgb24_and_errors() {
    with_cuda_backend(|client, device| {
        check_packed_rgb24::<CudaRuntime, _>(&client, &device);
        check_packed_rgb24_views::<CudaRuntime, _>(&client, &device);
        check_packed_rgb24_ties_and_clamps::<CudaRuntime, _>(&client, &device);
        check_packed_yuv_errors::<CudaRuntime, _>(&client, &device);
        check_packed_rgb24_errors::<CudaRuntime, _>(&client, &device);
    });
}
