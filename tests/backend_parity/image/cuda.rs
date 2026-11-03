#![cfg(feature = "cuda")]
// ImageOps on CUDA against the host reference, for every dtype, and against
// CPU on a batch that spans many blocks.

use numr::dtype::DType;
use numr::ops::{ConvLayout, NearestSize};
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;

use super::reference::{ALL_DTYPES, assert_bytes_eq, pattern};
use super::suite::{
    Api, bytes_of, call, check_dtype, check_nhwc_matches_nchw, check_roundtrip,
    check_strided_input, make,
};
use super::suite_into::{
    check_empty_batch, check_errors, check_into_channel_slice, check_into_shared_storage,
    check_nearest_golden, check_repeat_interleave,
};
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::create_cpu_client;

#[test]
fn test_image_ops_cuda_every_dtype_matches_reference() {
    with_cuda_backend(|client, device| {
        for dtype in ALL_DTYPES {
            check_dtype::<CudaRuntime, _>(&client, &device, dtype);
            check_roundtrip::<CudaRuntime, _>(&client, &device, dtype);
        }
    });
}

#[test]
fn test_image_ops_cuda_nhwc_matches_permuted_nchw() {
    with_cuda_backend(|client, device| {
        for dtype in [DType::F32, DType::U8, DType::BF16, DType::Complex128] {
            check_nhwc_matches_nchw::<CudaRuntime, _>(&client, &device, dtype);
        }
    });
}

#[test]
fn test_image_ops_cuda_strided_input() {
    with_cuda_backend(|client, device| {
        for dtype in [DType::F32, DType::U8, DType::F16] {
            check_strided_input::<CudaRuntime, _>(&client, &device, dtype);
        }
    });
}

#[test]
fn test_image_ops_cuda_into_views() {
    with_cuda_backend(|client, device| {
        for dtype in [DType::F32, DType::U8, DType::F64] {
            check_into_channel_slice::<CudaRuntime, _>(&client, &device, dtype);
            check_into_shared_storage::<CudaRuntime, _>(&client, &device, dtype);
        }
    });
}

#[test]
fn test_upsample_nearest2d_cuda_golden_tables_and_repeat_interleave() {
    with_cuda_backend(|client, device| {
        check_nearest_golden::<CudaRuntime, _>(&client, &device);
        check_repeat_interleave::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_image_ops_cuda_errors_and_empty_batch() {
    with_cuda_backend(|client, device| {
        check_errors::<CudaRuntime, _>(&client, &device);
        check_empty_batch::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_image_ops_cuda_matches_cpu_on_a_large_batch() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cases = [
            ([2, 64, 48, 40], Api::Shuffle(2)),
            ([2, 4, 96, 80], Api::Unshuffle(4)),
            ([2, 16, 37, 29], Api::Nearest(NearestSize::Scale(2.7, 1.9))),
            ([2, 16, 64, 64], Api::Nearest(NearestSize::Size(45, 101))),
        ];
        for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
            for (shape, api) in cases {
                let dtype = DType::BF16;
                let input = pattern(shape.iter().product(), dtype);
                let x_cpu = make::<CpuRuntime>(&input, &shape, dtype, &cpu_device);
                let x_cuda = make::<CudaRuntime>(&input, &shape, dtype, &cuda_device);
                let want = bytes_of(
                    &call::<CpuRuntime, _>(&cpu_client, &x_cpu, api, layout).expect("CPU"),
                );
                let got = bytes_of(
                    &call::<CudaRuntime, _>(&cuda_client, &x_cuda, api, layout).expect("CUDA"),
                );
                let label = format!("{api:?} {layout:?} CUDA vs CPU");
                assert_bytes_eq(&got, &want, dtype.size_in_bytes(), &label);
            }
        }
    });
}
