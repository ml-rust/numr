// ImageOps on CPU against the host reference, for every dtype.

use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;

use super::reference::ALL_DTYPES;
use super::suite::{check_dtype, check_nhwc_matches_nchw, check_roundtrip, check_strided_input};
use super::suite_into::{
    check_empty_batch, check_errors, check_into_channel_slice, check_into_shared_storage,
    check_nearest_golden, check_repeat_interleave,
};
use crate::common::create_cpu_client;

#[test]
fn test_image_ops_cpu_every_dtype_matches_reference() {
    let (client, device) = create_cpu_client();
    for dtype in ALL_DTYPES {
        check_dtype::<CpuRuntime, _>(&client, &device, dtype);
        check_roundtrip::<CpuRuntime, _>(&client, &device, dtype);
    }
}

#[test]
fn test_image_ops_cpu_nhwc_matches_permuted_nchw() {
    let (client, device) = create_cpu_client();
    for dtype in [DType::F32, DType::U8, DType::BF16, DType::Complex128] {
        check_nhwc_matches_nchw::<CpuRuntime, _>(&client, &device, dtype);
    }
}

#[test]
fn test_image_ops_cpu_strided_input() {
    let (client, device) = create_cpu_client();
    for dtype in [DType::F32, DType::U8, DType::F16] {
        check_strided_input::<CpuRuntime, _>(&client, &device, dtype);
    }
}

#[test]
fn test_image_ops_cpu_into_views() {
    let (client, device) = create_cpu_client();
    for dtype in [DType::F32, DType::U8, DType::F64] {
        check_into_channel_slice::<CpuRuntime, _>(&client, &device, dtype);
        check_into_shared_storage::<CpuRuntime, _>(&client, &device, dtype);
    }
}

#[test]
fn test_upsample_nearest2d_cpu_golden_tables_and_repeat_interleave() {
    let (client, device) = create_cpu_client();
    check_nearest_golden::<CpuRuntime, _>(&client, &device);
    check_repeat_interleave::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_image_ops_cpu_errors_and_empty_batch() {
    let (client, device) = create_cpu_client();
    check_errors::<CpuRuntime, _>(&client, &device);
    check_empty_batch::<CpuRuntime, _>(&client, &device);
}

/// A batch large enough to cross the CPU parallel threshold matches the
/// reference, so the pooled path writes the same elements.
#[test]
fn test_image_ops_cpu_parallel_batch_matches_reference() {
    use super::suite::{Api, check_case};
    use numr::ops::{ConvLayout, NearestSize};
    let (client, device) = create_cpu_client();
    for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
        check_case::<CpuRuntime, _>(
            &client,
            &device,
            DType::F32,
            [2, 64, 32, 32],
            Api::Shuffle(2),
            layout,
        );
        check_case::<CpuRuntime, _>(
            &client,
            &device,
            DType::U16,
            [2, 16, 64, 64],
            Api::Unshuffle(4),
            layout,
        );
        let size = Api::Nearest(NearestSize::Scale(2.5, 1.5));
        check_case::<CpuRuntime, _>(&client, &device, DType::F64, [2, 8, 40, 48], size, layout);
    }
}
