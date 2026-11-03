// ImageOps::resize on CPU against PyTorch CPU references.

use numr::runtime::cpu::CpuRuntime;

#[cfg(feature = "f16")]
use super::resize_suite::check_resize_half;
use super::resize_suite::{check_resize_f64, check_resize_golden, check_resize_srvgg};
use super::resize_suite_views::{
    check_resize_equal_sizes, check_resize_errors, check_resize_into_channel_slice,
    check_resize_layouts_agree, check_resize_strided_input,
};
use crate::common::create_cpu_client;

#[test]
fn test_resize_cpu_f32_matches_torch() {
    let (client, device) = create_cpu_client();
    check_resize_golden::<CpuRuntime, _>(&client, &device, 1.0);
}

#[test]
fn test_resize_cpu_srvgg_halving_matches_torch() {
    let (client, device) = create_cpu_client();
    check_resize_srvgg::<CpuRuntime, _>(&client, &device, 1.0);
}

#[test]
fn test_resize_cpu_f64_matches_torch() {
    let (client, device) = create_cpu_client();
    check_resize_f64::<CpuRuntime, _>(&client, &device);
}

#[cfg(feature = "f16")]
#[test]
fn test_resize_cpu_half_rounds_once() {
    let (client, device) = create_cpu_client();
    check_resize_half::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_resize_cpu_views_and_into() {
    let (client, device) = create_cpu_client();
    check_resize_strided_input::<CpuRuntime, _>(&client, &device);
    check_resize_into_channel_slice::<CpuRuntime, _>(&client, &device);
    check_resize_layouts_agree::<CpuRuntime, _>(&client, &device);
}

#[test]
fn test_resize_cpu_equal_sizes_and_errors() {
    let (client, device) = create_cpu_client();
    check_resize_equal_sizes::<CpuRuntime, _>(&client, &device);
    check_resize_errors::<CpuRuntime, _>(&client, &device);
}
