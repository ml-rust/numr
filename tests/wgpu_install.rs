//! `WgpuClient::install`: adopting a device the caller opened.
//!
//! The client cache is process-wide and an install has no undo, so each test
//! takes its own index far past any adapter. Every test prints a notice and
//! returns when no adapter is available.

#![cfg(feature = "wgpu")]

use std::sync::Arc;

use numr::dtype::DType;
use numr::ops::{BinaryOps, ColorOps, PackedPlane};
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuError, WgpuRuntime};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

const INSTALLED_INDEX: usize = 1 << 20;
const REFUSED_SECOND_INDEX: usize = (1 << 20) + 1;
const BELOW_FLOOR_INDEX: usize = (1 << 20) + 2;

fn first_adapter() -> Option<wgpu::Adapter> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .next();
    if adapter.is_none() {
        println!("skipping: no WGPU adapter available");
    }
    adapter
}

/// Open a device the way a caller outside numr does.
fn open(adapter: &wgpu::Adapter, limits: wgpu::Limits) -> Option<(wgpu::Device, wgpu::Queue)> {
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("install test device"),
        required_limits: limits,
        ..Default::default()
    };
    match pollster::block_on(adapter.request_device(&descriptor)) {
        Ok(opened) => Some(opened),
        Err(e) => {
            println!("skipping: the adapter refused the device: {e}");
            None
        }
    }
}

fn assert_runs_a_binary_op(client: &WgpuClient, index: usize) {
    let device = WgpuDevice::new(index);
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).expect("a");
    let b = Tensor::<WgpuRuntime>::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device).expect("b");
    let c = client.sub(&b, &a).expect("sub");
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, vec![3.0, 3.0, 3.0]);
}

/// Ops, allocation and later client lookups all run on the installed device.
/// `WgpuDevice::new(index)` carries no device: allocation finds it in the
/// cache, so a wrong entry fails here with a buffer from another device.
#[test]
fn test_install_runs_ops_and_serves_later_lookups() {
    let Some(adapter) = first_adapter() else {
        return;
    };
    let Some((device, queue)) = open(&adapter, adapter.limits()) else {
        return;
    };
    let limits = device.limits();
    let client = WgpuClient::install(INSTALLED_INDEX, &adapter, device, queue).expect("install");

    assert_eq!(client.device().id(), INSTALLED_INDEX);
    assert_eq!(client.device().adapter_name(), adapter.get_info().name);
    assert_eq!(client.device().backend(), Some(adapter.get_info().backend));
    assert_eq!(
        client.device().max_storage_buffer_size(),
        limits.max_storage_buffer_binding_size,
    );

    let index_device = WgpuDevice::new(INSTALLED_INDEX);
    let looked_up = WgpuClient::new(index_device.clone()).expect("lookup");
    assert!(Arc::ptr_eq(
        looked_up.wgpu_device_arc(),
        client.wgpu_device_arc()
    ));
    let default = WgpuRuntime::default_client(&index_device);
    assert!(Arc::ptr_eq(
        default.wgpu_device_arc(),
        client.wgpu_device_arc()
    ));

    assert_runs_a_binary_op(&client, INSTALLED_INDEX);

    // The packed luma kernel, the op vidr installs a device to run.
    let y =
        Tensor::<WgpuRuntime>::from_slice(&[0x0033_2211u32], &[1, 1], &index_device).expect("y");
    let padded = client
        .pad_packed_luma(PackedPlane::new(&y, 3), 2, 4, DType::U8)
        .expect("pad");
    let words: Vec<u32> = padded.to_vec();
    assert_eq!(words, vec![0x3333_2211, 0x3333_2211]);
}

/// A second install at an index is refused, and the first client stays.
#[test]
fn test_second_install_is_refused() {
    let Some(adapter) = first_adapter() else {
        return;
    };
    let (Some((first, first_queue)), Some((second, second_queue))) = (
        open(&adapter, adapter.limits()),
        open(&adapter, adapter.limits()),
    ) else {
        return;
    };
    let client =
        WgpuClient::install(REFUSED_SECOND_INDEX, &adapter, first, first_queue).expect("install");
    match WgpuClient::install(REFUSED_SECOND_INDEX, &adapter, second, second_queue) {
        Err(WgpuError::IndexInUse { index }) => assert_eq!(index, REFUSED_SECOND_INDEX),
        other => panic!("expected IndexInUse, got {other:?}"),
    }
    let looked_up = WgpuClient::new(WgpuDevice::new(REFUSED_SECOND_INDEX)).expect("lookup");
    assert!(Arc::ptr_eq(
        looked_up.wgpu_device_arc(),
        client.wgpu_device_arc()
    ));
    assert_runs_a_binary_op(&client, REFUSED_SECOND_INDEX);
}

/// An index whose client numr created on first use is refused too.
#[test]
fn test_install_refuses_an_index_created_on_first_use() {
    let Some(adapter) = first_adapter() else {
        return;
    };
    let Ok(existing) = WgpuClient::new(WgpuDevice::new(0)) else {
        println!("skipping: no client for adapter 0");
        return;
    };
    let Some((device, queue)) = open(&adapter, adapter.limits()) else {
        return;
    };
    match WgpuClient::install(0, &adapter, device, queue) {
        Err(WgpuError::IndexInUse { index: 0 }) => {}
        other => panic!("expected IndexInUse, got {other:?}"),
    }
    let looked_up = WgpuClient::new(WgpuDevice::new(0)).expect("lookup");
    assert!(Arc::ptr_eq(
        looked_up.wgpu_device_arc(),
        existing.wgpu_device_arc()
    ));
}

/// A device below numr's limits is refused by name and leaves the index free.
#[test]
fn test_install_refuses_a_device_below_the_limit_floor() {
    let Some(adapter) = first_adapter() else {
        return;
    };
    let low = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 4,
        ..wgpu::Limits::default()
    };
    let Some((device, queue)) = open(&adapter, low) else {
        return;
    };
    match WgpuClient::install(BELOW_FLOOR_INDEX, &adapter, device, queue) {
        Err(WgpuError::DeviceUnsupported { index, reason }) => {
            assert_eq!(index, BELOW_FLOOR_INDEX);
            assert!(
                reason.contains("max_storage_buffers_per_shader_stage"),
                "{reason}"
            );
        }
        other => panic!("expected DeviceUnsupported, got {other:?}"),
    }
    let Some((device, queue)) = open(&adapter, adapter.limits()) else {
        return;
    };
    let client = WgpuClient::install(BELOW_FLOOR_INDEX, &adapter, device, queue)
        .expect("the refused install left the index free");
    assert_runs_a_binary_op(&client, BELOW_FLOOR_INDEX);
}
