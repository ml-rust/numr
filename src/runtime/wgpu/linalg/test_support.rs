//! Shared test helpers for WebGPU linalg unit tests.
#![cfg(test)]

use super::super::{WgpuClient, WgpuDevice, WgpuRuntime};
use crate::runtime::Runtime;

pub(super) fn create_client() -> WgpuClient {
    let device = WgpuDevice::new(0);
    WgpuRuntime::default_client(&device)
}
