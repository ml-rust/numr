#![cfg(test)]
//! Shared test-only helpers for `runtime::cpu::linalg` unit tests.

use super::super::{CpuClient, CpuDevice};

pub(super) fn create_client() -> CpuClient {
    let device = CpuDevice::new();
    CpuClient::new(device)
}
