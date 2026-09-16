//! Shared test helpers for CUDA linalg unit tests.
#![cfg(test)]

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::runtime::Runtime;
use crate::runtime::cuda::{CudaDevice, is_cuda_available};

pub(super) fn create_client() -> Option<CudaClient> {
    if !is_cuda_available() {
        return None;
    }
    let device = CudaDevice::new(0);
    Some(CudaRuntime::default_client(&device))
}
