//! Common test utilities
#![allow(dead_code)]

use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

/// Create a CPU client and device for testing
pub fn create_cpu_client() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

/// Assert two f64 slices are close within tolerance
///
/// Uses the formula: |a - b| <= atol + rtol * |b|
pub fn assert_allclose_f64(a: &[f64], b: &[f64], rtol: f64, atol: f64, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {} (diff={}, tol={})",
            msg,
            i,
            x,
            y,
            diff,
            tol
        );
    }
}

/// Create a CUDA client and device, returning None if CUDA is unavailable
#[cfg(feature = "cuda")]
pub fn create_cuda_client() -> Option<(CudaClient, CudaDevice)> {
    if !numr::runtime::cuda::is_cuda_available() {
        return None;
    }
    let init = std::panic::catch_unwind(|| {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        (client, device)
    });
    init.ok()
}

/// Create a WebGPU client and device, returning None if WebGPU is unavailable
#[cfg(feature = "wgpu")]
pub fn create_wgpu_client() -> Option<(WgpuClient, WgpuDevice)> {
    if !numr::runtime::wgpu::is_wgpu_available() {
        return None;
    }
    let init = std::panic::catch_unwind(|| {
        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        (client, device)
    });
    init.ok()
}

/// Assert two f32 slices are close within tolerance
#[allow(dead_code)]
pub fn assert_allclose_f32(a: &[f32], b: &[f32], rtol: f32, atol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {} (diff={}, tol={})",
            msg,
            i,
            x,
            y,
            diff,
            tol
        );
    }
}
