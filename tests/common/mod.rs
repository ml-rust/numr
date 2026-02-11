//! Common test utilities
#![allow(dead_code)]

use numr::dtype::DType;
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

// ============================================================================
// DType Support Framework
// ============================================================================

/// Returns list of dtypes supported by a specific backend
///
/// Used internally by `supported_dtypes` to determine which dtypes to test.
/// This is the source of truth for backend capabilities.
pub fn backend_supported_dtypes(backend: &str) -> Vec<DType> {
    match backend {
        #[cfg(feature = "cuda")]
        "cuda" => build_dtype_list(&[DType::F32, DType::F64, DType::I32, DType::U32]),
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            // WebGPU: WGSL limitation - no F64, F16, BF16, FP8
            vec![DType::F32, DType::I32, DType::U32]
        }
        _ => build_dtype_list(&[DType::F32, DType::F64, DType::I32, DType::U32]),
    }
}

/// Build a dtype list from base types, appending feature-gated types
fn build_dtype_list(base: &[DType]) -> Vec<DType> {
    let mut dtypes = base.to_vec();

    if cfg!(feature = "f16") {
        dtypes.push(DType::F16);
        dtypes.push(DType::BF16);
    }
    if cfg!(feature = "fp8") {
        dtypes.push(DType::FP8E4M3);
        dtypes.push(DType::FP8E5M2);
    }

    dtypes
}

/// Check if a dtype is supported on a given backend
///
/// ## Example
///
/// ```ignore
/// if is_dtype_supported("wgpu", DType::F32) {
///     // Run WebGPU test for F32
/// }
/// ```
pub fn is_dtype_supported(backend: &str, dtype: DType) -> bool {
    backend_supported_dtypes(backend).contains(&dtype)
}

/// Returns list of dtypes to test for a given backend
///
/// This is used by test macros to determine which dtypes to parameterize over.
/// For testing purposes, we test:
/// - CPU: All supported dtypes (F32, F64 always; F16/BF16 if f16 feature; FP8 if fp8 feature)
/// - CUDA: All supported dtypes
/// - WebGPU: F32 only (WGSL limitation - F64/F16/BF16/FP8 not supported)
pub fn supported_dtypes(backend: &str) -> Vec<DType> {
    match backend {
        #[cfg(feature = "cuda")]
        "cuda" => build_dtype_list(&[DType::F32, DType::F64]),
        #[cfg(feature = "wgpu")]
        "wgpu" => vec![DType::F32],
        _ => build_dtype_list(&[DType::F32, DType::F64]),
    }
}

/// Returns (rtol, atol) tolerance pair for a given dtype
///
/// See `assert_allclose_for_dtype` for precision details per dtype.
pub fn tolerance_for_dtype(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F32 => (1e-5, 1e-6),   // 0.001% relative, 1e-6 absolute
        DType::F64 => (1e-12, 1e-14), // Machine epsilon-level tolerance
        DType::F16 => (0.01, 0.1),    // 1% relative tolerance for half-precision
        DType::BF16 => (0.01, 0.1),   // 1% relative tolerance for BF16
        DType::FP8E4M3 => (0.1, 0.5), // 10% relative — 4-bit mantissa, range [-448, 448]
        DType::FP8E5M2 => (1.0, 1.0), // Very coarse — 2-bit mantissa, range [-57344, 57344]
        _ => (1e-5, 1e-6),            // Default tolerance
    }
}

/// Assert two f64 slices are close, with tolerance based on dtype
///
/// This handles different precision levels appropriately:
/// - F64: Machine epsilon-level tolerance
/// - F32: Standard single-precision tolerance
/// - F16/BF16: Relaxed tolerance due to reduced precision (1%)
/// - FP8E4M3: Coarse tolerance (10%) — 4-bit mantissa
/// - FP8E5M2: Very coarse tolerance (100%) — 2-bit mantissa
pub fn assert_allclose_for_dtype(actual: &[f64], expected: &[f64], dtype: DType, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: dtype={:?}: length mismatch",
        msg,
        dtype
    );
    let (rtol, atol) = tolerance_for_dtype(dtype);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        assert!(
            diff <= tol,
            "{}: dtype={:?}: element {} differs: {} vs {} (diff={:.2e}, tol={:.2e})",
            msg,
            dtype,
            i,
            a,
            e,
            diff,
            tol
        );
    }
}

/// Macro for parameterized testing across dtypes
///
/// Usage:
/// ```ignore
/// #[test]
/// fn test_add_parity() {
///     test_all_dtypes!("cuda", dtype => {
///         // test body using `dtype`
///         let result = client.add(&a, &b)?;
///         assert_eq!(result.dtype(), dtype);
///     });
/// }
/// ```
#[macro_export]
macro_rules! test_all_dtypes {
    ($backend:expr, $dtype:ident => $body:block) => {
        for $dtype in $crate::common::supported_dtypes($backend) {
            $body
        }
    };
}

/// Macro for conditional dtype testing (only on CUDA)
///
/// Useful for tests that only work on specific backends
#[macro_export]
macro_rules! test_cuda_dtypes {
    ($dtype:ident => $body:block) => {
        #[cfg(feature = "cuda")]
        for $dtype in $crate::common::supported_dtypes("cuda") {
            $body
        }
    };
}
