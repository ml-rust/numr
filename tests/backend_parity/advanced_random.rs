// Backend parity-style correctness tests for AdvancedRandomOps.
// Range and reproducibility checks per generator. Philox known answers and
// bit parity live in `advanced_random_philox.rs`.

use numr::dtype::DType;
use numr::ops::AdvancedRandomOps;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

fn assert_uniform01_f32(v: &[f32]) {
    assert!(v.iter().all(|&x| (0.0..1.0).contains(&x)));
}

fn assert_reproducible(a: Vec<f32>, b: Vec<f32>, name: &str) {
    assert_eq!(a, b, "{} reproducibility failed", name);
}

#[test]
fn test_philox_uniform_repro_and_range() {
    let (cpu_client, _) = create_cpu_client();
    let a: Vec<f32> = cpu_client
        .philox_uniform(&[256], 42, 0, DType::F32)
        .unwrap()
        .to_vec();
    let b: Vec<f32> = cpu_client
        .philox_uniform(&[256], 42, 0, DType::F32)
        .unwrap()
        .to_vec();
    assert_uniform01_f32(&a);
    assert_reproducible(a, b, "philox_cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, _| {
        let a: Vec<f32> = cuda_client
            .philox_uniform(&[256], 42, 0, DType::F32)
            .unwrap()
            .to_vec();
        let b: Vec<f32> = cuda_client
            .philox_uniform(&[256], 42, 0, DType::F32)
            .unwrap()
            .to_vec();
        assert_uniform01_f32(&a);
        assert_reproducible(a, b, "philox_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, _| {
        let a: Vec<f32> = wgpu_client
            .philox_uniform(&[256], 42, 0, DType::F32)
            .unwrap()
            .to_vec();
        let b: Vec<f32> = wgpu_client
            .philox_uniform(&[256], 42, 0, DType::F32)
            .unwrap()
            .to_vec();
        assert_uniform01_f32(&a);
        assert_reproducible(a, b, "philox_wgpu");
    });
}

#[test]
fn test_threefry_uniform_repro_and_range() {
    let (cpu_client, _) = create_cpu_client();
    let a: Vec<f32> = cpu_client
        .threefry_uniform(&[256], 7, 11, DType::F32)
        .unwrap()
        .to_vec();
    let b: Vec<f32> = cpu_client
        .threefry_uniform(&[256], 7, 11, DType::F32)
        .unwrap()
        .to_vec();
    assert_uniform01_f32(&a);
    assert_reproducible(a, b, "threefry_cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, _| {
        let a: Vec<f32> = cuda_client
            .threefry_uniform(&[256], 7, 11, DType::F32)
            .unwrap()
            .to_vec();
        let b: Vec<f32> = cuda_client
            .threefry_uniform(&[256], 7, 11, DType::F32)
            .unwrap()
            .to_vec();
        assert_uniform01_f32(&a);
        assert_reproducible(a, b, "threefry_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, _| {
        let a: Vec<f32> = wgpu_client
            .threefry_uniform(&[256], 7, 11, DType::F32)
            .unwrap()
            .to_vec();
        let b: Vec<f32> = wgpu_client
            .threefry_uniform(&[256], 7, 11, DType::F32)
            .unwrap()
            .to_vec();
        assert_uniform01_f32(&a);
        assert_reproducible(a, b, "threefry_wgpu");
    });
}

#[test]
fn test_xoshiro_uniform_repro_and_range() {
    let (cpu_client, _) = create_cpu_client();
    let a: Vec<f32> = cpu_client
        .xoshiro256_uniform(&[256], 123, DType::F32)
        .unwrap()
        .to_vec();
    let b: Vec<f32> = cpu_client
        .xoshiro256_uniform(&[256], 123, DType::F32)
        .unwrap()
        .to_vec();
    assert_uniform01_f32(&a);
    assert_reproducible(a, b, "xoshiro_cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, _| {
        let a: Vec<f32> = cuda_client
            .xoshiro256_uniform(&[256], 123, DType::F32)
            .unwrap()
            .to_vec();
        let b: Vec<f32> = cuda_client
            .xoshiro256_uniform(&[256], 123, DType::F32)
            .unwrap()
            .to_vec();
        assert_uniform01_f32(&a);
        assert_reproducible(a, b, "xoshiro_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, _| {
        let a: Vec<f32> = wgpu_client
            .xoshiro256_uniform(&[256], 123, DType::F32)
            .unwrap()
            .to_vec();
        let b: Vec<f32> = wgpu_client
            .xoshiro256_uniform(&[256], 123, DType::F32)
            .unwrap()
            .to_vec();
        assert_uniform01_f32(&a);
        assert_reproducible(a, b, "xoshiro_wgpu");
    });
}

/// Cross-backend parity for `threefry_uniform` (ThreeFry4x64-20): all backends
/// use the same 64-bit counter-based generator. CPU and CUDA convert with the
/// full 53 bits, so they are bit-identical. WebGPU is F32-only and converts the
/// same u64 stream with 24-bit resolution, so it is equal within F32 rounding.
#[cfg(any(feature = "wgpu", feature = "cuda"))]
#[test]
fn test_threefry_uniform_cross_backend_parity() {
    let (cpu_client, _cd) = create_cpu_client();
    // Non-multiple-of-4 length to also cover the tail handling.
    let cpu = cpu_client
        .threefry_uniform(&[37], 7, 11, DType::F32)
        .unwrap()
        .to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cc, _cdev| {
        let c = cc
            .threefry_uniform(&[37], 7, 11, DType::F32)
            .unwrap()
            .to_vec::<f32>();
        // CPU and CUDA both use the full-precision (53-bit) conversion.
        assert_eq!(
            cpu, c,
            "threefry CPU vs CUDA diverges:\ncpu={cpu:?}\ncuda={c:?}"
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wc, _wd| {
        let w = wc
            .threefry_uniform(&[37], 7, 11, DType::F32)
            .unwrap()
            .to_vec::<f32>();
        // Same generator; difference is only the f32 conversion resolution (<2^-23).
        for (i, (a, b)) in cpu.iter().zip(w.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "threefry CPU vs WGPU diverges at {i}: {a} vs {b} (same stream expected)"
            );
        }
    });
}
