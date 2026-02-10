// Backend parity-style correctness tests for AdvancedRandomOps.
// Exact samples are RNG-implementation specific; we validate shared contracts.

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
