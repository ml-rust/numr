// Destination-passing rand_seeded_into / randn_seeded_into: parity against
// the allocating rand_seeded/randn_seeded, repeat-call reproducibility, and
// full-overwrite of a pre-filled destination.

use numr::dtype::DType;
use numr::ops::RandomOps;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::create_cpu_client;

// ============================================================
// Destination-passing rand_seeded_into / randn_seeded_into
// ============================================================

#[test]
fn test_rand_seeded_into_matches_rand_seeded_cpu() {
    let (client, device) = create_cpu_client();
    let allocated = client.rand_seeded(&[100], DType::F32, 42).unwrap();
    let out = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(&[100], DType::F32, &device).unwrap();
    client.rand_seeded_into(&out, 42).unwrap();
    let out_vec: Vec<f32> = out.to_vec();
    let allocated_vec: Vec<f32> = allocated.to_vec();
    assert_eq!(
        out_vec, allocated_vec,
        "rand_seeded_into must be byte-identical to rand_seeded for the same shape/dtype/seed"
    );
}

#[test]
fn test_randn_seeded_into_matches_randn_seeded_cpu() {
    let (client, device) = create_cpu_client();
    let allocated = client.randn_seeded(&[100], DType::F32, 42).unwrap();
    let out = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(&[100], DType::F32, &device).unwrap();
    client.randn_seeded_into(&out, 42).unwrap();
    let out_vec: Vec<f32> = out.to_vec();
    let allocated_vec: Vec<f32> = allocated.to_vec();
    assert_eq!(
        out_vec, allocated_vec,
        "randn_seeded_into must be byte-identical to randn_seeded for the same shape/dtype/seed"
    );
}

/// Calling `rand_seeded_into` twice with the same seed into the same
/// destination must produce identical values both times: no hidden
/// generator state may carry over between calls.
#[test]
fn test_rand_seeded_into_repeats_with_same_seed_cpu() {
    let (client, device) = create_cpu_client();
    let out = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(&[100], DType::F32, &device).unwrap();

    client.rand_seeded_into(&out, 42).unwrap();
    let first: Vec<f32> = out.to_vec();

    client.rand_seeded_into(&out, 42).unwrap();
    let second: Vec<f32> = out.to_vec();

    assert_eq!(
        first, second,
        "rand_seeded_into called twice with the same seed must reproduce, \
         proving no hidden generator state persists between calls"
    );
}

/// A draw into a sentinel-filled destination must fully overwrite it: no
/// sentinel value may survive.
#[test]
fn test_rand_seeded_into_overwrites_sentinel_cpu() {
    let (client, device) = create_cpu_client();
    let out = Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(&[9.0f32; 100], &[100], &device)
        .unwrap();

    client.rand_seeded_into(&out, 99).unwrap();
    let vals: Vec<f32> = out.to_vec();
    assert!(
        vals.iter().all(|&v| v != 9.0),
        "rand_seeded_into must fully overwrite its destination; a sentinel value survived"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_rand_seeded_into_matches_rand_seeded_cuda() {
    with_cuda_backend(|client, device| {
        let allocated = client.rand_seeded(&[100], DType::F32, 42).unwrap();
        let out =
            Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(&[100], DType::F32, &device).unwrap();
        client.rand_seeded_into(&out, 42).unwrap();
        let out_vec: Vec<f32> = out.to_vec();
        let allocated_vec: Vec<f32> = allocated.to_vec();
        assert_eq!(
            out_vec, allocated_vec,
            "rand_seeded_into must be byte-identical to rand_seeded on CUDA"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_randn_seeded_into_matches_randn_seeded_cuda() {
    with_cuda_backend(|client, device| {
        let allocated = client.randn_seeded(&[10000], DType::F32, 42).unwrap();
        let out = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(&[10000], DType::F32, &device)
            .unwrap();
        client.randn_seeded_into(&out, 42).unwrap();
        let out_vec: Vec<f32> = out.to_vec();
        let allocated_vec: Vec<f32> = allocated.to_vec();
        assert_eq!(
            out_vec, allocated_vec,
            "randn_seeded_into must be byte-identical to randn_seeded on CUDA"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_rand_seeded_into_repeats_with_same_seed_cuda() {
    with_cuda_backend(|client, device| {
        let out =
            Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(&[100], DType::F32, &device).unwrap();

        client.rand_seeded_into(&out, 42).unwrap();
        let first: Vec<f32> = out.to_vec();

        client.rand_seeded_into(&out, 42).unwrap();
        let second: Vec<f32> = out.to_vec();

        assert_eq!(
            first, second,
            "rand_seeded_into called twice with the same seed must reproduce on CUDA"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_rand_seeded_into_overwrites_sentinel_cuda() {
    with_cuda_backend(|client, device| {
        let out =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&[9.0f32; 100], &[100], &device)
                .unwrap();

        client.rand_seeded_into(&out, 99).unwrap();
        let vals: Vec<f32> = out.to_vec();
        assert!(
            vals.iter().all(|&v| v != 9.0),
            "rand_seeded_into must fully overwrite its destination on CUDA; a sentinel value survived"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_rand_seeded_into_matches_rand_seeded_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let allocated = client.rand_seeded(&[100], DType::F32, 42).unwrap();
        let out =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::zeros(&[100], DType::F32, &device).unwrap();
        client.rand_seeded_into(&out, 42).unwrap();
        let out_vec: Vec<f32> = out.to_vec();
        let allocated_vec: Vec<f32> = allocated.to_vec();
        assert_eq!(
            out_vec, allocated_vec,
            "rand_seeded_into must be byte-identical to rand_seeded on WebGPU"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_randn_seeded_into_matches_randn_seeded_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let allocated = client.randn_seeded(&[10000], DType::F32, 42).unwrap();
        let out = Tensor::<numr::runtime::wgpu::WgpuRuntime>::zeros(&[10000], DType::F32, &device)
            .unwrap();
        client.randn_seeded_into(&out, 42).unwrap();
        let out_vec: Vec<f32> = out.to_vec();
        let allocated_vec: Vec<f32> = allocated.to_vec();
        assert_eq!(
            out_vec, allocated_vec,
            "randn_seeded_into must be byte-identical to randn_seeded on WebGPU"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_rand_seeded_into_repeats_with_same_seed_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let out =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::zeros(&[100], DType::F32, &device).unwrap();

        client.rand_seeded_into(&out, 42).unwrap();
        let first: Vec<f32> = out.to_vec();

        client.rand_seeded_into(&out, 42).unwrap();
        let second: Vec<f32> = out.to_vec();

        assert_eq!(
            first, second,
            "rand_seeded_into called twice with the same seed must reproduce on WebGPU"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_rand_seeded_into_overwrites_sentinel_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let out =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&[9.0f32; 100], &[100], &device)
                .unwrap();

        client.rand_seeded_into(&out, 99).unwrap();
        let vals: Vec<f32> = out.to_vec();
        assert!(
            vals.iter().all(|&v| v != 9.0),
            "rand_seeded_into must fully overwrite its destination on WebGPU; a sentinel value survived"
        );
    });
}
