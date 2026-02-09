// Backend parity-style correctness tests for RandomOps.
// Random streams are backend-specific; these tests enforce shared invariants.

use numr::dtype::DType;
use numr::ops::RandomOps;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

fn check_uniform_f32(vals: &[f32]) {
    for &v in vals {
        assert!((0.0..1.0).contains(&v), "rand value out of range: {}", v);
    }
}

fn check_normal_stats_f32(vals: &[f32]) {
    let n = vals.len() as f32;
    let mean: f32 = vals.iter().sum::<f32>() / n;
    let var: f32 = vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    assert!(mean.abs() < 0.15, "randn mean too far from 0: {}", mean);
    assert!((var - 1.0).abs() < 0.2, "randn var too far from 1: {}", var);
}

#[test]
fn test_rand_invariants_all_backends() {
    let (cpu_client, _) = create_cpu_client();
    let cpu: Vec<f32> = cpu_client.rand(&[4096], DType::F32).unwrap().to_vec();
    check_uniform_f32(&cpu);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, _| {
        let got: Vec<f32> = cuda_client.rand(&[4096], DType::F32).unwrap().to_vec();
        check_uniform_f32(&got);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, _| {
        let got: Vec<f32> = wgpu_client.rand(&[4096], DType::F32).unwrap().to_vec();
        check_uniform_f32(&got);
    });
}

#[test]
fn test_randn_invariants_all_backends() {
    let (cpu_client, _) = create_cpu_client();
    let cpu: Vec<f32> = cpu_client.randn(&[4096], DType::F32).unwrap().to_vec();
    check_normal_stats_f32(&cpu);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, _| {
        let got: Vec<f32> = cuda_client.randn(&[4096], DType::F32).unwrap().to_vec();
        check_normal_stats_f32(&got);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, _| {
        let got: Vec<f32> = wgpu_client.randn(&[4096], DType::F32).unwrap().to_vec();
        check_normal_stats_f32(&got);
    });
}

#[test]
fn test_randint_invariants_all_backends() {
    let (cpu_client, _) = create_cpu_client();
    let cpu: Vec<i32> = cpu_client
        .randint(-7, 9, &[2048], DType::I32)
        .unwrap()
        .to_vec();
    assert!(cpu.iter().all(|&x| (-7..9).contains(&x)));

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, _| {
        let got: Vec<i32> = cuda_client
            .randint(-7, 9, &[2048], DType::I32)
            .unwrap()
            .to_vec();
        assert!(got.iter().all(|&x| (-7..9).contains(&x)));
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, _| {
        let got: Vec<i32> = wgpu_client
            .randint(-7, 9, &[2048], DType::I32)
            .unwrap()
            .to_vec();
        assert!(got.iter().all(|&x| (-7..9).contains(&x)));
    });
}

#[test]
fn test_rand_shape_dtype_all_backends() {
    let (cpu_client, _) = create_cpu_client();
    let cpu = cpu_client.rand(&[2, 3, 4], DType::F32).unwrap();
    assert_eq!(cpu.shape(), &[2, 3, 4]);
    assert_eq!(cpu.dtype(), DType::F32);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, _| {
        let t = cuda_client.rand(&[2, 3, 4], DType::F32).unwrap();
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.dtype(), DType::F32);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, _| {
        let t = wgpu_client.rand(&[2, 3, 4], DType::F32).unwrap();
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.dtype(), DType::F32);
    });
}
