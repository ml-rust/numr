// Backend parity tests for SpecialFunctions

use numr::ops::SpecialFunctions;
use numr::tensor::Tensor;

use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

#[test]
fn test_erf_gamma_parity() {
    let xvals = [0.0f32, 0.5, 1.0, 2.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let x = Tensor::from_slice(&xvals, &[4], &cpu_device);
    let cpu_erf: Vec<f32> = cpu_client.erf(&x).unwrap().to_vec();
    let cpu_gamma: Vec<f32> = cpu_client.gamma(&x).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&xvals, &[4], &cuda_device);
        let got_erf: Vec<f32> = cuda_client.erf(&x).unwrap().to_vec();
        let got_gamma: Vec<f32> = cuda_client.gamma(&x).unwrap().to_vec();
        assert_parity_f32(&cpu_erf, &got_erf, "erf_cuda");
        assert_parity_f32(&cpu_gamma, &got_gamma, "gamma_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&xvals, &[4], &wgpu_device);
        let got_erf: Vec<f32> = wgpu_client.erf(&x).unwrap().to_vec();
        let got_gamma: Vec<f32> = wgpu_client.gamma(&x).unwrap().to_vec();
        assert_parity_f32(&cpu_erf, &got_erf, "erf_wgpu");
        assert_parity_f32(&cpu_gamma, &got_gamma, "gamma_wgpu");
    });
}

#[test]
fn test_incomplete_gamma_complement_parity() {
    let avals = [2.0f32, 3.0, 5.0];
    let xvals = [1.0f32, 2.0, 3.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let a = Tensor::from_slice(&avals, &[3], &cpu_device);
    let x = Tensor::from_slice(&xvals, &[3], &cpu_device);
    let p: Vec<f32> = cpu_client.gammainc(&a, &x).unwrap().to_vec();
    let q: Vec<f32> = cpu_client.gammaincc(&a, &x).unwrap().to_vec();
    for i in 0..3 {
        assert!((p[i] + q[i] - 1.0).abs() < 1e-5, "cpu P+Q != 1 at {}", i);
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::from_slice(&avals, &[3], &cuda_device);
        let x = Tensor::from_slice(&xvals, &[3], &cuda_device);
        let p2: Vec<f32> = cuda_client.gammainc(&a, &x).unwrap().to_vec();
        let q2: Vec<f32> = cuda_client.gammaincc(&a, &x).unwrap().to_vec();
        assert_parity_f32(&p, &p2, "gammainc_cuda");
        assert_parity_f32(&q, &q2, "gammaincc_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a = Tensor::from_slice(&avals, &[3], &wgpu_device);
        let x = Tensor::from_slice(&xvals, &[3], &wgpu_device);
        let p2: Vec<f32> = wgpu_client.gammainc(&a, &x).unwrap().to_vec();
        let q2: Vec<f32> = wgpu_client.gammaincc(&a, &x).unwrap().to_vec();
        assert_parity_f32(&p, &p2, "gammainc_wgpu");
        assert_parity_f32(&q, &q2, "gammaincc_wgpu");
    });
}
