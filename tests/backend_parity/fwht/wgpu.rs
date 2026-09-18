#![cfg(feature = "wgpu")]
// Backend parity tests for FwhtOps - WebGPU vs CPU.
//
// CPU is the reference. Every case runs the same F32 input through both
// backends and compares in F64. Shapes cover four segments per row, a
// two-wide segment, a single segment at the local-pass tile cap, and two
// segments wide enough to need one and two global-pass strides.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::FwhtOps;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::{assert_allclose_f64, create_cpu_client};

/// Deterministic, non-periodic-in-power-of-two input.
fn input_data(numel: usize) -> Vec<f32> {
    (0..numel)
        .map(|i| ((i % 97) as f32) * 0.031 - 1.5)
        .collect()
}

/// Alternating +1/-1 sign row.
fn sign_data(width: usize) -> Vec<f32> {
    (0..width)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn widen(v: Vec<f32>) -> Vec<f64> {
    v.into_iter().map(f64::from).collect()
}

/// Runs `fwht` on CPU and WebGPU for one (shape, block_size, signs) case and
/// compares the results in F64.
fn check_parity(shape: [usize; 2], block_size: usize, with_signs: bool, tol: f64) {
    with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let last_dim = shape[1];
        let numel: usize = shape.iter().product();
        let data = input_data(numel);
        let signs = with_signs.then(|| sign_data(last_dim));

        let x_cpu = Tensor::<CpuRuntime>::from_slice(&data, &shape, &cpu_device)
            .expect("CPU tensor creation");
        let signs_cpu = signs
            .as_ref()
            .map(|s| Tensor::<CpuRuntime>::from_slice(s, &[last_dim], &cpu_device))
            .transpose()
            .expect("CPU signs creation");
        let expected = cpu_client
            .fwht(&x_cpu, block_size, signs_cpu.as_ref())
            .expect("CPU fwht failed");

        let x_wgpu = Tensor::<WgpuRuntime>::from_slice(&data, &shape, &wgpu_device)
            .expect("WGPU tensor creation");
        let signs_wgpu = signs
            .as_ref()
            .map(|s| Tensor::<WgpuRuntime>::from_slice(s, &[last_dim], &wgpu_device))
            .transpose()
            .expect("WGPU signs creation");
        let result = wgpu_client
            .fwht(&x_wgpu, block_size, signs_wgpu.as_ref())
            .expect("WGPU fwht failed");

        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.shape(), expected.shape());

        let got = widen(result.to_vec::<f32>());
        let want = widen(expected.to_vec::<f32>());

        assert_allclose_f64(
            &got,
            &want,
            tol,
            tol,
            &format!("fwht_wgpu_F32_{shape:?}_block{block_size}_signs{with_signs}"),
        );
    });
}

const TOL: f64 = 1e-4;

#[test]
fn test_fwht_f32_four_segments_wgpu_matches_cpu() {
    check_parity([3, 4096], 1024, false, TOL);
    check_parity([3, 4096], 1024, true, TOL);
}

#[test]
fn test_fwht_f32_block_2_wgpu_matches_cpu() {
    check_parity([3, 4096], 2, false, TOL);
    check_parity([3, 4096], 2, true, TOL);
}

#[test]
fn test_fwht_f32_block_at_tile_cap_wgpu_matches_cpu() {
    check_parity([3, 4096], 4096, false, TOL);
    check_parity([3, 4096], 4096, true, TOL);
}

#[test]
fn test_fwht_f32_block_8192_one_global_stride_wgpu_matches_cpu() {
    check_parity([2, 8192], 8192, false, TOL);
    check_parity([2, 8192], 8192, true, TOL);
}

#[test]
fn test_fwht_f32_block_16384_two_global_strides_wgpu_matches_cpu() {
    check_parity([1, 16384], 16384, false, TOL);
    check_parity([1, 16384], 16384, true, TOL);
}

#[test]
fn test_fwht_f64_is_unsupported_dtype() {
    with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
        let x = Tensor::<WgpuRuntime>::from_slice(&[0.0f64; 8], &[8], &wgpu_device)
            .expect("WGPU F64 tensor creation");
        let err = wgpu_client
            .fwht(&x, 8, None)
            .expect_err("F64 must be rejected on WebGPU");
        match err {
            Error::UnsupportedDType { dtype, op } => {
                assert_eq!(dtype, DType::F64);
                assert_eq!(op, "fwht");
            }
            other => panic!("expected UnsupportedDType, got {other:?}"),
        }
    });
}

#[test]
fn test_fwht_empty_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        let x_cpu = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 0], &[0, 8], &cpu_device)
            .expect("CPU tensor creation");
        let x_wgpu = Tensor::<WgpuRuntime>::from_slice(&[0.0f32; 0], &[0, 8], &wgpu_device)
            .expect("WGPU tensor creation");
        let expected = cpu_client.fwht(&x_cpu, 8, None).expect("CPU fwht failed");
        let result = wgpu_client
            .fwht(&x_wgpu, 8, None)
            .expect("WGPU fwht failed");
        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.numel(), 0);
    });
}
