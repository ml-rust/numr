#![cfg(feature = "cuda")]
// Backend parity tests for FwhtOps - CUDA vs CPU.
//
// CPU is the reference. Every case runs the same input through both backends
// and compares in F64. Shapes cover four segments per row, a two-wide
// segment, a single segment wider than one CUDA block (so every thread owns
// more than one element), and the widest F32 segment the shared-memory
// kernel accepts.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::FwhtOps;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{assert_allclose_f64, create_cpu_client};

/// Deterministic, non-periodic-in-power-of-two input.
fn input_data(numel: usize) -> Vec<f64> {
    (0..numel)
        .map(|i| ((i % 97) as f64) * 0.031 - 1.5)
        .collect()
}

/// Alternating +1/-1 sign row.
fn sign_data(width: usize) -> Vec<f64> {
    (0..width)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

/// Runs `fwht` on CPU and CUDA for one (dtype, shape, block_size, signs) case
/// and compares the results in F64.
fn check_parity(dtype: DType, shape: [usize; 2], block_size: usize, with_signs: bool, tol: f64) {
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        let last_dim = shape[1];
        let numel: usize = shape.iter().product();
        let data = input_data(numel);
        let signs = with_signs.then(|| sign_data(last_dim));

        let x_cpu = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU tensor creation");
        let signs_cpu = signs
            .as_ref()
            .map(|s| tensor_from_f64(s, &[last_dim], dtype, &cpu_device, &cpu_client))
            .transpose()
            .expect("CPU signs creation");
        let expected = cpu_client
            .fwht(&x_cpu, block_size, signs_cpu.as_ref())
            .expect("CPU fwht failed");

        let x_cuda = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
            .expect("CUDA tensor creation");
        let signs_cuda = signs
            .as_ref()
            .map(|s| tensor_from_f64(s, &[last_dim], dtype, &cuda_device, &cuda_client))
            .transpose()
            .expect("CUDA signs creation");
        let result = cuda_client
            .fwht(&x_cuda, block_size, signs_cuda.as_ref())
            .expect("CUDA fwht failed");

        assert_eq!(result.dtype(), dtype);
        assert_eq!(result.shape(), expected.shape());

        let got = result
            .to_dtype(DType::F64)
            .expect("cast to f64")
            .to_vec::<f64>();
        let want = expected
            .to_dtype(DType::F64)
            .expect("cast to f64")
            .to_vec::<f64>();

        assert_allclose_f64(
            &got,
            &want,
            tol,
            tol,
            &format!("fwht_cuda_{dtype:?}_{shape:?}_block{block_size}_signs{with_signs}"),
        );
    });
}

/// The shapes every dtype runs: four segments per row, two-wide segments, and
/// one segment per row wider than a CUDA block.
fn check_dtype(dtype: DType, tol: f64) {
    check_parity(dtype, [3, 4096], 1024, false, tol);
    check_parity(dtype, [3, 4096], 1024, true, tol);
    check_parity(dtype, [3, 4096], 2, false, tol);
    check_parity(dtype, [3, 4096], 2, true, tol);
    check_parity(dtype, [3, 4096], 4096, false, tol);
    check_parity(dtype, [3, 4096], 4096, true, tol);
}

#[test]
fn test_fwht_f32_cuda_matches_cpu() {
    check_dtype(DType::F32, 1e-4);
}

#[test]
fn test_fwht_f32_block_8192_cuda_matches_cpu() {
    check_parity(DType::F32, [3, 8192], 8192, false, 1e-4);
    check_parity(DType::F32, [3, 8192], 8192, true, 1e-4);
}

#[test]
fn test_fwht_f64_cuda_matches_cpu() {
    check_dtype(DType::F64, 1e-10);
}

#[cfg(feature = "f16")]
#[test]
fn test_fwht_f16_cuda_matches_cpu() {
    check_dtype(DType::F16, 0.05);
}

#[cfg(feature = "f16")]
#[test]
fn test_fwht_bf16_cuda_matches_cpu() {
    check_dtype(DType::BF16, 0.2);
}

#[test]
fn test_fwht_block_size_over_shared_memory_limit_is_invalid_argument() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let block_size = 16384;
        let data = vec![0.0f32; block_size];
        let x = Tensor::<CudaRuntime>::from_slice(&data, &[block_size], &cuda_device)
            .expect("CUDA tensor creation");
        let err = cuda_client
            .fwht(&x, block_size, None)
            .expect_err("block_size 16384 must exceed the F32 shared-memory cap");
        match err {
            Error::InvalidArgument { arg, reason } => {
                assert_eq!(arg, "block_size");
                assert!(reason.contains("12288"), "reason: {reason}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    });
}

#[test]
fn test_fwht_empty_cuda_matches_cpu() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        let x_cpu = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 0], &[0, 8], &cpu_device)
            .expect("CPU tensor creation");
        let x_cuda = Tensor::<CudaRuntime>::from_slice(&[0.0f32; 0], &[0, 8], &cuda_device)
            .expect("CUDA tensor creation");
        let expected = cpu_client.fwht(&x_cpu, 8, None).expect("CPU fwht failed");
        let result = cuda_client
            .fwht(&x_cuda, 8, None)
            .expect("CUDA fwht failed");
        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.numel(), 0);
    });
}
