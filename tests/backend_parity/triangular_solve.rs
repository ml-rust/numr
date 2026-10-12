// Backend parity tests for batched triangular solves.
//
// CPU is the reference. CUDA and WebGPU solve the same F32 batch in one call
// and must match within a tolerance that covers the different summation
// order: CPU subtracts one accumulated sum, the GPU kernels subtract term by
// term.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

const BATCH_DIMS: [usize; 2] = [2, 3];
const N: usize = 8;
const NRHS: usize = 5;

/// Deterministic values in `[lo, hi)`.
fn lcg(seed: u64, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lo + (hi - lo) * ((state >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

/// Batch of matrices with a diagonal in `[1, 2)`; both triangles are filled,
/// so each solve reads only its own.
fn matrices(seed: u64, batch: usize, n: usize) -> Vec<f32> {
    let mut a = lcg(seed, batch * n * n, -0.5, 0.5);
    let diag = lcg(seed ^ 0xd, batch * n, 1.0, 2.0);
    for m in 0..batch {
        for i in 0..n {
            a[m * n * n + i * n + i] = diag[m * n + i];
        }
    }
    a
}

fn shapes() -> (Vec<usize>, Vec<usize>, usize) {
    let mut a_shape = BATCH_DIMS.to_vec();
    a_shape.extend([N, N]);
    let mut b_shape = BATCH_DIMS.to_vec();
    b_shape.extend([N, NRHS]);
    (a_shape, b_shape, BATCH_DIMS.iter().product())
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-5 + 1e-5 * w.abs();
        assert!((g - w).abs() <= tol, "{what} at {i}: {g} vs {w}");
    }
}

/// Run one solve kind on CPU and every enabled GPU backend.
fn check(kind: &str, lower: bool, unit: bool) {
    let (a_shape, b_shape, batch) = shapes();
    let a = matrices(0x7a1e, batch, N);
    let b = lcg(0xb0b, batch * N * NRHS, -1.0, 1.0);

    let (cpu, cpu_device) = create_cpu_client();
    let at = Tensor::<CpuRuntime>::from_slice(&a, &a_shape, &cpu_device).unwrap();
    let bt = Tensor::<CpuRuntime>::from_slice(&b, &b_shape, &cpu_device).unwrap();
    let want = if lower {
        cpu.solve_triangular_lower(&at, &bt, unit).unwrap()
    } else {
        cpu.solve_triangular_upper(&at, &bt).unwrap()
    };
    assert_eq!(want.shape(), b_shape.as_slice());
    let want = want.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        use numr::runtime::cuda::CudaRuntime;
        let at = Tensor::<CudaRuntime>::from_slice(&a, &a_shape, &device).unwrap();
        let bt = Tensor::<CudaRuntime>::from_slice(&b, &b_shape, &device).unwrap();
        let got = if lower {
            client.solve_triangular_lower(&at, &bt, unit).unwrap()
        } else {
            client.solve_triangular_upper(&at, &bt).unwrap()
        };
        assert_eq!(got.shape(), b_shape.as_slice());
        assert_close(&got.to_vec::<f32>(), &want, &format!("{kind} CUDA vs CPU"));
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let at = Tensor::<WgpuRuntime>::from_slice(&a, &a_shape, &device).unwrap();
        let bt = Tensor::<WgpuRuntime>::from_slice(&b, &b_shape, &device).unwrap();
        let got = if lower {
            client.solve_triangular_lower(&at, &bt, unit).unwrap()
        } else {
            client.solve_triangular_upper(&at, &bt).unwrap()
        };
        assert_eq!(got.shape(), b_shape.as_slice());
        assert_close(
            &got.to_vec::<f32>(),
            &want,
            &format!("{kind} WebGPU vs CPU"),
        );
    });
}

#[test]
fn test_solve_triangular_lower_batched_parity() {
    check("solve_triangular_lower", true, false);
}

#[test]
fn test_solve_triangular_lower_unit_batched_parity() {
    check("solve_triangular_lower unit", true, true);
}

#[test]
fn test_solve_triangular_upper_batched_parity() {
    check("solve_triangular_upper", false, false);
}
