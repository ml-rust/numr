// Backend parity tests for the minimum-norm least-squares solve.
//
// CPU is the reference. The minimum-norm solution is unique, so every
// backend must return the same solution, singular values and rank within a
// tolerance that covers the different SVD and matmul summation order. The
// rank-deficient case passes an explicit rcond well inside the gap between
// the true and the noise singular values, so the rank is the same everywhere.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

/// One least-squares problem.
struct Case {
    name: &'static str,
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
    /// Explicit rcond per dtype: `(F32, F64)`. `None` uses the default.
    rcond: Option<(f64, f64)>,
}

/// Solution, singular values and rank, read back to the host.
struct Outcome {
    solution: Vec<f64>,
    solution_shape: Vec<usize>,
    singular_values: Vec<f64>,
    rank: Vec<i64>,
}

/// Deterministic values in `[-1, 1)`.
fn lcg(seed: u64, len: usize) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            2.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
        })
        .collect()
}

/// Random `[m, n]` matrix with `3` added to its leading diagonal.
fn well_conditioned(seed: u64, m: usize, n: usize) -> Vec<f64> {
    let mut a = lcg(seed, m * n);
    for i in 0..m.min(n) {
        a[i * n + i] += 3.0;
    }
    a
}

/// Rank-2 `[5, 4]` matrix: a `[5, 2]` factor times a `[2, 4]` factor.
fn rank_two() -> Vec<f64> {
    let l = well_conditioned(31, 5, 2);
    let r = well_conditioned(32, 2, 4);
    let mut a = vec![0.0; 20];
    for i in 0..5 {
        for j in 0..4 {
            a[i * 4 + j] = l[i * 2] * r[j] + l[i * 2 + 1] * r[4 + j];
        }
    }
    a
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "overdetermined 6x3",
            a: well_conditioned(11, 6, 3),
            a_shape: vec![6, 3],
            b: lcg(12, 12),
            b_shape: vec![6, 2],
            rcond: None,
        },
        Case {
            name: "underdetermined 3x6",
            a: well_conditioned(21, 3, 6),
            a_shape: vec![3, 6],
            b: lcg(22, 6),
            b_shape: vec![3, 2],
            rcond: None,
        },
        Case {
            name: "rank-deficient 5x4 rank 2",
            a: rank_two(),
            a_shape: vec![5, 4],
            b: lcg(33, 10),
            b_shape: vec![5, 2],
            rcond: Some((1e-4, 1e-10)),
        },
        Case {
            name: "batched 2x4x3",
            a: [well_conditioned(41, 4, 3), well_conditioned(42, 4, 3)].concat(),
            a_shape: vec![2, 4, 3],
            b: lcg(43, 8),
            b_shape: vec![2, 4, 1],
            rcond: None,
        },
    ]
}

fn tensor<R: Runtime<DType = DType>>(
    data: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> Tensor<R> {
    match dtype {
        DType::F64 => Tensor::<R>::from_slice(data, shape, device).unwrap(),
        DType::F32 => {
            let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            Tensor::<R>::from_slice(&v, shape, device).unwrap()
        }
        other => panic!("lstsq parity: unexpected dtype {other:?}"),
    }
}

fn read<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Vec<f64> {
    match t.dtype() {
        DType::F64 => t.to_vec::<f64>(),
        DType::F32 => t.to_vec::<f32>().into_iter().map(f64::from).collect(),
        other => panic!("lstsq parity: unexpected dtype {other:?}"),
    }
}

fn run<R, C>(client: &C, device: &R::Device, case: &Case, dtype: DType) -> Outcome
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R>,
{
    let a = tensor::<R>(&case.a, &case.a_shape, dtype, device);
    let b = tensor::<R>(&case.b, &case.b_shape, dtype, device);
    let rcond = case
        .rcond
        .map(|(r32, r64)| if dtype == DType::F64 { r64 } else { r32 });
    let r = client
        .lstsq_rcond(&a, &b, rcond)
        .unwrap_or_else(|e| panic!("{} [{dtype:?}]: lstsq_rcond failed: {e}", case.name));
    assert_eq!(r.solution.dtype(), dtype, "{}: solution dtype", case.name);
    assert_eq!(r.rank.dtype(), DType::I64, "{}: rank dtype", case.name);
    Outcome {
        solution: read(&r.solution),
        solution_shape: r.solution.shape().to_vec(),
        singular_values: read(&r.singular_values),
        rank: r.rank.to_vec::<i64>(),
    }
}

fn assert_close(got: &[f64], want: &[f64], rtol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what} at {i}: {g} is not finite");
        assert!(
            (g - w).abs() <= rtol * (1.0 + w.abs()),
            "{what} at {i}: {g} vs {w}"
        );
    }
}

fn assert_same(got: &Outcome, want: &Outcome, dtype: DType, what: &str) {
    let rtol = if dtype == DType::F64 { 1e-9 } else { 1e-3 };
    assert_eq!(got.solution_shape, want.solution_shape, "{what}: shape");
    assert_eq!(got.rank, want.rank, "{what}: rank");
    assert_close(
        &got.solution,
        &want.solution,
        rtol,
        &format!("{what}: solution"),
    );
    assert_close(
        &got.singular_values,
        &want.singular_values,
        rtol,
        &format!("{what}: singular values"),
    );
}

/// Run every case for `dtype` on CPU and each enabled GPU backend that
/// computes it.
fn check(dtype: DType) {
    let (cpu, cpu_device) = create_cpu_client();
    for case in cases() {
        let want = run::<CpuRuntime, _>(&cpu, &cpu_device, &case, dtype);

        #[cfg(feature = "cuda")]
        with_cuda_backend(|client, device| {
            use numr::runtime::cuda::CudaRuntime;
            let got = run::<CudaRuntime, _>(&client, &device, &case, dtype);
            assert_same(
                &got,
                &want,
                dtype,
                &format!("{} CUDA vs CPU [{dtype:?}]", case.name),
            );
        });

        #[cfg(feature = "wgpu")]
        if dtype == DType::F32 {
            with_wgpu_backend(|client, device| {
                use numr::runtime::wgpu::WgpuRuntime;
                let got = run::<WgpuRuntime, _>(&client, &device, &case, dtype);
                assert_same(
                    &got,
                    &want,
                    dtype,
                    &format!("{} WebGPU vs CPU [{dtype:?}]", case.name),
                );
            });
        }
    }
}

#[test]
fn test_lstsq_parity_f32() {
    check(DType::F32);
}

#[test]
fn test_lstsq_parity_f64() {
    check(DType::F64);
}

#[test]
fn test_lstsq_cpu_reference_ranks() {
    let (cpu, device) = create_cpu_client();
    let ranks: Vec<Vec<i64>> = cases()
        .iter()
        .map(|case| run::<CpuRuntime, _>(&cpu, &device, case, DType::F64).rank)
        .collect();
    assert_eq!(ranks, vec![vec![3], vec![3], vec![2], vec![3, 3]]);
}
