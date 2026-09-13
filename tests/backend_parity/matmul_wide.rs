// Backend parity for `MatmulOps::matmul_wide`: the product written in the
// accumulator dtype.
//
// F16 and BF16 operands produce an F32 output, checked against an F64 CPU
// reference at the F32 tolerance — not the half tolerance. That is the
// contract: the result is the F32 accumulator, so the half rounding that
// `matmul` applies at its store must not appear. The operands sit on a grid
// both half formats hold exactly, so the reference and every backend multiply
// the same numbers and only the accumulation is under test.
//
// I8 widens to I32 as `matmul` does, and F32/F64 are `matmul` bit for bit.

use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::common::{
    assert_allclose_f64, create_cpu_client, gemm_long_k_tolerance, is_dtype_supported,
};

/// Values on a 2^-5 grid in `[-1, 1]`: six significant bits, exact in F16
/// and BF16. A product of two carries up to twelve, so a sum of products is
/// not representable in either half format once it passes a few units —
/// which is what lets the long-K case tell a wide output from a narrowed one.
///
/// A multiplicative hash picks the grid level, so consecutive elements carry
/// no period: a dot product over them behaves like a random walk, and its
/// partial sums run well above its result. A short linear pattern would sum
/// coherently instead and hide the cancellation the long-K case relies on.
fn grid(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let h = ((i as u64) + 977 * (seed as u64)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
            ((h % 65) as f64) / 32.0 - 1.0
        })
        .collect()
}

/// Batch dims of a matmul operand, left-padded with 1 to `rank` dims.
fn batch_dims(shape: &[usize], rank: usize) -> Vec<usize> {
    let own = &shape[..shape.len() - 2];
    let mut dims = vec![1usize; rank - own.len()];
    dims.extend_from_slice(own);
    dims
}

/// The broadcast output shape of `A @ B`, computed independently of numr.
fn broadcast_out_shape(a_shape: &[usize], b_shape: &[usize]) -> Vec<usize> {
    let rank = (a_shape.len() - 2).max(b_shape.len() - 2);
    let (a_b, b_b) = (batch_dims(a_shape, rank), batch_dims(b_shape, rank));
    let mut out: Vec<usize> = a_b.iter().zip(&b_b).map(|(&x, &y)| x.max(y)).collect();
    out.push(a_shape[a_shape.len() - 2]);
    out.push(b_shape[b_shape.len() - 1]);
    out
}

/// Which batch slice of an operand the flat output batch `flat` reads, under
/// per-dimension broadcasting (an extent-1 dim always reads slice 0).
fn operand_batch_index(operand_shape: &[usize], out_shape: &[usize], flat: usize) -> usize {
    let rank = out_shape.len() - 2;
    let dims = batch_dims(operand_shape, rank);
    let out_batch = &out_shape[..rank];
    // Decompose `flat` over the output batch dims, row-major.
    let mut rem = flat;
    let mut coords = vec![0usize; rank];
    for d in (0..rank).rev() {
        coords[d] = rem % out_batch[d];
        rem /= out_batch[d];
    }
    let mut idx = 0usize;
    for d in 0..rank {
        let c = if dims[d] == 1 { 0 } else { coords[d] };
        idx = idx * dims[d] + c;
    }
    idx
}

/// `C = A @ B` summed in F64, with per-dimension batch broadcasting over
/// `out_shape`.
fn reference(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<f64> {
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];
    let batch: usize = out_shape[..out_shape.len() - 2].iter().product();

    let mut out = vec![0.0f64; batch * m * n];
    for bi in 0..batch {
        let a_off = operand_batch_index(a_shape, out_shape, bi) * m * k;
        let b_off = operand_batch_index(b_shape, out_shape, bi) * k * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[a_off + i * k + p] * b[b_off + p * n + j];
                }
                out[bi * m * n + i * n + j] = acc;
            }
        }
    }
    out
}

/// The half dtypes this build carries.
fn half_dtypes() -> Vec<DType> {
    [DType::F16, DType::BF16]
        .into_iter()
        .filter(|&d| is_dtype_supported("cpu", d))
        .collect()
}

/// One shape: operand shapes and the batch-broadcast output shape.
struct Case {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    label: &'static str,
}

impl Case {
    fn new(a_shape: &[usize], b_shape: &[usize], label: &'static str) -> Self {
        Self {
            a_shape: a_shape.to_vec(),
            b_shape: b_shape.to_vec(),
            label,
        }
    }

    fn k(&self) -> usize {
        self.a_shape[self.a_shape.len() - 1]
    }

    fn out_shape(&self) -> Vec<usize> {
        broadcast_out_shape(&self.a_shape, &self.b_shape)
    }
}

/// 2-D and batched, aligned and ragged. `m=520, n=35` is heavy enough per
/// copied element that the CUDA WMMA policy pads it first; `m=37` is not.
fn cases() -> Vec<Case> {
    vec![
        Case::new(&[64, 32], &[32, 64], "aligned 2-D"),
        Case::new(&[37, 24], &[24, 40], "ragged m 2-D"),
        Case::new(&[37, 24], &[24, 35], "ragged m and n 2-D"),
        Case::new(&[520, 64], &[64, 35], "ragged n, padded on CUDA"),
        Case::new(&[3, 37, 24], &[3, 24, 40], "batched ragged m"),
        Case::new(
            &[2, 1, 16, 24],
            &[1, 3, 24, 35],
            "broadcast batch, ragged n",
        ),
    ]
}

/// The F32-class tolerance for a length-K sum of exact products of O(1)
/// operands.
fn f32_tolerance(k: usize) -> (f64, f64) {
    gemm_long_k_tolerance(DType::F32, k, 1.0)
}

fn run_case_on<R>(
    case: &Case,
    dtype: DType,
    device: &R::Device,
    client: &(impl MatmulOps<R> + numr::ops::TypeConversionOps<R>),
    backend: &str,
) -> Vec<f32>
where
    R: numr::runtime::Runtime<DType = DType>,
{
    let a_len: usize = case.a_shape.iter().product();
    let b_len: usize = case.b_shape.iter().product();
    let label = case.label;
    let a = tensor_from_f64(&grid(a_len, 1), &case.a_shape, dtype, device, client)
        .unwrap_or_else(|e| panic!("{backend} A failed for {label} [{dtype:?}]: {e}"));
    let b = tensor_from_f64(&grid(b_len, 2), &case.b_shape, dtype, device, client)
        .unwrap_or_else(|e| panic!("{backend} B failed for {label} [{dtype:?}]: {e}"));
    let out = client
        .matmul_wide(&a, &b)
        .unwrap_or_else(|e| panic!("{backend} matmul_wide failed for {label} [{dtype:?}]: {e}"));
    assert_eq!(
        out.dtype(),
        DType::F32,
        "{backend} matmul_wide output dtype for {label} [{dtype:?}]"
    );
    assert_eq!(
        out.shape(),
        case.out_shape().as_slice(),
        "{backend} matmul_wide output shape for {label} [{dtype:?}]"
    );
    out.to_vec::<f32>()
}

fn as_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// CPU F16/BF16 vs the F64 reference at F32 tolerance, every shape.
#[test]
fn matmul_wide_half_cpu_matches_f64_reference_at_f32_tolerance() {
    let (client, device) = create_cpu_client();
    for case in cases() {
        let a_len: usize = case.a_shape.iter().product();
        let b_len: usize = case.b_shape.iter().product();
        let want = reference(
            &grid(a_len, 1),
            &case.a_shape,
            &grid(b_len, 2),
            &case.b_shape,
            &case.out_shape(),
        );
        let (rtol, atol) = f32_tolerance(case.k());
        for dtype in half_dtypes() {
            let got = run_case_on::<CpuRuntime>(&case, dtype, &device, &client, "CPU");
            assert_allclose_f64(
                &as_f64(&got),
                &want,
                rtol,
                atol,
                &format!(
                    "matmul_wide CPU vs F64 reference: {} [{dtype:?}]",
                    case.label
                ),
            );
        }
    }
}

/// A long contraction whose terms cancel: partial sums run far above the
/// result, so a half-narrowed output would miss the F32 tolerance by orders
/// of magnitude. The wide result must still meet it, and the test checks
/// that the narrowed value would not, so the tolerance is proven to bite.
#[test]
fn matmul_wide_long_k_cancellation_keeps_f32_precision() {
    let (client, device) = create_cpu_client();
    let case = Case::new(&[8, 4096], &[4096, 24], "long-K cancellation");
    let a_len: usize = case.a_shape.iter().product();
    let b_len: usize = case.b_shape.iter().product();
    let want = reference(
        &grid(a_len, 1),
        &case.a_shape,
        &grid(b_len, 2),
        &case.b_shape,
        &case.out_shape(),
    );
    let (rtol, atol) = f32_tolerance(case.k());

    for dtype in half_dtypes() {
        let got = run_case_on::<CpuRuntime>(&case, dtype, &device, &client, "CPU");
        assert_allclose_f64(
            &as_f64(&got),
            &want,
            rtol,
            atol,
            &format!("matmul_wide long-K CPU vs F64 reference [{dtype:?}]"),
        );

        // The value `matmul` would store: the same reference rounded to the
        // half dtype. It must fail the F32 bound somewhere, or this test
        // would pass for a `matmul_wide` that narrowed.
        let want_t = Tensor::<CpuRuntime>::from_slice(&want, &case.out_shape(), &device)
            .expect("reference tensor");
        let narrowed = tensor_from_f64(&want, &case.out_shape(), dtype, &device, &client)
            .expect("narrowed reference");
        let narrowed = numr::ops::TypeConversionOps::cast(&client, &narrowed, DType::F64)
            .expect("widen back")
            .to_vec::<f64>();
        let exact = want_t.to_vec::<f64>();
        let some_element_misses = narrowed
            .iter()
            .zip(&exact)
            .any(|(n, e)| (n - e).abs() > atol + rtol * e.abs());
        assert!(
            some_element_misses,
            "long-K case is not discriminating for {dtype:?}: a half-narrowed output meets the F32 bound"
        );
    }
}

/// I8 operands come back as I32 with the same values `matmul` gives.
#[test]
fn matmul_wide_i8_widens_to_i32_like_matmul() {
    let (client, device) = create_cpu_client();
    let a_data: Vec<i8> = (0..37 * 24).map(|i| ((i * 7) % 255) as i8).collect();
    let b_data: Vec<i8> = (0..24 * 35).map(|i| ((i * 11) % 255) as i8).collect();
    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[37, 24], &device).expect("A");
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[24, 35], &device).expect("B");

    let wide = client.matmul_wide(&a, &b).expect("matmul_wide i8");
    let plain = client.matmul(&a, &b).expect("matmul i8");
    assert_eq!(wide.dtype(), DType::I32);
    assert_eq!(plain.dtype(), DType::I32);
    assert_eq!(wide.to_vec::<i32>(), plain.to_vec::<i32>());
}

/// F32 and F64 are `matmul`, bit for bit, including the batched shapes.
#[test]
fn matmul_wide_f32_f64_identical_to_matmul() {
    let (client, device) = create_cpu_client();
    for case in cases() {
        let a_len: usize = case.a_shape.iter().product();
        let b_len: usize = case.b_shape.iter().product();
        for dtype in [DType::F32, DType::F64] {
            let a = tensor_from_f64(&grid(a_len, 1), &case.a_shape, dtype, &device, &client)
                .expect("A");
            let b = tensor_from_f64(&grid(b_len, 2), &case.b_shape, dtype, &device, &client)
                .expect("B");
            let wide = client.matmul_wide(&a, &b).expect("matmul_wide");
            let plain = client.matmul(&a, &b).expect("matmul");
            assert_eq!(wide.dtype(), dtype, "{} [{dtype:?}]", case.label);
            match dtype {
                DType::F32 => assert_eq!(
                    wide.to_vec::<f32>(),
                    plain.to_vec::<f32>(),
                    "{} [F32]",
                    case.label
                ),
                _ => assert_eq!(
                    wide.to_vec::<f64>(),
                    plain.to_vec::<f64>(),
                    "{} [F64]",
                    case.label
                ),
            }
        }
    }
}

/// CUDA F16/BF16 vs the F64 reference and vs CPU, every shape, including
/// the long-K cancellation case and the shape the WMMA policy pads.
#[cfg(feature = "cuda")]
#[test]
fn matmul_wide_half_cuda_matches_reference_and_cpu() {
    use crate::backend_parity::helpers::with_cuda_backend;
    use numr::runtime::cuda::CudaRuntime;

    let (cpu_client, cpu_device) = create_cpu_client();
    let mut all = cases();
    all.push(Case::new(&[8, 4096], &[4096, 24], "long-K cancellation"));

    for case in all {
        let a_len: usize = case.a_shape.iter().product();
        let b_len: usize = case.b_shape.iter().product();
        let want = reference(
            &grid(a_len, 1),
            &case.a_shape,
            &grid(b_len, 2),
            &case.b_shape,
            &case.out_shape(),
        );
        let (rtol, atol) = f32_tolerance(case.k());
        for dtype in half_dtypes() {
            if !is_dtype_supported("cuda", dtype) {
                continue;
            }
            let cpu = run_case_on::<CpuRuntime>(&case, dtype, &cpu_device, &cpu_client, "CPU");
            with_cuda_backend(|client, device| {
                let cuda = run_case_on::<CudaRuntime>(&case, dtype, &device, &client, "CUDA");
                assert_allclose_f64(
                    &as_f64(&cuda),
                    &want,
                    rtol,
                    atol,
                    &format!(
                        "matmul_wide CUDA vs F64 reference: {} [{dtype:?}]",
                        case.label
                    ),
                );
                // Both sides hold F32 accumulators of the same exact terms in
                // a different order, so the same F32-class bound applies.
                assert_allclose_f64(
                    &as_f64(&cuda),
                    &as_f64(&cpu),
                    rtol,
                    atol,
                    &format!("matmul_wide CUDA vs CPU: {} [{dtype:?}]", case.label),
                );
            });
        }
    }
}

/// CUDA I8 → I32 and F32/F64 delegate to `matmul` there too.
#[cfg(feature = "cuda")]
#[test]
fn matmul_wide_cuda_non_half_matches_matmul() {
    use crate::backend_parity::helpers::with_cuda_backend;
    use numr::runtime::cuda::CudaRuntime;

    with_cuda_backend(|client, device| {
        let a_data: Vec<i8> = (0..37 * 24).map(|i| ((i * 7) % 255) as i8).collect();
        let b_data: Vec<i8> = (0..24 * 35).map(|i| ((i * 11) % 255) as i8).collect();
        let a = Tensor::<CudaRuntime>::from_slice(&a_data, &[37, 24], &device).expect("A");
        let b = Tensor::<CudaRuntime>::from_slice(&b_data, &[24, 35], &device).expect("B");
        let wide = client.matmul_wide(&a, &b).expect("matmul_wide i8");
        let plain = client.matmul(&a, &b).expect("matmul i8");
        assert_eq!(wide.dtype(), DType::I32);
        assert_eq!(wide.to_vec::<i32>(), plain.to_vec::<i32>());

        let a = Tensor::<CudaRuntime>::from_slice(&grid(37 * 24, 1), &[37, 24], &device)
            .expect("A f64");
        let b = Tensor::<CudaRuntime>::from_slice(&grid(24 * 35, 2), &[24, 35], &device)
            .expect("B f64");
        let wide = client.matmul_wide(&a, &b).expect("matmul_wide f64");
        let plain = client.matmul(&a, &b).expect("matmul f64");
        assert_eq!(wide.dtype(), DType::F64);
        assert_eq!(wide.to_vec::<f64>(), plain.to_vec::<f64>());
    });
}
