// Backend parity tests for a rank-1 `b` operand.
//
// `matmul_output_shape` treats a `[k]` right operand as a `[k, 1]` column, so
// `[m, k] @ [k]` is `[m, 1]` and `[3, m, k] @ [k]` is `[3, m, 1]`. Every
// backend must derive its kernel geometry from the same rule
// (`numr::ops::matmul::matmul_mkn`): reading `n` off the last dim of `b`
// gives `n == k`, and the kernel then writes `m * k` values into an
// `m`-element output. These cases check the shape, the values against an F64
// dot product, and CUDA against CPU, through `matmul`, `matmul_bias` (bias
// `[1]`) and `matmul_wide`.

use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::cpu::CpuRuntime;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{create_cpu_client, is_dtype_supported, values_close};

/// Which entry point a case runs.
#[derive(Clone, Copy)]
enum Entry {
    Matmul,
    MatmulBias,
    MatmulWide,
}

impl Entry {
    fn name(self) -> &'static str {
        match self {
            Entry::Matmul => "matmul",
            Entry::MatmulBias => "matmul_bias",
            Entry::MatmulWide => "matmul_wide",
        }
    }
}

fn grid(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as f64) * 0.0137 + seed as f64 * 0.7).sin() * 0.5)
        .collect()
}

/// `[.., m, 1]` result of `a @ b (+ bias)` summed in F64.
fn reference(a: &[f64], b: &[f64], bias: f64, batch: usize, m: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * m];
    for bi in 0..batch {
        for i in 0..m {
            let row = &a[(bi * m + i) * k..(bi * m + i + 1) * k];
            out[bi * m + i] = row.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() + bias;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_on<R>(
    entry: Entry,
    dtype: DType,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    bias: &[f64],
    device: &R::Device,
    client: &(impl MatmulOps<R> + numr::ops::TypeConversionOps<R>),
    backend: &str,
) -> (Vec<usize>, Vec<f64>)
where
    R: numr::runtime::Runtime<DType = DType>,
{
    let k = b.len();
    let label = format!("{backend} {} [{dtype:?}] a={a_shape:?}", entry.name());
    let a_t = tensor_from_f64(a, a_shape, dtype, device, client)
        .unwrap_or_else(|e| panic!("{label}: A failed: {e}"));
    let b_t = tensor_from_f64(b, &[k], dtype, device, client)
        .unwrap_or_else(|e| panic!("{label}: B failed: {e}"));
    let out = match entry {
        Entry::Matmul => client.matmul(&a_t, &b_t),
        Entry::MatmulBias => {
            let bias_t = tensor_from_f64(bias, &[1], dtype, device, client)
                .unwrap_or_else(|e| panic!("{label}: bias failed: {e}"));
            client.matmul_bias(&a_t, &b_t, &bias_t)
        }
        Entry::MatmulWide => client.matmul_wide(&a_t, &b_t),
    }
    .unwrap_or_else(|e| panic!("{label}: op failed: {e}"));
    let values = client
        .cast(&out, DType::F64)
        .unwrap_or_else(|e| panic!("{label}: cast failed: {e}"))
        .to_vec::<f64>();
    (out.shape().to_vec(), values)
}

fn tolerance(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F32 => (1e-5, 1e-6),
        // Half operands, F32 sums; the stored result rounds once.
        _ => (1e-2, 1e-3),
    }
}

fn assert_rank1_b(entry: Entry, dtype: DType, a_shape: &[usize]) {
    let k = a_shape[a_shape.len() - 1];
    let m = a_shape[a_shape.len() - 2];
    let batch: usize = a_shape[..a_shape.len() - 2].iter().product();
    let a = grid(batch * m * k, 1);
    let b = grid(k, 2);
    let bias = [0.25f64];
    let bias_val = match entry {
        Entry::MatmulBias => bias[0],
        _ => 0.0,
    };
    let want = reference(&a, &b, bias_val, batch, m, k);
    let mut want_shape = a_shape.to_vec();
    let last = want_shape.len() - 1;
    want_shape[last] = 1;
    let (rtol, atol) = tolerance(dtype);

    let (cpu_client, cpu_device) = create_cpu_client();
    let (cpu_shape, cpu) = run_on::<CpuRuntime>(
        entry,
        dtype,
        &a,
        a_shape,
        &b,
        &bias,
        &cpu_device,
        &cpu_client,
        "CPU",
    );
    assert_eq!(
        cpu_shape,
        want_shape,
        "CPU {} [{dtype:?}] a={a_shape:?}: output shape",
        entry.name()
    );
    for (i, (g, w)) in cpu.iter().zip(&want).enumerate() {
        assert!(
            values_close(*g, *w, rtol, atol),
            "CPU {} [{dtype:?}] a={a_shape:?} at {i}: {g} vs F64 {w}",
            entry.name()
        );
    }

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        use numr::runtime::cuda::CudaRuntime;
        with_cuda_backend(|client, device| {
            let (shape, cuda) = run_on::<CudaRuntime>(
                entry, dtype, &a, a_shape, &b, &bias, &device, &client, "CUDA",
            );
            assert_eq!(
                shape,
                want_shape,
                "CUDA {} [{dtype:?}] a={a_shape:?}: output shape",
                entry.name()
            );
            for (i, ((g, w), c)) in cuda.iter().zip(&want).zip(&cpu).enumerate() {
                assert!(
                    values_close(*g, *w, rtol, atol),
                    "CUDA {} [{dtype:?}] a={a_shape:?} at {i}: {g} vs F64 {w}",
                    entry.name()
                );
                assert!(
                    values_close(*g, *c, rtol, atol),
                    "CUDA {} [{dtype:?}] a={a_shape:?} at {i}: {g} vs CPU {c}",
                    entry.name()
                );
            }
        });
    }
    #[cfg(not(feature = "cuda"))]
    let _ = is_dtype_supported;
}

fn dtypes() -> Vec<DType> {
    [DType::F32, DType::F16]
        .into_iter()
        .filter(|&d| is_dtype_supported("cpu", d))
        .collect()
}

/// `[m, k] @ [k]` and `[3, m, k] @ [k]`, F32 and F16, all three entry points.
/// `m = 37, k = 21` is ragged on every axis and light enough that nothing pads.
#[test]
fn matmul_rank1_b_all_entries_match_reference_and_cpu() {
    for entry in [Entry::Matmul, Entry::MatmulBias, Entry::MatmulWide] {
        for dtype in dtypes() {
            assert_rank1_b(entry, dtype, &[37, 21]);
            assert_rank1_b(entry, dtype, &[3, 37, 21]);
        }
    }
}

/// A long `k` with a small `m`, the shape the GEMV kernels take.
#[test]
fn matmul_rank1_b_gemv_shape_matches_reference_and_cpu() {
    for entry in [Entry::Matmul, Entry::MatmulBias, Entry::MatmulWide] {
        for dtype in dtypes() {
            assert_rank1_b(entry, dtype, &[2, 1024]);
            assert_rank1_b(entry, dtype, &[3, 2, 1024]);
        }
    }
}
