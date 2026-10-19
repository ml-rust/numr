// Backend parity for argmax and argmin on rows full of exact ties.
//
// The contract, set by `cpu/kernels/reduce/special.rs`:
// - Equal values resolve to the lowest index.
// - NaN never wins. A row of only NaN returns index 0.
//
// A GPU thread scans indices tid, tid + 256, ..., so the cases put the lowest
// tied index in a higher thread than a later duplicate. A merge that compares
// values only returns the later duplicate there.
//
// Every case checks the CPU result against a host reference built from the
// tensor's own values, then checks CUDA and WebGPU against CPU.

use numr::dtype::DType;
use numr::ops::{IndexingOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::{tensor_from_f32, tensor_from_f64};
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{DTypeDomain, create_cpu_client, is_dtype_supported, parity_dtypes};

fn is_fp8(dtype: DType) -> bool {
    matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2)
}

/// Dtypes that hold `data` with its order intact. FP8 has no infinity.
fn dtypes_for(data: &[f64]) -> Vec<DType> {
    let finite = data.iter().all(|v| v.is_finite());
    let small_int = finite && data.iter().all(|v| v.fract() == 0.0 && v.abs() <= 100.0);
    let non_negative = data.iter().all(|&v| v >= 0.0);
    parity_dtypes(DTypeDomain::AllNumeric, "cpu")
        .into_iter()
        .filter(|&dt| {
            if is_fp8(dt) {
                finite
            } else if dt.is_float() {
                true
            } else if dt.is_signed_int() {
                small_int
            } else {
                small_int && non_negative
            }
        })
        .collect()
}

/// Index tensor as i64. WebGPU returns I32 indices, the other backends I64.
fn indices<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Vec<i64> {
    let t = t.contiguous().expect("index tensor contiguous");
    match t.dtype() {
        DType::I64 => t.to_vec::<i64>(),
        DType::I32 => t.to_vec::<i32>().into_iter().map(i64::from).collect(),
        other => panic!("argreduce returned dtype {other:?}, expected I64 or I32"),
    }
}

/// Lowest index of the best non-NaN value per lane, or 0 when every value is NaN.
fn reference(values: &[f64], shape: &[usize], dim: usize, is_max: bool) -> Vec<i64> {
    let outer: usize = shape[..dim].iter().product();
    let reduce = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut best: Option<(f64, usize)> = None;
            for r in 0..reduce {
                let v = values[o * reduce * inner + r * inner + i];
                if v.is_nan() {
                    continue;
                }
                let wins = match best {
                    None => true,
                    Some((b, _)) => (is_max && v > b) || (!is_max && v < b),
                };
                if wins {
                    best = Some((v, r));
                }
            }
            out.push(best.map_or(0, |(_, r)| r as i64));
        }
    }
    out
}

/// Build a tensor of `dtype` from `data`.
///
/// Narrow floats round through F32, so every backend runs the same F32 cast.
fn make<R: Runtime<DType = DType>>(
    data: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
    client: &impl TypeConversionOps<R>,
) -> numr::error::Result<Tensor<R>> {
    if matches!(dtype, DType::F16 | DType::BF16) || is_fp8(dtype) {
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        tensor_from_f32(&f32_data, shape, dtype, device, client)
    } else {
        tensor_from_f64(data, shape, dtype, device, client)
    }
}

/// Run argmax and argmin over `dim` on every backend and dtype that fits `data`.
fn check_case(name: &str, data: &[f64], shape: &[usize], dim: usize) {
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in dtypes_for(data) {
        let cpu_t = make(data, shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("{name}: CPU tensor {dtype:?}: {e}"));
        // The reference reads the stored values, so narrow-float rounding is included.
        let stored = cpu_client
            .cast(&cpu_t, DType::F64)
            .unwrap_or_else(|e| panic!("{name}: CPU cast {dtype:?} to F64: {e}"))
            .to_vec::<f64>();

        for is_max in [true, false] {
            let op = if is_max { "argmax" } else { "argmin" };
            let ctx = format!("{name}: {op} {dtype:?} shape {shape:?} dim {dim}");
            let cpu_out = if is_max {
                cpu_client.argmax(&cpu_t, dim, false)
            } else {
                cpu_client.argmin(&cpu_t, dim, false)
            };
            let cpu_idx = indices(&cpu_out.unwrap_or_else(|e| panic!("{ctx}: CPU: {e}")));
            assert_eq!(
                cpu_idx,
                reference(&stored, shape, dim, is_max),
                "{ctx}: CPU vs reference"
            );

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|client, device| {
                    let t = make(data, shape, dtype, &device, &client)
                        .unwrap_or_else(|e| panic!("{ctx}: CUDA tensor: {e}"));
                    let out = if is_max {
                        client.argmax(&t, dim, false)
                    } else {
                        client.argmin(&t, dim, false)
                    };
                    let out = out.unwrap_or_else(|e| panic!("{ctx}: CUDA: {e}"));
                    assert_eq!(indices(&out), cpu_idx, "{ctx}: CUDA vs CPU");
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|client, device| {
                    let t = make(data, shape, dtype, &device, &client)
                        .unwrap_or_else(|e| panic!("{ctx}: WGPU tensor: {e}"));
                    let out = if is_max {
                        client.argmax(&t, dim, false)
                    } else {
                        client.argmin(&t, dim, false)
                    };
                    let out = out.unwrap_or_else(|e| panic!("{ctx}: WGPU: {e}"));
                    assert_eq!(indices(&out), cpu_idx, "{ctx}: WGPU vs CPU");
                });
            }
        }
    }
}

/// Row of `len` values in 10..=16, with 50 at every `peaks` id and 1 at every `troughs` id.
fn spiked_row(len: usize, peaks: &[usize], troughs: &[usize]) -> Vec<f64> {
    let mut row: Vec<f64> = (0..len).map(|i| (i % 7) as f64 + 10.0).collect();
    for &p in peaks {
        row[p] = 50.0;
    }
    for &t in troughs {
        row[t] = 1.0;
    }
    row
}

#[test]
fn test_argreduce_all_equal_rows() {
    check_case("all equal", &[7.0; 3 * 1000], &[3, 1000], 1);
    check_case("all equal, dim 0", &[7.0; 300 * 4], &[300, 4], 0);
    check_case("all equal, negative", &[-3.0; 2 * 513], &[2, 513], 1);
}

#[test]
fn test_argreduce_strided_ties_1000() {
    // Indices 1, 257, 513 all sit in thread 1.
    check_case(
        "1, 257, 513",
        &spiked_row(1000, &[1, 257, 513], &[1, 257, 513]),
        &[1000],
        0,
    );
    // 256 is thread 0 and 255 is thread 255. The lowest tie, 3, is thread 3.
    check_case(
        "3 behind 256",
        &spiked_row(1000, &[256, 255, 3], &[512, 300, 44]),
        &[1000],
        0,
    );
    // Five equal extrema. 300 is thread 44, the lowest tie 45 is thread 45.
    let five = [700, 300, 45, 513, 999];
    check_case("five ties", &spiked_row(1000, &five, &five), &[1000], 0);
}

#[test]
fn test_argreduce_strided_ties_vocab_row() {
    // One Llama 3 vocabulary row. Max: 515 is thread 3, the lowest tie 77 is
    // thread 77. Min: 70144 is thread 0, the lowest tie 600 is thread 88.
    let len = 128256;
    let peaks = [128000, 100355, 515, 1000, 77];
    let troughs = [128255, 70144, 600, 1001];
    check_case("vocab row", &spiked_row(len, &peaks, &troughs), &[len], 0);

    let mut batch = spiked_row(len, &peaks, &troughs);
    batch.extend(spiked_row(len, &[1, 257, 513], &[600, 344, 88]));
    check_case("vocab batch", &batch, &[2, len], 1);
}

#[test]
fn test_argreduce_negative_ties() {
    // Max -3 and min -7 each repeat every 5 elements.
    let row: Vec<f64> = (0..1000).map(|i| -((i % 5) as f64) - 3.0).collect();
    check_case("negative", &row, &[1000], 0);
    check_case("negative, batched", &row, &[4, 250], 1);
    check_case("negative, dim 0", &row, &[250, 4], 0);
}

#[test]
fn test_argreduce_ties_along_each_dim() {
    // Values repeat with period 3, so every lane holds ties.
    let shape = [4, 300, 3];
    let data: Vec<f64> = (0..shape.iter().product::<usize>())
        .map(|i| ((i * 7) % 3) as f64)
        .collect();
    for dim in 0..shape.len() {
        check_case("period 3", &data, &shape, dim);
    }
}

#[test]
fn test_argreduce_nonfinite_rows() {
    let n = 600;
    let neg_inf = vec![f64::NEG_INFINITY; n];
    check_case("all -inf", &neg_inf, &[n], 0);

    let pos_inf = vec![f64::INFINITY; n];
    check_case("all +inf", &pos_inf, &[n], 0);

    // NaN at index 0 never wins, so the first -inf (index 1) is the argmax.
    let mut nan_head = vec![f64::NEG_INFINITY; n];
    nan_head[0] = f64::NAN;
    nan_head[300] = f64::NAN;
    check_case("NaN head, -inf rest", &nan_head, &[n], 0);

    // NaN spread through a tied row. The answer stays the lowest finite tie.
    let mut nan_ties = spiked_row(n, &[257, 2], &[258, 3]);
    for i in [0, 1, 256, 512] {
        nan_ties[i] = f64::NAN;
    }
    check_case("NaN with ties", &nan_ties, &[n], 0);

    check_case("all NaN", &vec![f64::NAN; n], &[2, n / 2], 1);
}

#[test]
fn test_argreduce_narrow_float_rounding_ties() {
    // 1 + i * 1e-4 collapses to a few F16 and BF16 values, so each value repeats
    // across many threads. F32 and F64 keep the values distinct.
    let data: Vec<f64> = (0..2 * 1000)
        .map(|i| 1.0 + (i % 1000) as f64 * 1e-4)
        .collect();
    check_case("rounding ties", &data, &[2, 1000], 1);

    let rev: Vec<f64> = data.iter().rev().copied().collect();
    check_case("rounding ties, descending", &rev, &[2, 1000], 1);
}
