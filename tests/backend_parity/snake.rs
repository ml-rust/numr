// Backend parity tests for the fused Snake activation (ActivationOps trait).
//
// Forward: every backend against a reference composed from existing ops in F64
// on CPU, `x + sin(alpha x)^2 / (beta + eps)`, for every float dtype the build
// carries and for a spread of shapes and channel axes.
// Backward: CPU F64 against central finite differences for `x`, `alpha` and
// `beta`, then CUDA against CPU for all three gradients.
// Half dtypes: an argument large enough that a half-precision `alpha * x`
// visibly drifts must still land on the F64 reference, which shows the kernel
// computes in F32 and rounds once.

use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "cuda")]
use crate::common::is_dtype_supported;
#[cfg(feature = "f16")]
use crate::common::tolerance_for_dtype;
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

const EPS: f64 = 1e-9;

/// One forward case: `x` of `shape`, channel axis `dim`, parameters of length
/// `shape[dim]`.
struct Case {
    shape: Vec<usize>,
    dim: isize,
    x: Vec<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

fn channel_axis(shape: &[usize], dim: isize) -> usize {
    if dim >= 0 {
        dim as usize
    } else {
        (shape.len() as isize + dim) as usize
    }
}

/// Deterministic, non-degenerate data: `x` sweeps a few radians, the
/// parameters stay positive and away from each other.
fn case(shape: &[usize], dim: isize, shared_params: bool) -> Case {
    let numel: usize = shape.iter().product();
    let channels = shape[channel_axis(shape, dim)];
    let x = (0..numel)
        .map(|i| ((i as f64) * 0.37 - 2.0) * (if i % 3 == 0 { 1.0 } else { -0.8 }))
        .collect();
    let alpha: Vec<f64> = (0..channels).map(|c| 0.3 + 0.45 * c as f64).collect();
    let beta = if shared_params {
        alpha.clone()
    } else {
        (0..channels).map(|c| 0.7 + 0.25 * c as f64).collect()
    };
    Case {
        shape: shape.to_vec(),
        dim,
        x,
        alpha,
        beta,
    }
}

fn cases() -> Vec<Case> {
    vec![
        case(&[2, 5, 7], 1, false),
        case(&[3, 4], -1, false),
        case(&[6], 0, false),
        case(&[2, 3, 4, 5], 2, false),
        case(&[2, 5, 7], 1, true),
    ]
}

/// `x + sin(alpha x)^2 / (beta + eps)` from existing ops in F64 on CPU.
fn reference(
    client: &CpuClient,
    x: &Tensor<CpuRuntime>,
    alpha: &Tensor<CpuRuntime>,
    beta: &Tensor<CpuRuntime>,
    dim: isize,
) -> Vec<f64> {
    let shape = x.shape();
    let axis = channel_axis(shape, dim);
    let mut pshape = vec![1usize; shape.len()];
    pshape[axis] = shape[axis];
    let alpha_b = alpha.reshape(&pshape).unwrap();
    let beta_b = beta.reshape(&pshape).unwrap();
    let s = client.sin(&client.mul(x, &alpha_b).unwrap()).unwrap();
    let s2 = client.mul(&s, &s).unwrap();
    let denom = client.add_scalar(&beta_b, EPS).unwrap();
    let term = client.div(&s2, &denom).unwrap();
    client.add(x, &term).unwrap().to_vec()
}

/// The inputs rounded to `dtype` and read back as F64, so the reference sees
/// exactly what the kernel sees.
fn rounded_inputs(
    client: &CpuClient,
    device: &CpuDevice,
    c: &Case,
    dtype: DType,
) -> (
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Vec<f64>,
) {
    let channels = c.alpha.len();
    let x = tensor_from_f64::<CpuRuntime>(&c.x, &c.shape, dtype, device, client).unwrap();
    let alpha =
        tensor_from_f64::<CpuRuntime>(&c.alpha, &[channels], dtype, device, client).unwrap();
    let beta = tensor_from_f64::<CpuRuntime>(&c.beta, &[channels], dtype, device, client).unwrap();
    let x64 = client.cast(&x, DType::F64).unwrap();
    let alpha64 = client.cast(&alpha, DType::F64).unwrap();
    let beta64 = client.cast(&beta, DType::F64).unwrap();
    let want = reference(client, &x64, &alpha64, &beta64, c.dim);
    (x, alpha, beta, want)
}

fn expected_tensor(
    client: &CpuClient,
    device: &CpuDevice,
    want: &[f64],
    shape: &[usize],
    dtype: DType,
) -> Tensor<CpuRuntime> {
    tensor_from_f64::<CpuRuntime>(want, shape, dtype, device, client).unwrap()
}

/// Run `snake_beta` on backend `R` from CPU-side inputs and check it against
/// the F64 reference in the dtype's tolerance.
fn check_forward<R: Runtime<DType = DType>>(
    client: &(impl ActivationOps<R> + TypeConversionOps<R>),
    device: &R::Device,
    c: &Case,
    dtype: DType,
    label: &str,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let (x_cpu, alpha_cpu, beta_cpu, want) = rounded_inputs(&cpu_client, &cpu_device, c, dtype);
    let channels = c.alpha.len();
    let x = tensor_from_f64::<R>(&x_cpu.to_vec_f64(), &c.shape, dtype, device, client).unwrap();
    let alpha =
        tensor_from_f64::<R>(&alpha_cpu.to_vec_f64(), &[channels], dtype, device, client).unwrap();
    let beta =
        tensor_from_f64::<R>(&beta_cpu.to_vec_f64(), &[channels], dtype, device, client).unwrap();
    let out = client
        .snake_beta(&x, &alpha, &beta, c.dim, EPS)
        .unwrap_or_else(|e| panic!("{label} [{dtype:?}] shape {:?}: {e}", c.shape));
    assert_eq!(out.shape(), c.shape.as_slice(), "{label}: output shape");
    let expected = expected_tensor(&cpu_client, &cpu_device, &want, &c.shape, dtype);
    assert_tensor_allclose(
        &out,
        &expected,
        dtype,
        &format!("{label} [{dtype:?}] shape {:?} dim {}", c.shape, c.dim),
    );
}

/// Read any float tensor back as F64 through its own backend's cast.
trait ToVecF64 {
    fn to_vec_f64(&self) -> Vec<f64>;
}

impl ToVecF64 for Tensor<CpuRuntime> {
    fn to_vec_f64(&self) -> Vec<f64> {
        let (client, _) = create_cpu_client();
        client.cast(self, DType::F64).unwrap().to_vec()
    }
}

#[test]
fn snake_beta_cpu_matches_composed_reference() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        for c in cases() {
            let (client, device) = create_cpu_client();
            check_forward::<CpuRuntime>(&client, &device, &c, dtype, "snake_beta cpu");
        }
    }
}

#[test]
fn snake_beta_cpu_accepts_non_contiguous_input() {
    let (client, device) = create_cpu_client();
    let c = case(&[2, 5, 7], 1, false);
    let x = Tensor::<CpuRuntime>::from_slice(&c.x, &[7, 5, 2], &device).unwrap();
    let alpha = Tensor::<CpuRuntime>::from_slice(&c.alpha, &[5], &device).unwrap();
    let beta = Tensor::<CpuRuntime>::from_slice(&c.beta, &[5], &device).unwrap();
    let x_t = x.permute(&[2, 1, 0]).unwrap();
    assert!(!x_t.is_contiguous());
    let out = client.snake_beta(&x_t, &alpha, &beta, 1, EPS).unwrap();
    let x_c = x_t.contiguous().unwrap();
    let want = reference(&client, &x_c, &alpha, &beta, 1);
    let got: Vec<f64> = out.to_vec();
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-12, "element {i}: {g} vs {w}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn snake_beta_cuda_matches_composed_reference() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cuda") {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        for c in cases() {
            with_cuda_backend(|client, device| {
                check_forward(&client, &device, &c, dtype, "snake_beta cuda");
            });
        }
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn snake_beta_wgpu_matches_composed_reference() {
    for c in cases() {
        with_wgpu_backend_or_skip(|client, device| {
            check_forward(&client, &device, &c, DType::F32, "snake_beta wgpu");
        });
    }
}

// ============================================================================
// Half dtypes: a large `alpha * x` argument
// ============================================================================

/// `alpha * x` near 300 radians with `beta` near 0.01: an F16 product carries
/// a spacing of 0.25 rad, so `sin^2 / beta` drifts by tens of units if the
/// kernel multiplies in half. Computed in F32 it lands on the F64 reference.
fn large_argument_case() -> Case {
    Case {
        shape: vec![2, 2, 3],
        dim: 1,
        x: vec![
            1.001, -0.999, 0.5005, 2.002, 1.75, -1.25, 0.333, 0.667, -0.875, 1.5, 0.125, -0.0625,
        ],
        alpha: vec![300.0, 250.0],
        beta: vec![0.01, 0.02],
    }
}

/// The same case evaluated with `alpha * x` rounded to `dtype` before the
/// sine, the drift a half-precision kernel would produce.
#[cfg(feature = "f16")]
fn half_product_drift(client: &CpuClient, device: &CpuDevice, c: &Case, dtype: DType) -> f64 {
    let (x, alpha, beta, want) = rounded_inputs(client, device, c, dtype);
    let x64: Vec<f64> = client.cast(&x, DType::F64).unwrap().to_vec();
    let a64: Vec<f64> = client.cast(&alpha, DType::F64).unwrap().to_vec();
    let b64: Vec<f64> = client.cast(&beta, DType::F64).unwrap().to_vec();
    let inner = c.shape[2];
    let channels = c.shape[1];
    let mut worst = 0.0f64;
    for (i, xv) in x64.iter().enumerate() {
        let ch = (i / inner) % channels;
        let product = a64[ch] * xv;
        let rounded = match dtype {
            DType::F16 => half::f16::from_f64(product).to_f64(),
            DType::BF16 => half::bf16::from_f64(product).to_f64(),
            other => panic!("not a half dtype: {other:?}"),
        };
        let naive = xv + rounded.sin().powi(2) / (b64[ch] + EPS);
        worst = worst.max((naive - want[i]).abs());
    }
    worst
}

#[test]
fn snake_beta_half_dtypes_compute_in_f32() {
    let half_dtypes: Vec<DType> = parity_dtypes(DTypeDomain::FloatsOnly, "cpu")
        .into_iter()
        .filter(|d| matches!(d, DType::F16 | DType::BF16))
        .collect();
    let c = large_argument_case();
    for dtype in half_dtypes {
        let (cpu_client, cpu_device) = create_cpu_client();
        #[cfg(feature = "f16")]
        {
            let (rtol, atol) = tolerance_for_dtype(dtype);
            let drift = half_product_drift(&cpu_client, &cpu_device, &c, dtype);
            // The case only proves something if a half-precision product
            // would fail the tolerance the kernel is held to.
            let (_, _, _, want) = rounded_inputs(&cpu_client, &cpu_device, &c, dtype);
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(
                drift > atol + rtol * scale,
                "[{dtype:?}] a half-precision product only drifts {drift:.3}; pick a larger argument"
            );
        }
        check_forward::<CpuRuntime>(&cpu_client, &cpu_device, &c, dtype, "snake_beta half cpu");
        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                check_forward(&client, &device, &c, dtype, "snake_beta half cuda");
            });
        }
    }
}

// ============================================================================
// Backward
// ============================================================================

/// `(input tensor, its host values, analytic gradient, label)` for one finite-difference sweep.
type ParamCheck<'a> = (
    &'a Tensor<CpuRuntime>,
    &'a Vec<f64>,
    &'a Tensor<CpuRuntime>,
    &'a str,
);

fn sum_snake(
    client: &CpuClient,
    x: &Tensor<CpuRuntime>,
    alpha: &Tensor<CpuRuntime>,
    beta: &Tensor<CpuRuntime>,
    dim: isize,
) -> f64 {
    client
        .snake_beta(x, alpha, beta, dim, EPS)
        .unwrap()
        .to_vec::<f64>()
        .iter()
        .sum()
}

#[test]
fn snake_beta_bwd_cpu_f64_matches_finite_differences() {
    let (client, device) = create_cpu_client();
    for c in [case(&[2, 5, 7], 1, false), case(&[2, 3, 4, 5], 2, false)] {
        let channels = c.alpha.len();
        let x = Tensor::<CpuRuntime>::from_slice(&c.x, &c.shape, &device).unwrap();
        let alpha = Tensor::<CpuRuntime>::from_slice(&c.alpha, &[channels], &device).unwrap();
        let beta = Tensor::<CpuRuntime>::from_slice(&c.beta, &[channels], &device).unwrap();
        // Loss = sum(snake_beta), so the upstream gradient is all ones.
        let grad = Tensor::<CpuRuntime>::ones(&c.shape, DType::F64, &device).unwrap();
        let (d_x, d_alpha, d_beta) = client
            .snake_beta_bwd(&grad, &x, &alpha, &beta, c.dim, EPS)
            .unwrap();
        assert_eq!(d_x.shape(), c.shape.as_slice());
        assert_eq!(d_alpha.shape(), &[channels]);
        assert_eq!(d_beta.shape(), &[channels]);

        let h = 1e-6;
        let inputs: [ParamCheck<'_>; 3] = [
            (&x, &c.x, &d_x, "d_x"),
            (&alpha, &c.alpha, &d_alpha, "d_alpha"),
            (&beta, &c.beta, &d_beta, "d_beta"),
        ];
        for (which, (tensor, values, analytic, name)) in inputs.iter().enumerate() {
            let got: Vec<f64> = analytic.to_vec();
            for i in 0..values.len() {
                let eval = |delta: f64| {
                    let mut v = (*values).clone();
                    v[i] += delta;
                    let t = Tensor::<CpuRuntime>::from_slice(&v, tensor.shape(), &device).unwrap();
                    match which {
                        0 => sum_snake(&client, &t, &alpha, &beta, c.dim),
                        1 => sum_snake(&client, &x, &t, &beta, c.dim),
                        _ => sum_snake(&client, &x, &alpha, &t, c.dim),
                    }
                };
                let numeric = (eval(h) - eval(-h)) / (2.0 * h);
                assert!(
                    (got[i] - numeric).abs() < 1e-6 * (1.0 + numeric.abs()),
                    "{name}[{i}] shape {:?}: analytic {} vs numeric {numeric}",
                    c.shape,
                    got[i]
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn snake_beta_bwd_cuda_matches_cpu() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cuda") {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        for c in cases() {
            let (cpu_client, cpu_device) = create_cpu_client();
            let (x, alpha, beta, _) = rounded_inputs(&cpu_client, &cpu_device, &c, dtype);
            let channels = c.alpha.len();
            let grad_data: Vec<f64> = (0..c.x.len()).map(|i| 0.5 + (i % 5) as f64 * 0.3).collect();
            let grad = tensor_from_f64::<CpuRuntime>(
                &grad_data,
                &c.shape,
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap();
            let (dx_cpu, da_cpu, db_cpu) = cpu_client
                .snake_beta_bwd(&grad, &x, &alpha, &beta, c.dim, EPS)
                .unwrap();

            with_cuda_backend(|client, device| {
                let up = |t: &Tensor<CpuRuntime>| {
                    tensor_from_f64(&t.to_vec_f64(), t.shape(), dtype, &device, &client).unwrap()
                };
                let (dx, da, db) = client
                    .snake_beta_bwd(&up(&grad), &up(&x), &up(&alpha), &up(&beta), c.dim, EPS)
                    .unwrap_or_else(|e| panic!("snake_beta_bwd cuda [{dtype:?}]: {e}"));
                let label = format!("snake_beta_bwd cuda [{dtype:?}] shape {:?}", c.shape);
                assert_tensor_allclose(&dx, &dx_cpu, dtype, &format!("{label} d_x"));
                assert_tensor_allclose(&da, &da_cpu, dtype, &format!("{label} d_alpha"));
                assert_tensor_allclose(&db, &db_cpu, dtype, &format!("{label} d_beta"));
                assert_eq!(da.shape(), &[channels]);
            });
        }
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn snake_beta_bwd_wgpu_matches_cpu() {
    let dtype = DType::F32;
    for c in cases() {
        let (cpu_client, cpu_device) = create_cpu_client();
        let (x, alpha, beta, _) = rounded_inputs(&cpu_client, &cpu_device, &c, dtype);
        let grad_data: Vec<f64> = (0..c.x.len()).map(|i| 0.5 + (i % 5) as f64 * 0.3).collect();
        let grad =
            tensor_from_f64::<CpuRuntime>(&grad_data, &c.shape, dtype, &cpu_device, &cpu_client)
                .unwrap();
        let (dx_cpu, da_cpu, db_cpu) = cpu_client
            .snake_beta_bwd(&grad, &x, &alpha, &beta, c.dim, EPS)
            .unwrap();

        with_wgpu_backend_or_skip(|client, device| {
            let up = |t: &Tensor<CpuRuntime>| {
                tensor_from_f64(&t.to_vec_f64(), t.shape(), dtype, &device, &client).unwrap()
            };
            let (dx, da, db) = client
                .snake_beta_bwd(&up(&grad), &up(&x), &up(&alpha), &up(&beta), c.dim, EPS)
                .unwrap_or_else(|e| panic!("snake_beta_bwd wgpu: {e}"));
            let label = format!("snake_beta_bwd wgpu shape {:?}", c.shape);
            assert_tensor_allclose(&dx, &dx_cpu, dtype, &format!("{label} d_x"));
            assert_tensor_allclose(&da, &da_cpu, dtype, &format!("{label} d_alpha"));
            assert_tensor_allclose(&db, &db_cpu, dtype, &format!("{label} d_beta"));
        });
    }
}
