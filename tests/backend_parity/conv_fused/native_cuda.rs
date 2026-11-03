// CUDA native `conv2d_fused`: the implicit-GEMM tensor-core kernel for F16
// and BF16 NHWC with one group.
//
// Every case rounds its operands to the half dtype first, then compares the
// CUDA result against the CPU result computed in F64 from the same values.
// The kernel accumulates in F32 and rounds once at the store, so the
// tolerance is one storage ulp relative plus a small absolute floor for the
// F32 sum (`native_tolerance`).
//
// Cases cover the one-element staging path (C_in 3, 12) and the 128-bit path
// (C_in 8, 16, 64), every tile width (C_out 32, 40, 48, 64, 130), ragged
// spatial sizes, stride, dilation, Same and asymmetric padding, every
// activation, bias on and off, and residual scales 0.2 and 1.0. Routing is
// covered by native-vs-generic agreement on CUDA and by a grouped case that
// takes the generic path.

use numr::dtype::DType;
use numr::ops::{
    Conv2dConfig, ConvActivation, ConvLayout, ConvResidual, FusedConvOps, PaddingMode,
    TypeConversionOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use super::{ACTS, Act, LEAKY_SLOPE, pattern, slopes};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{create_cpu_client, gemm_long_k_tolerance, half_dtypes};

/// `(rtol, atol)` of the native kernel against the F64 reference.
///
/// The F32 result rounds once to the storage dtype: at most half a storage
/// ulp, which is under `2^-11 |y|` for F16 and `2^-8 |y|` for BF16. `rtol`
/// is one full ulp, twice that bound. `atol` covers the F32 accumulation
/// error, which stays far below it at these contraction lengths.
pub(super) fn native_tolerance(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F16 => (2f64.powi(-10), 1e-3),
        DType::BF16 => (2f64.powi(-7), 1e-3),
        other => panic!("native_tolerance: {other:?} is not a half dtype"),
    }
}

/// `values` rounded to `dtype` and widened back to F64.
pub(super) fn rounded(values: Vec<f64>, dtype: DType) -> Vec<f64> {
    let (cpu, device) = create_cpu_client();
    let n = values.len();
    let t = tensor_from_f64(&values, &[n], dtype, &device, &cpu).unwrap();
    cpu.cast(&t, DType::F64).unwrap().to_vec::<f64>()
}

/// Every element of `t` as F64. F16 and BF16 widen exactly through F32.
pub(super) fn host_f64<R, C>(client: &C, t: &Tensor<R>) -> Vec<f64>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    let wide = client.cast(&t.contiguous().unwrap(), DType::F32).unwrap();
    wide.to_vec::<f32>().into_iter().map(f64::from).collect()
}

/// Geometry of one case. Layout is NHWC, weight OHWI.
#[derive(Clone, Copy, Debug)]
pub(super) struct Spec {
    pub name: &'static str,
    pub batch: usize,
    pub h: usize,
    pub w: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel: usize,
    pub stride: (usize, usize),
    pub padding: PaddingMode,
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Spec {
    pub(super) fn config(self) -> Conv2dConfig {
        Conv2dConfig::new(ConvLayout::Nhwc)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_dilation(self.dilation)
            .with_groups(self.groups)
    }

    fn input_shape(self) -> [usize; 4] {
        [self.batch, self.h, self.w, self.c_in]
    }

    fn weight_shape(self) -> [usize; 4] {
        [
            self.c_out,
            self.kernel,
            self.kernel,
            self.c_in / self.groups,
        ]
    }

    fn contraction(self) -> usize {
        self.kernel * self.kernel * self.c_in / self.groups
    }
}

/// Bias, activation and residual of one call.
#[derive(Clone, Copy, Debug)]
pub(super) struct Epilogue {
    pub act: Act,
    pub bias: bool,
    pub residual_scale: Option<f64>,
}

/// Every epilogue: each activation, bias off and on, no residual and
/// residual scales 0.2 and 1.0.
fn all_epilogues() -> Vec<Epilogue> {
    let mut out = Vec::new();
    for act in ACTS {
        for bias in [false, true] {
            for residual_scale in [None, Some(0.2), Some(1.0)] {
                out.push(Epilogue {
                    act,
                    bias,
                    residual_scale,
                });
            }
        }
    }
    out
}

/// Operand values of one spec, rounded to the dtype, in NHWC / OHWI order.
pub(super) struct Values {
    x: Vec<f64>,
    w: Vec<f64>,
    bias: Vec<f64>,
    slope_c: Vec<f64>,
    slope_1: Vec<f64>,
    residual: Vec<f64>,
    out_shape: Vec<usize>,
}

fn values(spec: Spec, dtype: DType) -> Values {
    let x = rounded(pattern(spec.input_shape().iter().product(), 1), dtype);
    let w = rounded(pattern(spec.weight_shape().iter().product(), 2), dtype);
    let (cpu, device) = create_cpu_client();
    let out_shape = cpu
        .conv2d_fused(
            &Tensor::<CpuRuntime>::from_slice(&x, &spec.input_shape(), &device).unwrap(),
            &Tensor::<CpuRuntime>::from_slice(&w, &spec.weight_shape(), &device).unwrap(),
            None,
            &spec.config(),
            ConvActivation::None,
            None,
        )
        .unwrap()
        .shape()
        .to_vec();
    Values {
        bias: rounded(pattern(spec.c_out, 3), dtype),
        slope_c: rounded(slopes(spec.c_out, 4), dtype),
        slope_1: rounded(slopes(1, 5), dtype),
        residual: rounded(pattern(out_shape.iter().product(), 6), dtype),
        out_shape,
        x,
        w,
    }
}

/// `conv2d_fused` with every operand built in `dtype`.
pub(super) fn run<R, C>(
    client: &C,
    device: &R::Device,
    dtype: DType,
    spec: Spec,
    v: &Values,
    epi: Epilogue,
) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: FusedConvOps<R> + TypeConversionOps<R>,
{
    let make = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, device, client).unwrap()
    };
    let w = make(&v.w, &spec.weight_shape());
    run_with_weight(client, device, dtype, spec, v, epi, &w)
}

/// `conv2d_fused` with a caller-built weight, so a test can pass a strided
/// weight view.
fn run_with_weight<R, C>(
    client: &C,
    device: &R::Device,
    dtype: DType,
    spec: Spec,
    v: &Values,
    epi: Epilogue,
    w: &Tensor<R>,
) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: FusedConvOps<R> + TypeConversionOps<R>,
{
    let make = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, device, client).unwrap()
    };
    let x = make(&v.x, &spec.input_shape());
    let bias = make(&v.bias, &[spec.c_out]);
    let slope_c = make(&v.slope_c, &[spec.c_out]);
    let slope_1 = make(&v.slope_1, &[1]);
    let residual = make(&v.residual, &v.out_shape);
    let activation = match epi.act {
        Act::None => ConvActivation::None,
        Act::Relu => ConvActivation::Relu,
        Act::Leaky => ConvActivation::LeakyRelu(LEAKY_SLOPE),
        Act::PreluPerChannel => ConvActivation::Prelu(&slope_c),
        Act::PreluShared => ConvActivation::Prelu(&slope_1),
    };
    let residual = epi.residual_scale.map(|scale| ConvResidual {
        tensor: &residual,
        scale,
    });
    client
        .conv2d_fused(
            &x,
            w,
            epi.bias.then_some(&bias),
            &spec.config(),
            activation,
            residual,
        )
        .unwrap_or_else(|e| panic!("conv2d_fused {} {epi:?} [{dtype:?}]: {e}", spec.name))
}

/// The CPU result in F64 from the same rounded values.
fn reference(spec: Spec, v: &Values, epi: Epilogue) -> Vec<f64> {
    let (cpu, device) = create_cpu_client();
    let out = run(&cpu, &device, DType::F64, spec, v, epi);
    out.to_vec::<f64>()
}

/// Run every epilogue of `spec` on CUDA and compare with the F64 reference.
fn check(spec: Spec, epilogues: &[Epilogue]) {
    for dtype in half_dtypes("cuda") {
        let v = values(spec, dtype);
        let (rtol, atol) = native_tolerance(dtype);
        with_cuda_backend(|client, device| {
            for &epi in epilogues {
                let label = format!("{} {epi:?} [{dtype:?}]", spec.name);
                let got = run(&client, &device, dtype, spec, &v, epi);
                assert_eq!(got.shape(), v.out_shape.as_slice(), "{label}: shape");
                assert!(got.is_contiguous(), "{label}: output must be contiguous");
                let got = host_f64(&client, &got);
                assert_values_close(&got, &reference(spec, &v, epi), rtol, atol, &label);
            }
        });
    }
}

pub(super) fn assert_values_close(
    got: &[f64],
    expected: &[f64],
    rtol: f64,
    atol: f64,
    label: &str,
) {
    assert_eq!(got.len(), expected.len(), "{label}: length");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let tol = atol + rtol * e.abs();
        assert!(
            (g - e).abs() <= tol,
            "{label}: element {i}: CUDA {g} vs F64 reference {e} (diff {:.3e}, tol {tol:.3e})",
            (g - e).abs()
        );
    }
}

const fn spec(name: &'static str, c_in: usize, c_out: usize) -> Spec {
    Spec {
        name,
        batch: 2,
        h: 17,
        w: 23,
        c_in,
        c_out,
        kernel: 3,
        stride: (1, 1),
        padding: PaddingMode::Same,
        dilation: (1, 1),
        groups: 1,
    }
}

/// C_in 3: one-element staging for A and B, 128x32 tile.
#[test]
fn test_native_conv_cin3_all_epilogues() {
    check(spec("cin3_cout32_same", 3, 32), &all_epilogues());
}

/// C_in 64: 128-bit staging for A and B.
#[test]
fn test_native_conv_cin64_all_epilogues() {
    check(spec("cin64_cout64_same", 64, 64), &all_epilogues());
}

/// Stride, dilation, padding and tile-width cases with two epilogues each.
#[test]
fn test_native_conv_geometry() {
    let epilogues = [
        Epilogue {
            act: Act::Leaky,
            bias: true,
            residual_scale: Some(0.2),
        },
        Epilogue {
            act: Act::PreluPerChannel,
            bias: false,
            residual_scale: None,
        },
    ];
    let specs = [
        Spec {
            stride: (2, 2),
            padding: PaddingMode::Valid,
            ..spec("cin64_cout48_stride2_valid", 64, 48)
        },
        Spec {
            dilation: (2, 2),
            ..spec("cin64_cout32_dilation2_same", 64, 32)
        },
        Spec {
            padding: PaddingMode::Custom(2, 0, 1, 3),
            ..spec("cin16_cout64_custom_pad", 16, 64)
        },
        Spec {
            padding: PaddingMode::Valid,
            ..spec("cin12_cout32_valid", 12, 32)
        },
        spec("cin64_cout130_same", 64, 130),
        Spec {
            kernel: 1,
            padding: PaddingMode::Valid,
            ..spec("cin64_cout32_1x1", 64, 32)
        },
        Spec {
            stride: (2, 1),
            dilation: (1, 2),
            padding: PaddingMode::Custom(1, 2, 0, 2),
            ..spec("cin8_cout40_mixed", 8, 40)
        },
    ];
    for s in specs {
        check(s, &epilogues);
    }
}

/// Two runs on the same inputs give bitwise-equal outputs.
#[test]
fn test_native_conv_deterministic() {
    let s = spec("cin64_cout64_same", 64, 64);
    let epi = Epilogue {
        act: Act::Leaky,
        bias: true,
        residual_scale: Some(0.2),
    };
    for dtype in half_dtypes("cuda") {
        let v = values(s, dtype);
        with_cuda_backend(|client, device| {
            let first = host_f64(&client, &run(&client, &device, dtype, s, &v, epi));
            let second = host_f64(&client, &run(&client, &device, dtype, s, &v, epi));
            let same = first
                .iter()
                .zip(&second)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "two runs differ [{dtype:?}]");
        });
    }
}

/// A contiguous OHWI weight takes the native kernel; the same values as a
/// permuted OIHW buffer take the generic path. Both match the reference, and
/// each other within the generic path's tolerance.
#[test]
fn test_native_conv_agrees_with_generic_path() {
    let s = spec("cin64_cout48_routing", 64, 48);
    let epi = Epilogue {
        act: Act::PreluShared,
        bias: true,
        residual_scale: Some(0.2),
    };
    for dtype in half_dtypes("cuda") {
        let v = values(s, dtype);
        let expected = reference(s, &v, epi);
        let (rtol, atol) = native_tolerance(dtype);
        let (_, generic_atol) = gemm_long_k_tolerance(dtype, s.contraction(), 1.0);
        // OHWI values reordered to OIHW, so `permute` gives an OHWI view
        // that is not contiguous.
        let [co_n, kh_n, kw_n, ci_n] = s.weight_shape();
        let mut oihw = vec![0.0; v.w.len()];
        for co in 0..co_n {
            for kh in 0..kh_n {
                for kw in 0..kw_n {
                    for ci in 0..ci_n {
                        let src = ((co * kh_n + kh) * kw_n + kw) * ci_n + ci;
                        let dst = ((co * ci_n + ci) * kh_n + kh) * kw_n + kw;
                        oihw[dst] = v.w[src];
                    }
                }
            }
        }
        with_cuda_backend(|client, device| {
            let label = format!("routing [{dtype:?}]");
            let native = host_f64(&client, &run(&client, &device, dtype, s, &v, epi));
            let oihw_shape = [co_n, ci_n, kh_n, kw_n];
            let w_view = tensor_from_f64(&oihw, &oihw_shape, dtype, &device, &client)
                .unwrap()
                .permute(&[0, 2, 3, 1])
                .unwrap();
            assert!(!w_view.is_contiguous());
            let generic = host_f64(
                &client,
                &run_with_weight(&client, &device, dtype, s, &v, epi, &w_view),
            );
            let both = format!("native vs generic {label}");
            assert_values_close(&native, &expected, rtol, atol, &format!("native {label}"));
            assert_values_close(&generic, &expected, 0.0, generic_atol, &label);
            assert_values_close(&native, &generic, 0.0, generic_atol, &both);
        });
    }
}

/// Two groups route to the generic path, which must still match.
#[test]
fn test_native_conv_grouped_falls_back() {
    let s = Spec {
        groups: 2,
        ..spec("cin64_cout64_groups2", 64, 64)
    };
    let epi = Epilogue {
        act: Act::Relu,
        bias: true,
        residual_scale: Some(1.0),
    };
    for dtype in half_dtypes("cuda") {
        let v = values(s, dtype);
        let expected = reference(s, &v, epi);
        let (_, atol) = gemm_long_k_tolerance(dtype, s.contraction(), 1.0);
        with_cuda_backend(|client, device| {
            let got = host_f64(&client, &run(&client, &device, dtype, s, &v, epi));
            assert_values_close(&got, &expected, 0.0, atol, &format!("groups2 [{dtype:?}]"));
        });
    }
}
