// CUDA native `conv2d_fused`: the implicit-GEMM tensor-core kernel for F16
// and BF16 NHWC with one group.
//
// Every case rounds its operands to the half dtype first, then compares the
// CUDA result against the CPU result computed in F64 from the same values.
// The kernel accumulates in F32 and rounds once at the store, so the
// tolerance is one storage ulp relative plus a small absolute floor for the
// F32 sum (`native_tolerance`).
//
// Cases cover all three staging variants: scalar (C_in 3, 12, and any C_in
// with a misaligned view), halo (C_in a multiple of 8 with a stride-1
// kernel of up to 3x3) and vector gather (C_in a multiple of 8 with stride
// 2, dilation 2 or 5x5). They cover every tile (256x32 for C_out <= 32,
// 128x64 for wider C_out), partial N tiles, C_out over several grid-y
// tiles, M below one tile and not a multiple of 256, halo patches cut by
// the image edge in H, in W and in both, 1x1, 2x2, 3x3 and 5x5 kernels,
// stride, dilation, Valid, Same and asymmetric padding, every activation,
// bias on and off, and residual scales 0.2 and 1.0.
//
// The crate exposes no routing probe, so these tests cannot assert that a
// call took the native kernel. Every spec here meets the native conditions
// of `ops/cuda/conv_fused_native.rs`: F16 or BF16, NHWC, one group, a
// contiguous OHWI weight, channel stride 1, and Kh, Kw <= 32. On a device
// without the native path (below sm_75, or BF16 below sm_80) the same cases
// check the generic path instead. Routing is also covered by
// native-vs-generic agreement and by a grouped case that takes the generic
// path.

use numr::dtype::DType;
use numr::ops::{
    Conv2dConfig, ConvAccumulation, ConvActivation, ConvLayout, ConvResidual, FusedConvOps,
    PaddingMode, TypeConversionOps,
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
    pub accumulation: ConvAccumulation,
}

impl Spec {
    pub(super) fn config(self) -> Conv2dConfig {
        Conv2dConfig::new(ConvLayout::Nhwc)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_dilation(self.dilation)
            .with_groups(self.groups)
            .with_accumulation(self.accumulation)
    }

    pub(super) fn input_shape(self) -> [usize; 4] {
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

    pub(super) fn contraction(self) -> usize {
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
pub(super) fn all_epilogues() -> Vec<Epilogue> {
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
    pub x: Vec<f64>,
    pub w: Vec<f64>,
    pub bias: Vec<f64>,
    pub slope_c: Vec<f64>,
    pub slope_1: Vec<f64>,
    pub residual: Vec<f64>,
    pub out_shape: Vec<usize>,
}

pub(super) fn values(spec: Spec, dtype: DType) -> Values {
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

/// Bias, PReLU weights and residual of one [`Values`], as tensors.
struct EpilogueTensors<R: Runtime> {
    bias: Tensor<R>,
    slope_c: Tensor<R>,
    slope_1: Tensor<R>,
    residual: Tensor<R>,
}

impl<R: Runtime<DType = DType>> EpilogueTensors<R> {
    fn new<C>(client: &C, device: &R::Device, dtype: DType, spec: Spec, v: &Values) -> Self
    where
        C: TypeConversionOps<R>,
    {
        let make = |data: &[f64], shape: &[usize]| {
            tensor_from_f64(data, shape, dtype, device, client).unwrap()
        };
        Self {
            bias: make(&v.bias, &[spec.c_out]),
            slope_c: make(&v.slope_c, &[spec.c_out]),
            slope_1: make(&v.slope_1, &[1]),
            residual: make(&v.residual, &v.out_shape),
        }
    }

    fn bias(&self, epi: Epilogue) -> Option<&Tensor<R>> {
        epi.bias.then_some(&self.bias)
    }

    fn activation(&self, epi: Epilogue) -> ConvActivation<'_, R> {
        match epi.act {
            Act::None => ConvActivation::None,
            Act::Relu => ConvActivation::Relu,
            Act::Leaky => ConvActivation::LeakyRelu(LEAKY_SLOPE),
            Act::PreluPerChannel => ConvActivation::Prelu(&self.slope_c),
            Act::PreluShared => ConvActivation::Prelu(&self.slope_1),
        }
    }

    fn residual(&self, epi: Epilogue) -> Option<ConvResidual<'_, R>> {
        epi.residual_scale.map(|scale| ConvResidual {
            tensor: &self.residual,
            scale,
        })
    }
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
    let x = tensor_from_f64(&v.x, &spec.input_shape(), dtype, device, client).unwrap();
    let ep = EpilogueTensors::new(client, device, dtype, spec, v);
    client
        .conv2d_fused(
            &x,
            w,
            ep.bias(epi),
            &spec.config(),
            ep.activation(epi),
            ep.residual(epi),
        )
        .unwrap_or_else(|e| panic!("conv2d_fused {} {epi:?} [{dtype:?}]: {e}", spec.name))
}

/// `conv2d_fused_into` into `out` with a caller-built input, so a test can
/// pass channel slices of one buffer. Weight, bias, PReLU weights and
/// residual come from `v`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_into<R, C>(
    client: &C,
    device: &R::Device,
    dtype: DType,
    spec: Spec,
    v: &Values,
    epi: Epilogue,
    x: &Tensor<R>,
    out: &Tensor<R>,
) where
    R: Runtime<DType = DType>,
    C: FusedConvOps<R> + TypeConversionOps<R>,
{
    let w = tensor_from_f64(&v.w, &spec.weight_shape(), dtype, device, client).unwrap();
    let ep = EpilogueTensors::new(client, device, dtype, spec, v);
    client
        .conv2d_fused_into(
            out,
            x,
            &w,
            ep.bias(epi),
            &spec.config(),
            ep.activation(epi),
            ep.residual(epi),
        )
        .unwrap_or_else(|e| panic!("conv2d_fused_into {} {epi:?} [{dtype:?}]: {e}", spec.name));
}

/// The CPU result in F64 from the same rounded values.
pub(super) fn reference(spec: Spec, v: &Values, epi: Epilogue) -> Vec<f64> {
    let (cpu, device) = create_cpu_client();
    let out = run(&cpu, &device, DType::F64, spec, v, epi);
    out.to_vec::<f64>()
}

/// Run every epilogue of `spec` on CUDA and compare with the F64 reference.
fn check(spec: Spec, epilogues: &[Epilogue]) {
    for dtype in half_dtypes("cuda") {
        check_dtype(spec, epilogues, dtype);
    }
}

/// [`check`] for one dtype, with the [`native_tolerance`] of that dtype.
pub(super) fn check_dtype(spec: Spec, epilogues: &[Epilogue], dtype: DType) {
    let v = values(spec, dtype);
    let (rtol, atol) = native_tolerance(dtype);
    with_cuda_backend(|client, device| {
        for &epi in epilogues {
            let label = format!(
                "{} c_in {} c_out {} {epi:?} [{dtype:?}]",
                spec.name, spec.c_in, spec.c_out
            );
            let got = run(&client, &device, dtype, spec, &v, epi);
            assert_eq!(got.shape(), v.out_shape.as_slice(), "{label}: shape");
            assert!(got.is_contiguous(), "{label}: output must be contiguous");
            let got = host_f64(&client, &got);
            assert_values_close(&got, &reference(spec, &v, epi), rtol, atol, &label);
        }
    });
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

pub(super) const fn spec(name: &'static str, c_in: usize, c_out: usize) -> Spec {
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
        accumulation: ConvAccumulation::F32,
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

/// `spec` on a 9x13 image: M = 234, below one 256-row tile and above one
/// 128-row tile.
pub(super) const fn small(name: &'static str, c_in: usize, c_out: usize) -> Spec {
    Spec {
        h: 9,
        w: 13,
        ..spec(name, c_in, c_out)
    }
}

/// Every operand in one, PReLU alone in the other.
pub(super) const SWEEP_EPILOGUES: [Epilogue; 2] = [
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

/// C_in 3, 24 and 40 leave a partial chunk or K-tile, 8 and 72 fill whole
/// chunks; 3 takes the scalar variant. C_out 16 and 32 take 256x32, the
/// rest a 64-wide tile with a partial N tile at 48 and 96 and two or three
/// grid-y tiles at 96 and 160.
#[test]
fn test_native_conv_channel_sweep() {
    for c_in in [3, 8, 24, 40, 72] {
        for c_out in [16, 32, 48, 64, 96, 160] {
            check(small("channel_sweep", c_in, c_out), &SWEEP_EPILOGUES);
        }
    }
}

/// Every epilogue on a vector-variant spec with a partial N tile, and on a
/// scalar-variant spec with the 256x32 tile.
#[test]
fn test_native_conv_mid_channels_all_epilogues() {
    check(spec("cin40_cout96_same", 40, 96), &all_epilogues());
    check(spec("cin24_cout16_same", 24, 16), &all_epilogues());
    check(spec("cin3_cout48_same", 3, 48), &all_epilogues());
}

/// Stride 2, dilation 2, Valid and Same padding, 1x1 and 5x5 kernels, on
/// both staging variants.
#[test]
fn test_native_conv_geometry_sweep() {
    let specs = [
        Spec {
            stride: (2, 2),
            ..spec("cin40_cout96_stride2_same", 40, 96)
        },
        Spec {
            stride: (2, 2),
            padding: PaddingMode::Valid,
            ..spec("cin24_cout48_stride2_valid", 24, 48)
        },
        Spec {
            stride: (2, 2),
            dilation: (2, 2),
            ..spec("cin3_cout32_stride2_dilation2_same", 3, 32)
        },
        Spec {
            kernel: 5,
            dilation: (2, 2),
            ..spec("cin24_cout32_5x5_dilation2_same", 24, 32)
        },
        Spec {
            kernel: 5,
            padding: PaddingMode::Valid,
            ..spec("cin3_cout16_5x5_valid", 3, 16)
        },
        Spec {
            kernel: 5,
            ..spec("cin72_cout160_5x5_same", 72, 160)
        },
        Spec {
            kernel: 1,
            padding: PaddingMode::Valid,
            ..spec("cin40_cout64_1x1_valid", 40, 64)
        },
        Spec {
            kernel: 1,
            ..spec("cin3_cout48_1x1_same", 3, 48)
        },
        Spec {
            dilation: (2, 2),
            padding: PaddingMode::Valid,
            ..spec("cin8_cout96_dilation2_valid", 8, 96)
        },
    ];
    for s in specs {
        check(s, &SWEEP_EPILOGUES);
    }
}

/// M far below one tile: a 5x7 image, a 1x1 image, and a Valid 3x3 on a
/// 3x3 image, which leaves one output pixel per batch.
#[test]
fn test_native_conv_tiny_m() {
    let specs = [
        Spec {
            batch: 1,
            h: 5,
            w: 7,
            ..spec("m35_cin8_cout64", 8, 64)
        },
        Spec {
            batch: 1,
            h: 5,
            w: 7,
            ..spec("m35_cin3_cout32", 3, 32)
        },
        Spec {
            batch: 1,
            h: 1,
            w: 1,
            ..spec("m1_cin24_cout48", 24, 48)
        },
        Spec {
            h: 3,
            w: 3,
            padding: PaddingMode::Valid,
            ..spec("m2_cin40_cout16_valid", 40, 16)
        },
    ];
    for s in specs {
        check(s, &SWEEP_EPILOGUES);
    }
}

/// M = 16384: 128 row tiles of 128 at C_out 64, and a second, partial grid-y
/// tile at C_out 96. Both take the 128x64 tile.
#[test]
fn test_native_conv_large_grid() {
    let specs = [
        Spec {
            batch: 1,
            h: 128,
            w: 128,
            ..spec("m16384_cin16_cout64", 16, 64)
        },
        Spec {
            batch: 1,
            h: 128,
            w: 128,
            ..spec("m16384_cin16_cout96", 16, 96)
        },
    ];
    for s in specs {
        check(s, &SWEEP_EPILOGUES[..1]);
    }
}

/// Halo-staging shapes whose output patches (8x16 for 128x64, 8x32 for the
/// 256-row tiles) are cut by the image edge: H not a multiple of 8, W not a
/// multiple of 16 or 32, one image or two, and kernels whose halo still fits.
pub(super) fn halo_edge_specs() -> [Spec; 9] {
    [
        Spec {
            batch: 1,
            h: 19,
            w: 45,
            ..spec("halo_h19_w45_cin64_cout64", 64, 64)
        },
        Spec {
            batch: 1,
            h: 19,
            w: 45,
            ..spec("halo_h19_w45_cin16_cout32", 16, 32)
        },
        Spec {
            batch: 1,
            h: 8,
            w: 32,
            ..spec("halo_h8_w32_exact_cin64_cout96", 64, 96)
        },
        Spec {
            h: 17,
            w: 33,
            ..spec("halo_h17_w33_batch2_cin24_cout48", 24, 48)
        },
        Spec {
            batch: 1,
            h: 9,
            w: 70,
            ..spec("halo_h9_w70_cin72_cout16", 72, 16)
        },
        Spec {
            batch: 1,
            h: 19,
            w: 45,
            padding: PaddingMode::Valid,
            ..spec("halo_h19_w45_valid_cin40_cout64", 40, 64)
        },
        Spec {
            batch: 1,
            h: 19,
            w: 45,
            padding: PaddingMode::Custom(2, 0, 1, 3),
            ..spec("halo_h19_w45_custom_pad_cin8_cout32", 8, 32)
        },
        Spec {
            batch: 1,
            h: 19,
            w: 45,
            kernel: 1,
            padding: PaddingMode::Valid,
            ..spec("halo_h19_w45_1x1_cin64_cout96", 64, 96)
        },
        Spec {
            h: 11,
            w: 37,
            kernel: 2,
            padding: PaddingMode::Custom(1, 0, 0, 1),
            ..spec("halo_h11_w37_2x2_cin32_cout64", 32, 64)
        },
    ]
}

/// Halo patches cut by the image edge, every tile.
#[test]
fn test_native_conv_halo_patch_edges() {
    for s in halo_edge_specs() {
        check(s, &SWEEP_EPILOGUES);
    }
}
