// CUDA native `conv2d_fused` in NCHW: the implicit-GEMM kernel with NCHW
// staging and the NCHW epilogue. `native_cuda_nchw_views` and
// `native_cuda_nchw_conv2d` reuse this file's operands.
//
// Each case builds its operands with the NHWC helpers of `native_cuda`
// (values rounded to the half dtype first), permutes them to NCHW and OIHW
// on the device, runs the op in NCHW and permutes the result back to NHWC.
// The reference is the CPU result in F64 from the same values
// (`native_cuda::reference`). The F32 accumulator meets `native_tolerance`,
// `AllowHalf` the K-dependent bound of `native_cuda_half_acc`.
//
// Staging (`select_conv_staging` in
// `src/runtime/cuda/kernels/loader/conv_implicit_gemm_staging.rs`):
// - nchw_halo: C_in % 8 == 0, a stride-1 kernel of up to 3x3, input N/C/H
//   strides that are multiples of 4 (W a multiple of 4 when contiguous) and
//   W stride 1.
// - nchw_gather: every other shape. Here C_in 3, W not a multiple of 4,
//   stride 2, dilation 2, 5x5, and channels-last views with W stride C.
// Store (`conv_implicit_gemm_epilogue_nchw.cuh`): 16-byte stores need W
// stride 1 and N/C/H strides that are multiples of 8, one-element stores
// take the rest. A W slice of a wider buffer reaches the halo runs and the
// store runs that the image edge cuts.
//
// The crate exposes no routing probe, so these tests cannot assert which
// kernel ran. On a device without the native path the same cases check the
// generic path, with the generic tolerance (`tolerance`).

use numr::dtype::DType;
use numr::ops::{
    Conv2dConfig, ConvAccumulation, ConvActivation, ConvLayout, ConvResidual, FusedConvOps,
    PaddingMode,
};
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use super::native_cuda::{
    Epilogue, SWEEP_EPILOGUES, Spec, Values, all_epilogues, assert_values_close, host_f64,
    native_tolerance, reference, small, spec, values,
};
use super::native_cuda_half_acc::{
    accumulation_bound, assert_within, half, half_acc_tolerance, has_f16,
};
use super::{Act, LEAKY_SLOPE, NCHW_TO_NHWC, NHWC_TO_NCHW};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{gemm_long_k_tolerance, half_dtypes};

/// Value of buffer elements a call must not write or read.
pub(super) const SENTINEL: f64 = 7.0;

/// `(rtol, atol)` against the F64 reference: [`native_tolerance`] when the
/// device has the native path for `dtype`, else the generic path's
/// accumulation-aware bound.
pub(super) fn tolerance(device: &CudaDevice, dtype: DType, contraction: usize) -> (f64, f64) {
    let caps = device.profile().caps;
    let native = match dtype {
        DType::F16 => caps.f16_mma && caps.int8_mma,
        _ => caps.bf16,
    };
    if native {
        native_tolerance(dtype)
    } else {
        gemm_long_k_tolerance(dtype, contraction, 1.0)
    }
}

/// `spec` as an NCHW config.
fn nchw_config(spec: Spec) -> Conv2dConfig {
    Conv2dConfig::new(ConvLayout::Nchw)
        .with_stride(spec.stride)
        .with_padding(spec.padding)
        .with_dilation(spec.dilation)
        .with_groups(spec.groups)
        .with_accumulation(spec.accumulation)
}

/// A contiguous copy of `t` with its axes moved by `perm`.
pub(super) fn permuted(t: &Tensor<CudaRuntime>, perm: [usize; 4]) -> Tensor<CudaRuntime> {
    t.permute(&perm).unwrap().contiguous().unwrap()
}

/// Operands of one spec on CUDA: input NCHW, weight OIHW, residual NCHW.
pub(super) struct Nchw {
    pub(super) x: Tensor<CudaRuntime>,
    pub(super) w: Tensor<CudaRuntime>,
    pub(super) bias: Tensor<CudaRuntime>,
    slope_c: Tensor<CudaRuntime>,
    slope_1: Tensor<CudaRuntime>,
    pub(super) residual: Tensor<CudaRuntime>,
}

impl Nchw {
    pub(super) fn new(
        client: &CudaClient,
        device: &CudaDevice,
        dtype: DType,
        spec: Spec,
        v: &Values,
    ) -> Self {
        let make = |data: &[f64], shape: &[usize]| {
            tensor_from_f64(data, shape, dtype, device, client).unwrap()
        };
        // OHWI to OIHW is the same axis move as NHWC to NCHW.
        Self {
            x: permuted(&make(&v.x, &spec.input_shape()), NHWC_TO_NCHW),
            w: permuted(&make(&v.w, &spec.weight_shape()), NHWC_TO_NCHW),
            bias: make(&v.bias, &[spec.c_out]),
            slope_c: make(&v.slope_c, &[spec.c_out]),
            slope_1: make(&v.slope_1, &[1]),
            residual: permuted(&make(&v.residual, &v.out_shape), NHWC_TO_NCHW),
        }
    }

    fn activation(&self, act: Act) -> ConvActivation<'_, CudaRuntime> {
        match act {
            Act::None => ConvActivation::None,
            Act::Relu => ConvActivation::Relu,
            Act::Leaky => ConvActivation::LeakyRelu(LEAKY_SLOPE),
            Act::PreluPerChannel => ConvActivation::Prelu(&self.slope_c),
            Act::PreluShared => ConvActivation::Prelu(&self.slope_1),
        }
    }

    /// `conv2d_fused` in NCHW on input `x` with residual `residual`,
    /// returned as an NCHW tensor.
    pub(super) fn fused(
        &self,
        client: &CudaClient,
        spec: Spec,
        epi: Epilogue,
        x: &Tensor<CudaRuntime>,
        residual: &Tensor<CudaRuntime>,
    ) -> Tensor<CudaRuntime> {
        client
            .conv2d_fused(
                x,
                &self.w,
                epi.bias.then_some(&self.bias),
                &nchw_config(spec),
                self.activation(epi.act),
                epi.residual_scale.map(|scale| ConvResidual {
                    tensor: residual,
                    scale,
                }),
            )
            .unwrap_or_else(|e| panic!("conv2d_fused NCHW {} {epi:?}: {e}", spec.name))
    }

    /// `conv2d_fused_into` in NCHW.
    pub(super) fn fused_into(
        &self,
        client: &CudaClient,
        spec: Spec,
        epi: Epilogue,
        out: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        residual: &Tensor<CudaRuntime>,
    ) {
        client
            .conv2d_fused_into(
                out,
                x,
                &self.w,
                epi.bias.then_some(&self.bias),
                &nchw_config(spec),
                self.activation(epi.act),
                epi.residual_scale.map(|scale| ConvResidual {
                    tensor: residual,
                    scale,
                }),
            )
            .unwrap_or_else(|e| panic!("conv2d_fused_into NCHW {} {epi:?}: {e}", spec.name));
    }
}

/// NCHW `out_shape` of a spec whose NHWC output shape is `nhwc`.
pub(super) fn nchw_shape(nhwc: &[usize]) -> [usize; 4] {
    [nhwc[0], nhwc[3], nhwc[1], nhwc[2]]
}

/// [`check`] for one dtype.
fn check_dtype(spec: Spec, epilogues: &[Epilogue], dtype: DType) {
    let v = values(spec, dtype);
    with_cuda_backend(|client, device| {
        let ops = Nchw::new(&client, &device, dtype, spec, &v);
        let (rtol, atol) = tolerance(&device, dtype, spec.contraction());
        for &epi in epilogues {
            let label = format!(
                "NCHW {} c_in {} c_out {} {epi:?} [{dtype:?}]",
                spec.name, spec.c_in, spec.c_out
            );
            let got = ops.fused(&client, spec, epi, &ops.x, &ops.residual);
            assert_eq!(
                got.shape(),
                nchw_shape(&v.out_shape).as_slice(),
                "{label}: shape"
            );
            assert!(got.is_contiguous(), "{label}: output must be contiguous");
            let got = host_f64(&client, &permuted(&got, NCHW_TO_NHWC));
            assert_values_close(&got, &reference(spec, &v, epi), rtol, atol, &label);
        }
    });
}

/// Every epilogue of `spec` in NCHW on CUDA against the F64 reference.
fn check(spec: Spec, epilogues: &[Epilogue]) {
    for dtype in half_dtypes("cuda") {
        check_dtype(spec, epilogues, dtype);
    }
}

/// `spec` with the image `h x w`.
pub(super) const fn sized(s: Spec, h: usize, w: usize) -> Spec {
    Spec { h, w, ..s }
}

/// Every epilogue on both staging variants and both tiles: halo for C_in
/// 64 and 72 at W 24 and 20, gather for C_in 3 and for W 23.
#[test]
fn test_nchw_all_epilogues() {
    for s in [
        sized(spec("halo_cin64_cout64", 64, 64), 17, 24),
        sized(spec("halo_cin72_cout96", 72, 96), 9, 20),
        spec("gather_cin3_cout32", 3, 32),
        spec("gather_cin8_cout16_w23", 8, 16),
    ] {
        check(s, &all_epilogues());
    }
}

/// C_in 3, 8, 64 and 72 against C_out 4, 16, 32, 64 and 96: both tiles, a
/// partial N tile at 4 and 96, two grid-y tiles at 96. W 12 takes the halo
/// where C_in % 8 == 0, W 13 the gather.
#[test]
fn test_nchw_channel_sweep() {
    for c_in in [3, 8, 64, 72] {
        for c_out in [4, 16, 32, 64, 96] {
            for w in [12, 13] {
                check(
                    sized(small("nchw_channel_sweep", c_in, c_out), 9, w),
                    &SWEEP_EPILOGUES,
                );
            }
        }
    }
}

/// Stride, dilation, kernel size and padding. Stride 2, dilation 2 and 5x5
/// take the gather; the stride-1 cases at W 24 take the halo with each
/// column offset `(-pad_left) mod 4` of the halo runs.
#[test]
fn test_nchw_geometry() {
    let specs = [
        Spec {
            stride: (2, 2),
            padding: PaddingMode::Valid,
            ..spec("cin64_cout48_stride2_valid", 64, 48)
        },
        Spec {
            stride: (2, 2),
            ..sized(spec("cin8_cout64_stride2_same_w24", 8, 64), 17, 24)
        },
        Spec {
            dilation: (2, 2),
            ..spec("cin64_cout32_dilation2_same", 64, 32)
        },
        Spec {
            kernel: 5,
            ..spec("cin72_cout96_5x5_same", 72, 96)
        },
        Spec {
            kernel: 5,
            padding: PaddingMode::Valid,
            ..spec("cin3_cout16_5x5_valid", 3, 16)
        },
        Spec {
            kernel: 1,
            padding: PaddingMode::Valid,
            ..sized(spec("halo_cin64_cout32_1x1", 64, 32), 19, 24)
        },
        Spec {
            padding: PaddingMode::Custom(2, 0, 1, 3),
            ..sized(spec("halo_cin16_cout64_pad_left1", 16, 64), 17, 24)
        },
        Spec {
            padding: PaddingMode::Custom(1, 1, 2, 0),
            ..sized(spec("halo_cin64_cout64_pad_left2", 64, 64), 17, 24)
        },
        Spec {
            padding: PaddingMode::Custom(0, 2, 3, 0),
            ..sized(spec("halo_cin8_cout16_pad_left3", 8, 16), 17, 24)
        },
        Spec {
            padding: PaddingMode::Valid,
            ..sized(spec("halo_cin24_cout40_valid", 24, 40), 17, 24)
        },
        Spec {
            kernel: 2,
            padding: PaddingMode::Custom(1, 0, 0, 1),
            ..sized(spec("halo_cin32_cout64_2x2", 32, 64), 11, 20)
        },
        Spec {
            stride: (2, 1),
            dilation: (1, 2),
            padding: PaddingMode::Custom(1, 2, 0, 2),
            ..spec("cin8_cout40_mixed", 8, 40)
        },
        Spec {
            padding: PaddingMode::Custom(2, 0, 1, 3),
            ..spec("cin8_cout32_custom_pad_w23", 8, 32)
        },
    ];
    for s in specs {
        check(s, &SWEEP_EPILOGUES);
    }
}

/// Halo patches (8x16 for 128x64, 8x32 for the 256-row tiles) cut by the
/// image edge in H, in W and in both, batch 1 and 2.
#[test]
fn test_nchw_halo_patch_edges() {
    for s in [
        Spec {
            batch: 1,
            ..sized(spec("halo_h19_w44_cin64_cout64", 64, 64), 19, 44)
        },
        Spec {
            batch: 1,
            ..sized(spec("halo_h19_w44_cin16_cout32", 16, 32), 19, 44)
        },
        sized(spec("halo_h17_w36_batch2_cin24_cout48", 24, 48), 17, 36),
        Spec {
            batch: 1,
            ..sized(spec("halo_h9_w68_cin72_cout16", 72, 16), 9, 68)
        },
        Spec {
            batch: 1,
            ..sized(spec("halo_h8_w32_exact_cin64_cout96", 64, 96), 8, 32)
        },
    ] {
        check(s, &SWEEP_EPILOGUES);
    }
}

/// Two runs on the same inputs give bitwise-equal outputs, on both staging
/// variants.
#[test]
fn test_nchw_deterministic() {
    let epi = SWEEP_EPILOGUES[0];
    for s in [
        sized(spec("halo_cin64_cout64", 64, 64), 17, 24),
        spec("gather_cin64_cout64", 64, 64),
    ] {
        for dtype in half_dtypes("cuda") {
            let v = values(s, dtype);
            with_cuda_backend(|client, device| {
                let ops = Nchw::new(&client, &device, dtype, s, &v);
                let run = || host_f64(&client, &ops.fused(&client, s, epi, &ops.x, &ops.residual));
                let first = run();
                let second = run();
                let same = first
                    .iter()
                    .zip(&second)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{}: two runs differ [{dtype:?}]", s.name);
            });
        }
    }
}

/// `AllowHalf` in F16 on both staging variants and both F16-accumulator
/// tiles, within the K-dependent bound. BF16 ignores `AllowHalf` and meets
/// the F32-accumulator tolerance.
#[test]
fn test_nchw_half_accumulation() {
    let specs = [
        half(sized(spec("halo_cin64_cout64", 64, 64), 17, 24)),
        half(sized(spec("halo_cin8_cout16", 8, 16), 19, 44)),
        half(spec("gather_cin3_cout96", 3, 96)),
        half(Spec {
            stride: (2, 2),
            ..spec("gather_cin72_cout32_stride2", 72, 32)
        }),
    ];
    for s in specs {
        if has_f16() {
            let v = values(s, DType::F16);
            let bound = accumulation_bound(s, &v);
            with_cuda_backend(|client, device| {
                let ops = Nchw::new(&client, &device, DType::F16, s, &v);
                for epi in all_epilogues() {
                    let label = format!("NCHW {} {epi:?} AllowHalf [F16]", s.name);
                    let got = ops.fused(&client, s, epi, &ops.x, &ops.residual);
                    let got = host_f64(&client, &permuted(&got, NCHW_TO_NHWC));
                    let expected = reference(s, &v, epi);
                    let tol = half_acc_tolerance(&bound, &expected);
                    assert_within(&got, &expected, &tol, &label);
                }
            });
        }
        if half_dtypes("cuda").contains(&DType::BF16) {
            check_dtype(s, &SWEEP_EPILOGUES, DType::BF16);
        }
    }
}

/// Grouped NCHW calls stay on the generic path, which still matches.
#[test]
fn test_nchw_grouped_falls_back() {
    let s = Spec {
        groups: 2,
        accumulation: ConvAccumulation::F32,
        ..spec("nchw_groups2", 64, 64)
    };
    let epi = SWEEP_EPILOGUES[0];
    for dtype in half_dtypes("cuda") {
        let v = values(s, dtype);
        let expected = reference(s, &v, epi);
        let (_, atol) = gemm_long_k_tolerance(dtype, s.contraction(), 1.0);
        with_cuda_backend(|client, device| {
            let ops = Nchw::new(&client, &device, dtype, s, &v);
            let got = ops.fused(&client, s, epi, &ops.x, &ops.residual);
            let got = host_f64(&client, &permuted(&got, NCHW_TO_NHWC));
            assert_values_close(&got, &expected, 0.0, atol, &format!("groups2 [{dtype:?}]"));
        });
    }
}
