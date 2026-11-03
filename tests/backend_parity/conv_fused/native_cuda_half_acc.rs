// CUDA native `conv2d_fused` with `ConvAccumulation::AllowHalf`.
//
// F16 inputs on sm_75+ take the F16-accumulator kernels: the 256x32 tile for
// C_out <= 32 and the 256x64 tile for wider C_out, each with halo, vector
// gather and scalar staging. Every case rounds its
// operands to F16 first, then compares the CUDA result against the CPU
// result in F64 from the same values, with the K-dependent bound of
// `half_acc_tolerance`.
//
// Cases cover C_in 3, 8, 64 and 72 (scalar and vector staging), C_out 16,
// 32, 64 and 96 (both tiles, a partial N tile, two grid-y tiles), stride 2,
// dilation 2, every activation, bias on and off, residual scales 0.2 and
// 1.0, channel-slice input and output through `conv2d_fused_into`, M below
// one tile, halo patches cut by the image edge, and grids of many 256x64
// tiles.
//
// `test_half_acc_conv_sum_stalls_in_f16` proves the F16 accumulator ran: its
// running sum stalls at 2.0 where the F32 sum reaches 4.0. On a device
// without the native F16 path, `AllowHalf` is ignored and the same test
// expects 4.0.
//
// BF16 ignores `AllowHalf`. Its cases must meet the F32-accumulator
// tolerance of `native_tolerance`.

use numr::dtype::DType;
use numr::ops::{
    Conv2dConfig, ConvAccumulation, ConvActivation, ConvLayout, FusedConvOps, PaddingMode,
};
use numr::runtime::Device;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use super::Act;
use super::native_cuda::{
    Epilogue, SWEEP_EPILOGUES, Spec, Values, all_epilogues, check_dtype, halo_edge_specs, host_f64,
    reference, run, run_into, small, spec, values,
};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::half_dtypes;

/// `spec` with `AllowHalf`.
fn half(spec: Spec) -> Spec {
    Spec {
        accumulation: ConvAccumulation::AllowHalf,
        ..spec
    }
}

/// Per-element bound of the F16 accumulation error of `spec` on `v`.
///
/// - Each F16 rounding of a partial sum errs by at most half an F16 ulp of
///   that sum, which is under `2^-11 |s|`.
/// - Every partial sum of output `i` is at most `A_i`, the sum of
///   `|x| * |w|` over its K terms.
/// - Up to K roundings add like a random walk, so the K sum errs by about
///   `2^-11 * sqrt(K) * A_i`. The bound doubles that for margin.
///
/// The epilogue scales this error by at most 1. Every activation here is
/// 1-Lipschitz, and every residual scale is at most 1.
fn accumulation_bound(spec: Spec, v: &Values) -> Vec<f64> {
    let abs = Values {
        x: v.x.iter().map(|a| a.abs()).collect(),
        w: v.w.iter().map(|a| a.abs()).collect(),
        bias: v.bias.clone(),
        slope_c: v.slope_c.clone(),
        slope_1: v.slope_1.clone(),
        residual: v.residual.clone(),
        out_shape: v.out_shape.clone(),
    };
    let plain = Epilogue {
        act: Act::None,
        bias: false,
        residual_scale: None,
    };
    let walk = 2f64.powi(-10) * (spec.contraction() as f64).sqrt();
    reference(spec, &abs, plain)
        .into_iter()
        .map(|a| walk * a)
        .collect()
}

/// Per-element tolerance against the F64 reference `expected`: the
/// [`accumulation_bound`], plus `2^-10 |expected_i|` for the final rounding
/// to F16 and `1e-3` for the F32 epilogue.
fn half_acc_tolerance(bound: &[f64], expected: &[f64]) -> Vec<f64> {
    bound
        .iter()
        .zip(expected)
        .map(|(&b, &e)| b + 2f64.powi(-10) * e.abs() + 1e-3)
        .collect()
}

/// Every element of `got` is within its entry of `tol` of `expected`.
fn assert_within(got: &[f64], expected: &[f64], tol: &[f64], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length");
    for (i, ((&g, &e), &t)) in got.iter().zip(expected).zip(tol).enumerate() {
        assert!(
            (g - e).abs() <= t,
            "{label}: element {i}: CUDA {g} vs F64 reference {e} (diff {:.3e}, tol {t:.3e})",
            (g - e).abs()
        );
    }
}

/// True when this build has F16 on CUDA.
fn has_f16() -> bool {
    half_dtypes("cuda").contains(&DType::F16)
}

/// Run every epilogue of `spec` with `AllowHalf` in F16 on CUDA and compare
/// with the F64 reference within [`half_acc_tolerance`].
fn check(spec: Spec, epilogues: &[Epilogue]) {
    if !has_f16() {
        return;
    }
    let spec = half(spec);
    let dtype = DType::F16;
    let v = values(spec, dtype);
    let bound = accumulation_bound(spec, &v);
    with_cuda_backend(|client, device| {
        for &epi in epilogues {
            let label = format!(
                "{} c_in {} c_out {} {epi:?} AllowHalf [F16]",
                spec.name, spec.c_in, spec.c_out
            );
            let got = run(&client, &device, dtype, spec, &v, epi);
            assert_eq!(got.shape(), v.out_shape.as_slice(), "{label}: shape");
            assert!(got.is_contiguous(), "{label}: output must be contiguous");
            let got = host_f64(&client, &got);
            let expected = reference(spec, &v, epi);
            let tol = half_acc_tolerance(&bound, &expected);
            assert_within(&got, &expected, &tol, &label);
        }
    });
}

/// C_in 3 takes the scalar staging and the 256x32 tile. C_in 64 takes the
/// vector staging and the 256x64 tile.
#[test]
fn test_half_acc_conv_all_epilogues() {
    check(spec("cin3_cout32_same", 3, 32), &all_epilogues());
    check(spec("cin64_cout64_same", 64, 64), &all_epilogues());
    check(spec("cin72_cout96_same", 72, 96), &all_epilogues());
}

/// M = 234 on a 9x13 image, below one 256-row tile. C_in 3 takes the scalar
/// staging; 8, 64 and 72 the vector staging, 72 with a partial K-tile per
/// tap. C_out 16 and 32 take 256x32. 64 and 96 take 256x64, 96 with a
/// partial second grid-y tile.
#[test]
fn test_half_acc_conv_channel_sweep() {
    for c_in in [3, 8, 64, 72] {
        for c_out in [16, 32, 64, 96] {
            check(small("half_channel_sweep", c_in, c_out), &SWEEP_EPILOGUES);
        }
    }
}

/// Stride 2 and dilation 2 on both tiles and both staging variants.
#[test]
fn test_half_acc_conv_geometry() {
    let specs = [
        Spec {
            stride: (2, 2),
            ..spec("cin64_cout96_stride2_same", 64, 96)
        },
        Spec {
            stride: (2, 2),
            padding: PaddingMode::Valid,
            ..spec("cin3_cout32_stride2_valid", 3, 32)
        },
        Spec {
            dilation: (2, 2),
            ..spec("cin72_cout64_dilation2_same", 72, 64)
        },
        Spec {
            dilation: (2, 2),
            padding: PaddingMode::Valid,
            ..spec("cin8_cout16_dilation2_valid", 8, 16)
        },
        Spec {
            stride: (2, 2),
            dilation: (2, 2),
            ..spec("cin3_cout64_stride2_dilation2_same", 3, 64)
        },
    ];
    for s in specs {
        check(s, &SWEEP_EPILOGUES);
    }
}

/// M far below one tile: a 5x7 image and a 1x1 image.
#[test]
fn test_half_acc_conv_tiny_m() {
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
            ..spec("m1_cin64_cout96", 64, 96)
        },
    ];
    for s in specs {
        check(s, &SWEEP_EPILOGUES);
    }
}

/// Halo patches cut by the image edge, on both F16-accumulator tiles.
#[test]
fn test_half_acc_conv_halo_patch_edges() {
    for s in halo_edge_specs() {
        check(s, &SWEEP_EPILOGUES);
    }
}

/// M = 16384: 64 row tiles of 256 on the 256x64 tile, one grid-y tile at
/// C_out 64 and a partial second one at C_out 96.
#[test]
fn test_half_acc_conv_large_grid() {
    let specs = [
        Spec {
            batch: 1,
            h: 128,
            w: 128,
            ..spec("m16384_cin64_cout64", 64, 64)
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

/// Channels of the shared NHWC buffer in the channel-slice test.
const SLICE_CHANNELS: usize = 160;
/// Value of every buffer channel outside the input slice.
const SENTINEL: f64 = 7.0;

/// One slice case: the input reads `c_in` channels from `in_start`, the
/// output writes `c_out` channels from `out_start`.
struct SliceCase {
    name: &'static str,
    in_start: usize,
    c_in: usize,
    out_start: usize,
    c_out: usize,
}

/// `conv2d_fused_into` reads a channel slice of one buffer and writes a
/// disjoint channel slice of the same buffer.
///
/// - Aligned slices take the vector staging and the 16-byte store.
/// - An input slice at channel 3 takes the scalar staging.
/// - An output slice at channel 70 takes the half2 store.
/// - C_out 96 at channel 8 has one whole and one partial 64-wide N tile.
///
/// The written slice matches the reference. Every other channel keeps its
/// value exactly.
#[test]
fn test_half_acc_conv_into_channel_slices() {
    if !has_f16() {
        return;
    }
    let dtype = DType::F16;
    let cases = [
        SliceCase {
            name: "vec_in_vec_out_256x64",
            in_start: 0,
            c_in: 64,
            out_start: 64,
            c_out: 64,
        },
        SliceCase {
            name: "scalar_in_pair_out_256x64",
            in_start: 3,
            c_in: 64,
            out_start: 70,
            c_out: 64,
        },
        SliceCase {
            name: "scalar_in_vec_out_256x32",
            in_start: 3,
            c_in: 64,
            out_start: 128,
            c_out: 32,
        },
        SliceCase {
            name: "vec_in_partial_n_256x64",
            in_start: 0,
            c_in: 8,
            out_start: 8,
            c_out: 96,
        },
    ];
    for case in cases {
        let s = half(Spec {
            batch: 1,
            ..spec(case.name, case.c_in, case.c_out)
        });
        let v = values(s, dtype);
        let bound = accumulation_bound(s, &v);
        let [_, h, w, _] = s.input_shape();
        let pixels = h * w;
        let initial: Vec<f64> = (0..pixels * SLICE_CHANNELS)
            .map(|i| {
                let (p, c) = (i / SLICE_CHANNELS, i % SLICE_CHANNELS);
                if (case.in_start..case.in_start + case.c_in).contains(&c) {
                    v.x[p * case.c_in + c - case.in_start]
                } else {
                    SENTINEL
                }
            })
            .collect();
        with_cuda_backend(|client, device| {
            let buf = tensor_from_f64(
                &initial,
                &[1, h, w, SLICE_CHANNELS],
                dtype,
                &device,
                &client,
            )
            .unwrap();
            let input = buf.narrow(-1, case.in_start, case.c_in).unwrap();
            let out = buf.narrow(-1, case.out_start, case.c_out).unwrap();
            for epi in SWEEP_EPILOGUES {
                let label = format!("{} {epi:?} AllowHalf [F16]", case.name);
                let before = host_f64(&client, &buf);
                run_into(&client, &device, dtype, s, &v, epi, &input, &out);
                let after = host_f64(&client, &buf);

                let expected = reference(s, &v, epi);
                let tol = half_acc_tolerance(&bound, &expected);
                let written: Vec<f64> = after
                    .chunks(SLICE_CHANNELS)
                    .flat_map(|px| {
                        px[case.out_start..case.out_start + case.c_out]
                            .iter()
                            .copied()
                    })
                    .collect();
                assert_within(&written, &expected, &tol, &label);
                for (i, (a, b)) in after.iter().zip(&before).enumerate() {
                    let c = i % SLICE_CHANNELS;
                    if !(case.out_start..case.out_start + case.c_out).contains(&c) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{label}: channel {c} changed");
                    }
                }
            }
        });
    }
}

/// Two runs on the same inputs give bitwise-equal outputs on both tiles.
#[test]
fn test_half_acc_conv_deterministic() {
    if !has_f16() {
        return;
    }
    let dtype = DType::F16;
    let epi = SWEEP_EPILOGUES[0];
    for s in [
        half(spec("cin64_cout64_same", 64, 64)),
        half(spec("cin3_cout32_same", 3, 32)),
    ] {
        let v = values(s, dtype);
        with_cuda_backend(|client, device| {
            let first = host_f64(&client, &run(&client, &device, dtype, s, &v, epi));
            let second = host_f64(&client, &run(&client, &device, dtype, s, &v, epi));
            let same = first
                .iter()
                .zip(&second)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{}: two AllowHalf runs differ [F16]", s.name);
        });
    }
}

/// BF16 ignores `AllowHalf` and accumulates in F32, so it meets the F32
/// tolerance on shapes that take either F16-accumulator tile.
#[test]
fn test_half_acc_conv_bf16_ignores_allow_half() {
    if !half_dtypes("cuda").contains(&DType::BF16) {
        return;
    }
    for s in [
        spec("cin3_cout32_same", 3, 32),
        spec("cin64_cout64_same", 64, 64),
        small("cin72_cout96_small", 72, 96),
    ] {
        check_dtype(half(s), &SWEEP_EPILOGUES, DType::BF16);
    }
}

/// Kernel extent of the stall control: 32x32 = 1024 taps.
const STALL_TAPS: usize = 32;
/// Input channels of the stall control. Channels 0, 16, 32 and 48 carry
/// weight.
const STALL_C_IN: usize = 64;
/// Weight value: 4096 of them sum to exactly 4.0.
const STALL_WEIGHT: f64 = 1.0 / 1024.0;

/// Inputs of 1.0 and a 32x32 Valid kernel whose weight is 2^-10 at channels
/// 0, 16, 32 and 48 and zero elsewhere. Each output gathers 4096 nonzero
/// products, so the exact sum is 4.0.
///
/// The vector staging puts each group of 16 channels of a tap in its own
/// k16 step, so every mma.sync adds one nonzero product to each output. F16
/// has spacing 2^-9 on [2, 4): once the sum reaches 2.0, adding 2^-10 is a
/// tie that rounds back to 2.0. The F16 accumulator stalls there. The F32
/// accumulator, and BF16 under `AllowHalf`, reach 4.0 exactly.
///
/// C_out 16 takes the 256x32 tile, C_out 64 the 256x64 tile.
#[test]
fn test_half_acc_conv_sum_stalls_in_f16() {
    for dtype in half_dtypes("cuda") {
        // One storage ulp at 4.0, as in `conv_half_accumulation.rs`.
        let ulp = match dtype {
            DType::F16 => 4e-3,
            _ => 3.2e-2,
        };
        for c_out in [16, 64] {
            let w_shape = [c_out, STALL_TAPS, STALL_TAPS, STALL_C_IN];
            let w_data: Vec<f64> = (0..w_shape.iter().product::<usize>())
                .map(|i| {
                    if (i % STALL_C_IN).is_multiple_of(16) {
                        STALL_WEIGHT
                    } else {
                        0.0
                    }
                })
                .collect();
            with_cuda_backend(|client, device| {
                let caps = device.profile().caps;
                let native_f16 = dtype == DType::F16 && caps.f16_mma && caps.int8_mma;
                let x = Tensor::<CudaRuntime>::full_scalar(
                    &[1, STALL_TAPS, STALL_TAPS, STALL_C_IN],
                    dtype,
                    1.0,
                    &device,
                )
                .unwrap();
                let w = tensor_from_f64(&w_data, &w_shape, dtype, &device, &client).unwrap();
                let sum = |accumulation: ConvAccumulation| {
                    let config =
                        Conv2dConfig::new(ConvLayout::Nhwc).with_accumulation(accumulation);
                    let out = client
                        .conv2d_fused(&x, &w, None, &config, ConvActivation::None, None)
                        .unwrap_or_else(|e| {
                            panic!("stall c_out {c_out} {accumulation:?} [{dtype:?}]: {e}")
                        });
                    host_f64(&client, &out)
                };
                let strict = sum(ConvAccumulation::F32);
                let allow_half = sum(ConvAccumulation::AllowHalf);
                assert_eq!(strict.len(), c_out);
                assert_eq!(allow_half.len(), c_out);
                let expected_half = if native_f16 { 2.0 } else { 4.0 };
                for (co, (&s, &h)) in strict.iter().zip(&allow_half).enumerate() {
                    assert!(
                        (s - 4.0).abs() <= ulp,
                        "stall c_out {c_out} channel {co} F32 [{dtype:?}]: {s} is not 4.0"
                    );
                    assert!(
                        (h - expected_half).abs() <= ulp,
                        "stall c_out {c_out} channel {co} AllowHalf [{dtype:?}]: {h} is not \
                         {expected_half}, native F16 accumulation is {native_f16}"
                    );
                }
            });
        }
    }
}
