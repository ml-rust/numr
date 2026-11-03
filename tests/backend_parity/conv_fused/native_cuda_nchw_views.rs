// CUDA native `conv2d_fused_into` in NCHW on strided views: W slices,
// channel slices and channels-last buffers viewed as NCHW. Operands and
// helpers come from `native_cuda_nchw`.

use numr::ops::BinaryOps;

use super::native_cuda::{
    SWEEP_EPILOGUES, Spec, assert_values_close, host_f64, reference, spec, values,
};
use super::native_cuda_nchw::{Nchw, SENTINEL, permuted, sized, tolerance};
use super::{NCHW_TO_NHWC, NHWC_TO_NCHW};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::half_dtypes;

/// NCHW `[n, c, h, wide]` data whose columns `[0, w)` hold the NHWC
/// `[n, h, w, c]` `data` and whose other columns hold `fill`.
fn nchw_wide(data: &[f64], [n, h, w, c]: [usize; 4], wide: usize, fill: f64) -> Vec<f64> {
    let mut out = vec![fill; n * c * h * wide];
    for (i, &v) in data.iter().enumerate() {
        let ci = i % c;
        let wi = (i / c) % w;
        let hi = (i / (c * w)) % h;
        let ni = i / (c * w * h);
        out[((ni * c + ci) * h + hi) * wide + wi] = v;
    }
    out
}

/// Input and output are W slices `[0, w)` of buffers 24 wide, so the H
/// stride is 24 while W is 23 or 21. The halo runs at the right image edge
/// load pixel by pixel, the halo column past W reads as zero although the
/// buffer holds 1000.0 there, and the 16-byte store runs cut by W store
/// pixel by pixel. Output columns past W keep the sentinel.
#[test]
fn test_nchw_w_slices() {
    const WIDE: usize = 24;
    let epi = SWEEP_EPILOGUES[0];
    for (c_in, c_out, w) in [(64, 64, 23), (16, 32, 21), (8, 96, 23)] {
        let s = sized(spec("halo_w_slice", c_in, c_out), 17, w);
        for dtype in half_dtypes("cuda") {
            let v = values(s, dtype);
            let expected = reference(s, &v, epi);
            with_cuda_backend(|client, device| {
                let (rtol, atol) = tolerance(&device, dtype, s.contraction());
                let make = |data: &[f64], shape: &[usize]| {
                    tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
                };
                let ops = Nchw::new(&client, &device, dtype, s, &v);
                let label = format!("W slice {w} of {WIDE}, c_in {c_in} c_out {c_out} [{dtype:?}]");
                let x_buf = make(
                    &nchw_wide(&v.x, s.input_shape(), WIDE, 1000.0),
                    &[s.batch, c_in, s.h, WIDE],
                );
                let out_shape = [s.batch, c_out, s.h, WIDE];
                let out_buf = make(&vec![SENTINEL; out_shape.iter().product()], &out_shape);
                let out = out_buf.narrow(3, 0, w).unwrap();
                let x = x_buf.narrow(3, 0, w).unwrap();
                ops.fused_into(&client, s, epi, &out, &x, &ops.residual);
                let got = host_f64(&client, &permuted(&out, NCHW_TO_NHWC));
                assert_values_close(&got, &expected, rtol, atol, &label);
                let after = host_f64(&client, &out_buf);
                for (i, val) in after.iter().enumerate() {
                    let col = i % WIDE;
                    if col >= w {
                        assert_eq!(
                            val.to_bits(),
                            SENTINEL.to_bits(),
                            "{label}: column {col} changed"
                        );
                    }
                }
            });
        }
    }
}

/// `conv2d_fused_into` with input, output and residual as channel slices of
/// wider NCHW buffers. W 24 gives N/C/H strides that are multiples of 8, so
/// the output takes 16-byte stores at every start; W 23 takes one-element
/// stores and the gather. Channels outside the output slice keep the
/// sentinel.
#[test]
fn test_nchw_into_channel_slices() {
    const BATCH: usize = 2;
    const H: usize = 17;
    let epi = SWEEP_EPILOGUES[0];
    // (c_in, c_out, w, wide, start)
    let cases = [
        (64, 64, 24, 96, 16),
        (24, 32, 24, 48, 5),
        (8, 96, 24, 104, 8),
        (40, 48, 23, 64, 8),
        (3, 16, 23, 21, 3),
    ];
    for (c_in, c_out, w, wide, start) in cases {
        let s = Spec {
            batch: BATCH,
            ..sized(spec("into_slices", c_in, c_out), H, w)
        };
        for dtype in half_dtypes("cuda") {
            let v = values(s, dtype);
            let expected = reference(s, &v, epi);
            with_cuda_backend(|client, device| {
                let (rtol, atol) = tolerance(&device, dtype, s.contraction());
                let label = format!(
                    "slices [{start}, {}) of {wide}, c_in {c_in}, W {w} [{dtype:?}]",
                    start + c_out
                );
                let make = |data: &[f64], shape: &[usize]| {
                    tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
                };
                let ops = Nchw::new(&client, &device, dtype, s, &v);
                let shape = [BATCH, wide, H, w];
                let numel: usize = shape.iter().product();
                // Input: channels [start, start + c_in) of a wide buffer.
                let x_buf = make(&vec![SENTINEL; numel], &shape);
                let x = x_buf.narrow(1, start, c_in).unwrap();
                client.copy_into(&x, &ops.x).unwrap();
                let res_buf = make(&vec![SENTINEL; numel], &shape);
                let residual = res_buf.narrow(1, start, c_out).unwrap();
                client.copy_into(&residual, &ops.residual).unwrap();
                let out_buf = make(&vec![SENTINEL; numel], &shape);
                let out = out_buf.narrow(1, start, c_out).unwrap();
                ops.fused_into(&client, s, epi, &out, &x, &residual);
                let got = host_f64(&client, &permuted(&out, NCHW_TO_NHWC));
                assert_values_close(&got, &expected, rtol, atol, &label);
                let plane = H * w;
                for (i, val) in host_f64(&client, &out_buf).iter().enumerate() {
                    let c = (i / plane) % wide;
                    if c < start || c >= start + c_out {
                        assert_eq!(
                            val.to_bits(),
                            SENTINEL.to_bits(),
                            "{label}: channel {c} changed"
                        );
                    }
                }
            });
        }
    }
}

/// Input, output and residual as NCHW views of NHWC buffers: W stride C.
/// The input takes the gather, the output one-element stores.
#[test]
fn test_nchw_channels_last_views() {
    let epi = SWEEP_EPILOGUES[0];
    for s in [
        sized(spec("channels_last_cin64_cout64", 64, 64), 17, 24),
        spec("channels_last_cin8_cout32", 8, 32),
    ] {
        for dtype in half_dtypes("cuda") {
            let v = values(s, dtype);
            let expected = reference(s, &v, epi);
            with_cuda_backend(|client, device| {
                let (rtol, atol) = tolerance(&device, dtype, s.contraction());
                let make = |data: &[f64], shape: &[usize]| {
                    tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
                };
                let ops = Nchw::new(&client, &device, dtype, s, &v);
                let x = make(&v.x, &s.input_shape()).permute(&NHWC_TO_NCHW).unwrap();
                let residual = make(&v.residual, &v.out_shape)
                    .permute(&NHWC_TO_NCHW)
                    .unwrap();
                let out_nhwc = make(&vec![SENTINEL; v.residual.len()], &v.out_shape);
                let out = out_nhwc.permute(&NHWC_TO_NCHW).unwrap();
                assert!(!x.is_contiguous() && !out.is_contiguous());
                ops.fused_into(&client, s, epi, &out, &x, &residual);
                let label = format!("channels-last views {} [{dtype:?}]", s.name);
                assert_values_close(&host_f64(&client, &out_nhwc), &expected, rtol, atol, &label);
            });
        }
    }
}
