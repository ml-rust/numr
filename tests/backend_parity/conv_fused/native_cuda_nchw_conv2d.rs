// CUDA native plain `conv2d` in NCHW. Operands and helpers come from
// `native_cuda_nchw`.

use numr::ops::{ConvOps, PaddingMode};

use super::native_cuda::{Epilogue, Spec, assert_values_close, host_f64, reference, spec, values};
use super::native_cuda_nchw::{Nchw, nchw_shape, permuted, sized, tolerance};
use super::{Act, NCHW_TO_NHWC, NHWC_TO_NCHW};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::half_dtypes;

/// Plain `conv2d` in NCHW, with and without bias, against the same
/// reference. The 64 to 64 3x3 case runs on a 40x72 image.
#[test]
fn test_nchw_conv2d_matches_cpu() {
    let specs = [
        Spec {
            batch: 1,
            ..sized(spec("conv2d_cin64_cout64_40x72", 64, 64), 40, 72)
        },
        sized(spec("conv2d_halo_cin72_cout96", 72, 96), 17, 36),
        spec("conv2d_gather_cin3_cout32", 3, 32),
        Spec {
            stride: (2, 2),
            padding: PaddingMode::Custom(0, 1, 1, 0),
            ..spec("conv2d_cin64_cout4_stride2", 64, 4)
        },
        Spec {
            kernel: 5,
            dilation: (2, 2),
            ..spec("conv2d_cin8_cout48_5x5_dilation2", 8, 48)
        },
    ];
    for s in specs {
        for dtype in half_dtypes("cuda") {
            let v = values(s, dtype);
            with_cuda_backend(|client, device| {
                let (rtol, atol) = tolerance(&device, dtype, s.contraction());
                let ops = Nchw::new(&client, &device, dtype, s, &v);
                for bias in [false, true] {
                    let epi = Epilogue {
                        act: Act::None,
                        bias,
                        residual_scale: None,
                    };
                    let label = format!("conv2d NCHW {} bias {bias} [{dtype:?}]", s.name);
                    let got = client
                        .conv2d(
                            &ops.x,
                            &ops.w,
                            bias.then_some(&ops.bias),
                            s.stride,
                            s.padding,
                            s.dilation,
                            1,
                        )
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(
                        got.shape(),
                        nchw_shape(&v.out_shape).as_slice(),
                        "{label}: shape"
                    );
                    let got = host_f64(&client, &permuted(&got, NCHW_TO_NHWC));
                    assert_values_close(&got, &reference(s, &v, epi), rtol, atol, &label);
                }
            });
        }
    }
}

/// `conv2d` on a strided input: a channels-last buffer viewed as NCHW. The
/// native route reads the view as it is.
#[test]
fn test_nchw_conv2d_strided_input() {
    let s = spec("conv2d_channels_last_cin16_cout32", 16, 32);
    let epi = Epilogue {
        act: Act::None,
        bias: true,
        residual_scale: None,
    };
    for dtype in half_dtypes("cuda") {
        let v = values(s, dtype);
        with_cuda_backend(|client, device| {
            let (rtol, atol) = tolerance(&device, dtype, s.contraction());
            let ops = Nchw::new(&client, &device, dtype, s, &v);
            let x = tensor_from_f64(&v.x, &s.input_shape(), dtype, &device, &client)
                .unwrap()
                .permute(&NHWC_TO_NCHW)
                .unwrap();
            let got = client
                .conv2d(
                    &x,
                    &ops.w,
                    Some(&ops.bias),
                    s.stride,
                    s.padding,
                    s.dilation,
                    1,
                )
                .unwrap();
            let got = host_f64(&client, &permuted(&got, NCHW_TO_NHWC));
            let label = format!("conv2d channels-last input [{dtype:?}]");
            assert_values_close(&got, &reference(s, &v, epi), rtol, atol, &label);
        });
    }
}
