// ColorOps argument errors: odd sizes, plane dtypes and shapes, RGB shapes,
// unsupported dtypes and overlapping written views.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, YuvPlanes, YuvSpec};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

fn zeros<R: Runtime<DType = DType>>(shape: &[usize], dtype: DType, d: &R::Device) -> Tensor<R> {
    Tensor::<R>::zeros(shape, dtype, d).expect("zeros")
}

/// Assert `InvalidArgument` for `arg` whose reason contains `text`.
fn invalid<T: std::fmt::Debug>(result: numr::error::Result<T>, arg: &str, text: &str) {
    match result {
        Err(Error::InvalidArgument { arg: a, reason }) => {
            assert_eq!(a, arg, "wrong argument in: {reason}");
            assert!(reason.contains(text), "reason {reason:?} lacks {text:?}");
        }
        other => panic!("expected InvalidArgument for {arg} ({text}), got {other:?}"),
    }
}

/// Every documented error of the YCbCr methods.
pub fn check_yuv_errors<R: Runtime<DType = DType>, C: ColorOps<R>>(client: &C, d: &R::Device) {
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let (y, uv) = (
        zeros::<R>(&[4, 6], DType::U8, d),
        zeros::<R>(&[2, 6], DType::U8, d),
    );
    let f32_out = |shape: &[usize]| zeros::<R>(shape, DType::F32, d);

    let odd_w = zeros::<R>(&[4, 5], DType::U8, d);
    let planes = YuvPlanes::Nv12 { y: &odd_w, uv: &uv };
    invalid(
        client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw),
        "planes",
        "W = 5",
    );
    let odd_h = zeros::<R>(&[3, 6], DType::U8, d);
    let planes = YuvPlanes::Nv12 { y: &odd_h, uv: &uv };
    invalid(
        client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw),
        "planes",
        "H = 3",
    );

    let planes = YuvPlanes::P010 { y: &y, uv: &uv };
    invalid(
        client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw),
        "planes",
        "U16",
    );
    let narrow_uv = zeros::<R>(&[2, 3], DType::U8, d);
    let planes = YuvPlanes::Nv12 {
        y: &y,
        uv: &narrow_uv,
    };
    invalid(
        client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw),
        "planes",
        "plane uv",
    );
    let (u, v) = (
        zeros::<R>(&[2, 3], DType::U8, d),
        zeros::<R>(&[2, 6], DType::U8, d),
    );
    let planes = YuvPlanes::I420 {
        y: &y,
        u: &u,
        v: &v,
    };
    invalid(
        client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw),
        "planes",
        "plane v",
    );

    let planes = YuvPlanes::Nv12 { y: &y, uv: &uv };
    assert!(matches!(
        client.yuv_to_rgb(planes, &spec, DType::I32, ConvLayout::Nchw),
        Err(Error::UnsupportedDType { .. })
    ));
    assert!(matches!(
        client.yuv_to_rgb_into(&f32_out(&[1, 3, 4, 5]), planes, &spec, ConvLayout::Nchw),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        client.yuv_to_rgb_into(&f32_out(&[1, 3, 4, 6]), planes, &spec, ConvLayout::Nhwc),
        Err(Error::ShapeMismatch { .. })
    ));

    let batch = f32_out(&[2, 3, 4, 6]);
    invalid(
        client.rgb_to_yuv_into(&batch, planes, &spec, ConvLayout::Nchw),
        "rgb",
        "one RGB frame",
    );
    assert!(matches!(
        client.rgb_to_yuv_into(&f32_out(&[1, 3, 4, 8]), planes, &spec, ConvLayout::Nchw),
        Err(Error::ShapeMismatch { .. })
    ));
    let int_rgb = zeros::<R>(&[1, 3, 4, 6], DType::I32, d);
    assert!(matches!(
        client.rgb_to_yuv_into(&int_rgb, planes, &spec, ConvLayout::Nchw),
        Err(Error::UnsupportedDType { .. })
    ));

    let surface = zeros::<R>(&[6, 8], DType::U8, d);
    let y_view = surface
        .narrow(0, 0, 4)
        .and_then(|t| t.narrow(1, 0, 6))
        .expect("y");
    let uv_view = surface
        .narrow(0, 3, 2)
        .and_then(|t| t.narrow(1, 0, 6))
        .expect("uv");
    let planes = YuvPlanes::Nv12 {
        y: &y_view,
        uv: &uv_view,
    };
    let rgb = f32_out(&[1, 3, 4, 6]);
    invalid(
        client.rgb_to_yuv_into(&rgb, planes, &spec, ConvLayout::Nchw),
        "planes",
        "overlaps",
    );
}

/// Every documented error of the RGB24 methods.
pub fn check_rgb24_errors<R: Runtime<DType = DType>, C: ColorOps<R>>(client: &C, d: &R::Device) {
    let not_u8 = zeros::<R>(&[2, 2, 3], DType::F32, d);
    invalid(
        client.rgb24_to_float(&not_u8, DType::F32, ConvLayout::Nchw),
        "rgb24",
        "U8",
    );
    let four = zeros::<R>(&[2, 2, 4], DType::U8, d);
    invalid(
        client.rgb24_to_float(&four, DType::F32, ConvLayout::Nchw),
        "rgb24",
        "[H, W, 3]",
    );
    let good = zeros::<R>(&[2, 2, 3], DType::U8, d);
    assert!(matches!(
        client.rgb24_to_float(&good, DType::U8, ConvLayout::Nchw),
        Err(Error::UnsupportedDType { .. })
    ));

    let rgb = zeros::<R>(&[1, 3, 2, 2], DType::F32, d);
    invalid(
        client.float_to_rgb24(&rgb, ConvLayout::Nhwc),
        "rgb",
        "one RGB frame",
    );
    let f32_out = zeros::<R>(&[2, 2, 3], DType::F32, d);
    invalid(
        client.float_to_rgb24_into(&f32_out, &rgb, ConvLayout::Nchw),
        "out",
        "U8",
    );
    assert!(matches!(
        client.float_to_rgb24_into(
            &zeros::<R>(&[2, 3, 3], DType::U8, d),
            &rgb,
            ConvLayout::Nchw
        ),
        Err(Error::ShapeMismatch { .. })
    ));
}
