// ColorOps argument errors of the packed forms: odd sizes, word dtypes,
// short rows, strided words, plane extents, overlapping written words, word
// pitches and packed RGB24 shapes.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{
    ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, PackedPlane, YuvPlanes, YuvSpec,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

fn zeros<R: Runtime<DType = DType>>(shape: &[usize], dtype: DType, d: &R::Device) -> Tensor<R> {
    Tensor::<R>::zeros(shape, dtype, d).expect("zeros")
}

/// Assert `InvalidArgument` for `arg` whose reason contains `text`.
pub fn invalid<T: std::fmt::Debug>(result: numr::error::Result<T>, arg: &str, text: &str) {
    match result {
        Err(Error::InvalidArgument { arg: a, reason }) => {
            assert_eq!(a, arg, "wrong argument in: {reason}");
            assert!(reason.contains(text), "reason {reason:?} lacks {text:?}");
        }
        other => panic!("expected InvalidArgument for {arg} ({text}), got {other:?}"),
    }
}

fn decode<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    planes: YuvPlanes<'_, R>,
) -> numr::error::Result<Tensor<R>> {
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    client.yuv_to_rgb(planes, &spec, DType::F32, ConvLayout::Nchw)
}

fn nv12<'a, R: Runtime>(y: &'a Tensor<R>, yw: usize, uv: &'a Tensor<R>) -> YuvPlanes<'a, R> {
    YuvPlanes::Nv12Packed {
        y: PackedPlane::new(y, yw),
        uv: PackedPlane::new(uv, yw),
    }
}

/// Every documented error of the YCbCr methods on packed planes.
pub fn check_packed_yuv_errors<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    d: &R::Device,
) {
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let words = |shape: &[usize]| zeros::<R>(shape, DType::U32, d);
    let (y, uv) = (words(&[4, 2]), words(&[2, 2]));

    invalid(decode(client, nv12(&y, 5, &uv)), "planes", "W = 5");
    let short = words(&[3, 2]);
    invalid(decode(client, nv12(&short, 6, &uv)), "planes", "H = 3");
    let bytes = zeros::<R>(&[4, 8], DType::U8, d);
    invalid(
        decode(client, nv12(&bytes, 6, &uv)),
        "planes",
        "expected U32",
    );
    let narrow = words(&[4, 1]);
    invalid(decode(client, nv12(&narrow, 6, &uv)), "planes", "need 2");
    let strided = words(&[2, 4]).transpose(0, 1).expect("transpose");
    invalid(
        decode(client, nv12(&strided, 6, &uv)),
        "planes",
        "column stride",
    );
    let planes = YuvPlanes::Nv12Packed {
        y: PackedPlane::new(&y, 6),
        uv: PackedPlane::new(&uv, 4),
    };
    invalid(decode(client, planes), "planes", "plane uv");
    let (u, v) = (words(&[2, 1]), words(&[1, 1]));
    let planes = YuvPlanes::I420Packed {
        y: PackedPlane::new(&y, 6),
        u: PackedPlane::new(&u, 3),
        v: PackedPlane::new(&v, 3),
    };
    invalid(decode(client, planes), "planes", "plane v");
    let p010 = YuvPlanes::P010Packed {
        y: PackedPlane::new(&y, 6),
        uv: PackedPlane::new(&uv, 6),
    };
    invalid(decode(client, p010), "planes", "need 3");

    let planes = nv12(&y, 6, &uv);
    assert!(matches!(
        client.yuv_to_rgb(planes, &spec, DType::I32, ConvLayout::Nchw),
        Err(Error::UnsupportedDType { .. })
    ));
    let int_rgb = zeros::<R>(&[1, 3, 4, 6], DType::I32, d);
    assert!(matches!(
        client.rgb_to_yuv_into(&int_rgb, planes, &spec, ConvLayout::Nchw),
        Err(Error::UnsupportedDType { .. })
    ));
    let wrong = zeros::<R>(&[1, 3, 4, 8], DType::F32, d);
    assert!(matches!(
        client.rgb_to_yuv_into(&wrong, planes, &spec, ConvLayout::Nchw),
        Err(Error::ShapeMismatch { .. })
    ));

    let surface = words(&[6, 2]);
    let y_view = surface.narrow(0, 0, 4).expect("y");
    let uv_view = surface.narrow(0, 3, 2).expect("uv");
    let rgb = zeros::<R>(&[1, 3, 4, 6], DType::F32, d);
    invalid(
        client.rgb_to_yuv_into(&rgb, nv12(&y_view, 6, &uv_view), &spec, ConvLayout::Nchw),
        "planes",
        "overlaps",
    );

    invalid(
        PackedPlane::<R>::words_shape(4, 6),
        "pitch_bytes",
        "align the pitch to 4 bytes",
    );
}

/// Every documented error of the packed RGB24 methods.
pub fn check_packed_rgb24_errors<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    d: &R::Device,
) {
    let bytes = zeros::<R>(&[2, 8], DType::U8, d);
    invalid(
        client.packed_rgb24_to_float(PackedPlane::new(&bytes, 2), DType::F32, ConvLayout::Nchw),
        "rgb24",
        "expected U32",
    );
    let short = zeros::<R>(&[2, 1], DType::U32, d);
    invalid(
        client.packed_rgb24_to_float(PackedPlane::new(&short, 2), DType::F32, ConvLayout::Nchw),
        "rgb24",
        "need 2",
    );
    let good = zeros::<R>(&[2, 2], DType::U32, d);
    assert!(matches!(
        client.packed_rgb24_to_float(PackedPlane::new(&good, 2), DType::I32, ConvLayout::Nchw),
        Err(Error::UnsupportedDType { .. })
    ));

    let rgb = zeros::<R>(&[1, 3, 2, 2], DType::F32, d);
    invalid(
        client.float_to_packed_rgb24(&rgb, ConvLayout::Nhwc),
        "rgb",
        "one RGB frame",
    );
    let wide = zeros::<R>(&[2, 3], DType::U32, d);
    assert!(matches!(
        client.float_to_packed_rgb24_into(PackedPlane::new(&wide, 3), &rgb, ConvLayout::Nchw),
        Err(Error::ShapeMismatch { .. })
    ));
    invalid(
        client.float_to_packed_rgb24_into(PackedPlane::new(&bytes, 2), &rgb, ConvLayout::Nchw),
        "out",
        "expected U32",
    );
    let repeated = zeros::<R>(&[1, 2], DType::U32, d)
        .broadcast_to(&[2, 2])
        .expect("broadcast");
    let packed = PackedPlane::new(&repeated, 2);
    invalid(
        client.float_to_packed_rgb24_into(packed, &rgb, ConvLayout::Nchw),
        "out",
        "twice",
    );
    let float_out = zeros::<R>(&[1, 3, 2, 3], DType::F32, d);
    let packed = PackedPlane::new(&good, 2);
    assert!(matches!(
        client.packed_rgb24_to_float_into(&float_out, packed, ConvLayout::Nchw),
        Err(Error::ShapeMismatch { .. })
    ));
}
