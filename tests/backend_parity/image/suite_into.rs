// Backend-generic ImageOps checks for `_into`, golden nearest tables,
// `repeat_interleave` equivalence and error cases.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{ConvLayout, ImageOps, NearestSize, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::reference::{assert_bytes_eq, from_canonical, pattern, reference, to_canonical};
use super::suite::{Api, LAYOUTS, bytes_of, call, call_into, cases, make, ref_op};

/// Write each case into a channel slice of a larger buffer. The slice result
/// must match the reference, and every other element keeps its value.
pub fn check_into_channel_slice<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    dtype: DType,
) {
    let elem = dtype.size_in_bytes();
    for layout in LAYOUTS {
        for (dims, api) in cases() {
            let shape = from_canonical(dims, layout);
            let input = pattern(dims.iter().product(), dtype);
            let x = make::<R>(&input, &shape, dtype, device);
            let (want, out_shape) = reference(&input, elem, shape, layout, &ref_op(api, dims));
            let out_dims = to_canonical(out_shape, layout);

            // Two extra channels before the slice and three after it.
            let mut buf_dims = out_dims;
            buf_dims[1] += 5;
            let buf_shape = from_canonical(buf_dims, layout);
            let fill = vec![0xA5u8; buf_shape.iter().product::<usize>() * elem];
            let buf = make::<R>(&fill, &buf_shape, dtype, device);
            let c_axis = if layout == ConvLayout::Nchw { 1 } else { 3 };
            let slice = buf.narrow(c_axis, 2, out_dims[1]).expect("narrow");

            let label = format!("{api:?}_into {layout:?} {dtype:?} channel slice");
            call_into(client, &slice, &x, api, layout).unwrap_or_else(|e| panic!("{label}: {e}"));

            let mut expected = fill.clone();
            let [_, cb, hb, wb] = buf_dims;
            for (i, chunk) in want.chunks(elem).enumerate() {
                let [n, c, h, w] = canonical_coords(i, out_dims, layout);
                let c = c + 2;
                let at = match layout {
                    ConvLayout::Nchw => ((n * cb + c) * hb + h) * wb + w,
                    ConvLayout::Nhwc => ((n * hb + h) * wb + w) * cb + c,
                };
                expected[at * elem..(at + 1) * elem].copy_from_slice(chunk);
            }
            assert_bytes_eq(&bytes_of(&buf), &expected, elem, &label);
        }
    }
}

/// Canonical coordinates of element `i` of a contiguous tensor in `layout`.
fn canonical_coords(i: usize, [_, c, h, w]: [usize; 4], layout: ConvLayout) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => [i / (c * h * w), (i / (h * w)) % c, (i / w) % h, i % w],
        ConvLayout::Nhwc => [i / (h * w * c), i % c, (i / (w * c)) % h, (i / c) % w],
    }
}

/// Input and output as disjoint views of one storage, at nonzero offsets.
pub fn check_into_shared_storage<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    dtype: DType,
) {
    let elem = dtype.size_in_bytes();
    let fill = pattern(80, dtype);
    let buf = make::<R>(&fill, &[80], dtype, device);
    let x = buf
        .narrow(0, 8, 16)
        .expect("narrow")
        .reshape(&[1, 4, 2, 2])
        .expect("reshape");
    let out = buf
        .narrow(0, 40, 16)
        .expect("narrow")
        .reshape(&[1, 1, 4, 4])
        .expect("reshape");
    client
        .pixel_shuffle_into(&out, &x, 2, ConvLayout::Nchw)
        .expect("pixel_shuffle_into on a disjoint view of the input's storage");
    let (want, _) = reference(
        &fill[8 * elem..24 * elem],
        elem,
        [1, 4, 2, 2],
        ConvLayout::Nchw,
        &ref_op(Api::Shuffle(2), [1, 4, 2, 2]),
    );
    let mut expected = fill.clone();
    expected[40 * elem..56 * elem].copy_from_slice(&want);
    assert_bytes_eq(
        &bytes_of(&buf),
        &expected,
        elem,
        &format!("shared storage {dtype:?}"),
    );
}

/// PyTorch CPU `F.interpolate(mode="nearest")` index tables, read through
/// an I32 tensor whose values are their own indices.
pub fn check_nearest_golden<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let golden: [(usize, NearestSize, &[i32]); 11] = [
        (3, NearestSize::Size(1, 7), &[0, 0, 0, 1, 1, 2, 2]),
        (7, NearestSize::Size(1, 3), &[0, 2, 4]),
        (5, NearestSize::Size(1, 8), &[0, 0, 1, 1, 2, 3, 3, 4]),
        (5, NearestSize::Size(1, 3), &[0, 1, 3]),
        (3, NearestSize::Size(1, 6), &[0, 0, 1, 1, 2, 2]),
        (5, NearestSize::Scale(1.0, 1.7), &[0, 0, 1, 1, 2, 2, 3, 4]),
        (3, NearestSize::Scale(1.0, 2.3), &[0, 0, 1, 1, 2, 2]),
        (4, NearestSize::Scale(1.0, 1.5), &[0, 0, 1, 2, 2, 3]),
        (5, NearestSize::Scale(1.0, 0.75), &[0, 1, 2]),
        (3, NearestSize::Scale(1.0, 2.5), &[0, 0, 0, 1, 1, 2, 2]),
        (7, NearestSize::Scale(1.0, 3.0 / 7.0), &[0, 2, 4]),
    ];
    for (in_w, size, table) in golden {
        let values: Vec<i32> = (0..in_w as i32).collect();
        let row = Tensor::<R>::from_slice(&values, &[1, 1, 1, in_w], device).expect("tensor");
        let y = client
            .upsample_nearest2d(&row, size, ConvLayout::Nchw)
            .expect("upsample_nearest2d");
        assert_eq!(y.to_vec::<i32>(), table, "W axis, in {in_w}, {size:?}");

        // The same table along H, in NHWC.
        let col = Tensor::<R>::from_slice(&values, &[1, in_w, 1, 1], device).expect("tensor");
        let swapped = match size {
            NearestSize::Size(h, w) => NearestSize::Size(w, h),
            NearestSize::Scale(h, w) => NearestSize::Scale(w, h),
        };
        let y = client
            .upsample_nearest2d(&col, swapped, ConvLayout::Nhwc)
            .expect("upsample_nearest2d");
        assert_eq!(y.to_vec::<i32>(), table, "H axis, in {in_w}, {swapped:?}");
    }
}

/// Integer factors equal `repeat_interleave` along H then W, in scale mode
/// and in size mode.
pub fn check_repeat_interleave<R: Runtime<DType = DType>, C: ImageOps<R> + ShapeOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let values: Vec<f32> = (0..2 * 3 * 4 * 5).map(|i| i as f32 * 0.5 - 7.0).collect();
    let x = Tensor::<R>::from_slice(&values, &[2, 3, 4, 5], device).expect("tensor");
    let tall = client
        .repeat_interleave(&x, 2, Some(2))
        .expect("repeat_interleave H");
    let want = client
        .repeat_interleave(&tall, 3, Some(3))
        .expect("repeat_interleave W");
    for size in [NearestSize::Scale(2.0, 3.0), NearestSize::Size(8, 15)] {
        let y = client
            .upsample_nearest2d(&x, size, ConvLayout::Nchw)
            .expect("upsample_nearest2d");
        assert_eq!(y.shape(), want.shape(), "{size:?}: shape");
        assert_eq!(
            y.to_vec::<f32>(),
            want.to_vec::<f32>(),
            "{size:?} vs repeat_interleave"
        );
    }
}

fn expect_invalid<T: std::fmt::Debug>(result: numr::error::Result<T>, arg: &str, label: &str) {
    match result {
        Err(Error::InvalidArgument { arg: got, .. }) => assert_eq!(got, arg, "{label}"),
        other => panic!("{label}: expected InvalidArgument for {arg}, got {other:?}"),
    }
}

/// Argument and destination errors.
pub fn check_errors<R: Runtime<DType = DType>, C: ImageOps<R>>(client: &C, device: &R::Device) {
    let t = |shape: &[usize]| Tensor::<R>::zeros(shape, DType::F32, device).expect("zeros");
    let nchw = ConvLayout::Nchw;
    expect_invalid(
        client.pixel_shuffle(&t(&[4, 2, 2]), 2, nchw),
        "input",
        "rank 3",
    );
    expect_invalid(
        client.pixel_shuffle(&t(&[1, 4, 2, 2]), 0, nchw),
        "factor",
        "factor 0",
    );
    expect_invalid(
        client.pixel_shuffle(&t(&[1, 6, 2, 2]), 2, nchw),
        "input",
        "C % r²",
    );
    expect_invalid(
        client.pixel_unshuffle(&t(&[1, 1, 4, 3]), 2, nchw),
        "input",
        "W % r",
    );
    expect_invalid(
        client.pixel_unshuffle(&t(&[1, 1, 3, 4]), 2, nchw),
        "input",
        "H % r",
    );
    let x = t(&[1, 1, 2, 2]);
    for size in [
        NearestSize::Scale(-1.0, 2.0),
        NearestSize::Scale(2.0, f64::INFINITY),
        NearestSize::Size(0, 2),
    ] {
        expect_invalid(
            client.upsample_nearest2d(&x, size, nchw),
            "size",
            &format!("{size:?}"),
        );
    }

    let x = t(&[1, 4, 2, 2]);
    let wrong_shape = t(&[1, 1, 4, 5]);
    assert!(
        matches!(
            client.pixel_shuffle_into(&wrong_shape, &x, 2, nchw),
            Err(Error::ShapeMismatch { .. })
        ),
        "out shape mismatch"
    );
    let wrong_dtype = Tensor::<R>::zeros(&[1, 1, 4, 4], DType::I32, device).expect("zeros");
    assert!(
        matches!(
            client.pixel_shuffle_into(&wrong_dtype, &x, 2, nchw),
            Err(Error::DTypeMismatch { .. })
        ),
        "out dtype mismatch"
    );
    let alias = x.reshape(&[1, 1, 4, 4]).expect("reshape");
    expect_invalid(
        client.pixel_shuffle_into(&alias, &x, 2, nchw),
        "out",
        "out aliases input",
    );
}

/// A zero batch gives an empty output of the right shape.
pub fn check_empty_batch<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let x = Tensor::<R>::zeros(&[0, 4, 2, 3], DType::F32, device).expect("zeros");
    for (api, shape) in [
        (Api::Shuffle(2), [0, 1, 4, 6]),
        (Api::Unshuffle(1), [0, 4, 2, 3]),
        (Api::Nearest(NearestSize::Size(5, 7)), [0, 4, 5, 7]),
    ] {
        let y = call(client, &x, api, ConvLayout::Nchw).expect("empty batch");
        assert_eq!(y.shape(), shape, "{api:?} empty batch");
    }
}
