// Backend-generic ImageOps checks. cpu.rs, cuda.rs and wgpu.rs run them on
// their own client, so every backend meets the same host reference.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ConvLayout, ImageOps, NearestSize};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::reference::{
    RefOp, assert_bytes_eq, from_canonical, nearest_table, pattern, reference, scaled_extent,
    to_canonical,
};

/// One call of the API under test.
#[derive(Debug, Clone, Copy)]
pub enum Api {
    /// `pixel_shuffle(factor)`.
    Shuffle(usize),
    /// `pixel_unshuffle(factor)`.
    Unshuffle(usize),
    /// `upsample_nearest2d(size)`.
    Nearest(NearestSize),
}

pub const LAYOUTS: [ConvLayout; 2] = [ConvLayout::Nchw, ConvLayout::Nhwc];

/// Reference op of `api` on canonical input extents.
pub fn ref_op(api: Api, [_, _, h, w]: [usize; 4]) -> RefOp {
    match api {
        Api::Shuffle(r) => RefOp::Shuffle(r),
        Api::Unshuffle(r) => RefOp::Unshuffle(r),
        Api::Nearest(NearestSize::Scale(sh, sw)) => RefOp::Nearest {
            rows: nearest_table(h, scaled_extent(h, sh), Some(sh)),
            cols: nearest_table(w, scaled_extent(w, sw), Some(sw)),
        },
        Api::Nearest(NearestSize::Size(oh, ow)) => RefOp::Nearest {
            rows: nearest_table(h, oh, None),
            cols: nearest_table(w, ow, None),
        },
    }
}

pub fn make<R: Runtime<DType = DType>>(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> Tensor<R> {
    Tensor::<R>::from_bytes(bytes, shape, dtype, device).expect("tensor from bytes")
}

pub fn bytes_of<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Vec<u8> {
    t.to_bytes().expect("tensor readback")
}

pub fn call<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    x: &Tensor<R>,
    api: Api,
    layout: ConvLayout,
) -> Result<Tensor<R>> {
    match api {
        Api::Shuffle(r) => client.pixel_shuffle(x, r, layout),
        Api::Unshuffle(r) => client.pixel_unshuffle(x, r, layout),
        Api::Nearest(size) => client.upsample_nearest2d(x, size, layout),
    }
}

pub fn call_into<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    out: &Tensor<R>,
    x: &Tensor<R>,
    api: Api,
    layout: ConvLayout,
) -> Result<()> {
    match api {
        Api::Shuffle(r) => client.pixel_shuffle_into(out, x, r, layout),
        Api::Unshuffle(r) => client.pixel_unshuffle_into(out, x, r, layout),
        Api::Nearest(size) => client.upsample_nearest2d_into(out, x, size, layout),
    }
}

/// Canonical input extents and the API call of every basic case. Batch 2.
pub fn cases() -> Vec<([usize; 4], Api)> {
    vec![
        ([2, 8, 3, 5], Api::Shuffle(2)),
        ([2, 18, 2, 3], Api::Shuffle(3)),
        ([2, 3, 2, 2], Api::Shuffle(1)),
        ([2, 2, 6, 4], Api::Unshuffle(2)),
        ([2, 1, 6, 9], Api::Unshuffle(3)),
        ([2, 3, 3, 5], Api::Nearest(NearestSize::Scale(2.0, 3.0))),
        ([2, 3, 3, 5], Api::Nearest(NearestSize::Scale(2.3, 1.7))),
        (
            [2, 2, 7, 4],
            Api::Nearest(NearestSize::Scale(3.0 / 7.0, 1.5)),
        ),
        ([2, 2, 3, 5], Api::Nearest(NearestSize::Size(7, 8))),
        ([2, 2, 7, 5], Api::Nearest(NearestSize::Size(3, 3))),
    ]
}

/// Run `api` on a contiguous input and compare with the reference.
pub fn check_case<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    dtype: DType,
    dims: [usize; 4],
    api: Api,
    layout: ConvLayout,
) {
    let shape = from_canonical(dims, layout);
    let numel: usize = dims.iter().product();
    let input = pattern(numel, dtype);
    let x = make::<R>(&input, &shape, dtype, device);
    let label = format!("{api:?} {layout:?} {dtype:?} input {shape:?}");
    let y = call(client, &x, api, layout).unwrap_or_else(|e| panic!("{label}: {e}"));
    let elem = dtype.size_in_bytes();
    let (want, want_shape) = reference(&input, elem, shape, layout, &ref_op(api, dims));
    assert_eq!(y.shape(), want_shape, "{label}: output shape");
    assert_eq!(y.dtype(), dtype, "{label}: output dtype");
    assert_bytes_eq(&bytes_of(&y), &want, elem, &label);
}

/// Every basic case in both layouts for `dtype`.
pub fn check_dtype<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    dtype: DType,
) {
    for layout in LAYOUTS {
        for (dims, api) in cases() {
            check_case::<R, C>(client, device, dtype, dims, api, layout);
        }
    }
}

/// `unshuffle(shuffle(x)) == x` and `shuffle(unshuffle(x)) == x`.
pub fn check_roundtrip<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    dtype: DType,
) {
    let elem = dtype.size_in_bytes();
    for layout in LAYOUTS {
        for (dims, r) in [([2, 12, 3, 4], 2), ([2, 9, 2, 5], 3)] {
            let shape = from_canonical(dims, layout);
            let input = pattern(dims.iter().product(), dtype);
            let x = make::<R>(&input, &shape, dtype, device);
            let up = client.pixel_shuffle(&x, r, layout).expect("pixel_shuffle");
            let back = client
                .pixel_unshuffle(&up, r, layout)
                .expect("pixel_unshuffle");
            let label = format!("unshuffle(shuffle) r={r} {layout:?} {dtype:?}");
            assert_eq!(back.shape(), &shape[..], "{label}: shape");
            assert_bytes_eq(&bytes_of(&back), &input, elem, &label);

            let down = client
                .pixel_unshuffle(&up, r, layout)
                .expect("pixel_unshuffle");
            let again = client
                .pixel_shuffle(&down, r, layout)
                .expect("pixel_shuffle");
            let label = format!("shuffle(unshuffle) r={r} {layout:?} {dtype:?}");
            assert_bytes_eq(&bytes_of(&again), &bytes_of(&up), elem, &label);
        }
    }
}

/// An NHWC call equals the NCHW call with dimensions permuted.
pub fn check_nhwc_matches_nchw<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    dtype: DType,
) {
    let elem = dtype.size_in_bytes();
    for (dims, api) in cases() {
        let input = pattern(dims.iter().product(), dtype);
        let x_nchw = make::<R>(&input, &dims, dtype, device);
        let nhwc_bytes = bytes_of(&x_nchw.permute(&[0, 2, 3, 1]).expect("permute"));
        let x_nhwc = make::<R>(
            &nhwc_bytes,
            &from_canonical(dims, ConvLayout::Nhwc),
            dtype,
            device,
        );
        let y_nchw = call(client, &x_nchw, api, ConvLayout::Nchw).expect("NCHW call");
        let y_nhwc = call(client, &x_nhwc, api, ConvLayout::Nhwc).expect("NHWC call");
        let permuted = y_nchw.permute(&[0, 2, 3, 1]).expect("permute");
        let label = format!("{api:?} NHWC vs permuted NCHW {dtype:?}");
        assert_eq!(y_nhwc.shape(), permuted.shape(), "{label}: shape");
        assert_bytes_eq(&bytes_of(&y_nhwc), &bytes_of(&permuted), elem, &label);
    }
}

/// Strided input views: a W slice of a wider buffer and an NHWC buffer
/// read through an NCHW permute.
pub fn check_strided_input<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    dtype: DType,
) {
    let elem = dtype.size_in_bytes();
    for layout in LAYOUTS {
        for (dims, api) in cases() {
            let shape = from_canonical(dims, layout);
            let mut wide = shape;
            wide[3] += 3;
            let buf = make::<R>(&pattern(wide.iter().product(), dtype), &wide, dtype, device);
            let sliced = buf.narrow(3, 2, shape[3]).expect("narrow");
            let label = format!("{api:?} {layout:?} {dtype:?} sliced input");
            check_view(client, &sliced, api, layout, elem, &label);
        }
    }
    for (dims, api) in cases() {
        let nhwc = from_canonical(dims, ConvLayout::Nhwc);
        let buf = make::<R>(&pattern(dims.iter().product(), dtype), &nhwc, dtype, device);
        let view = buf.permute(&[0, 3, 1, 2]).expect("permute");
        let label = format!("{api:?} NCHW {dtype:?} permuted input");
        check_view(client, &view, api, ConvLayout::Nchw, elem, &label);
    }
}

fn check_view<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    view: &Tensor<R>,
    api: Api,
    layout: ConvLayout,
    elem: usize,
    label: &str,
) {
    let shape: [usize; 4] = view.shape().try_into().expect("rank 4");
    let logical = bytes_of(view);
    let op = ref_op(api, to_canonical(shape, layout));
    let (want, want_shape) = reference(&logical, elem, shape, layout, &op);
    let y = call(client, view, api, layout).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(y.shape(), want_shape, "{label}: shape");
    assert_bytes_eq(&bytes_of(&y), &want, elem, label);
}
