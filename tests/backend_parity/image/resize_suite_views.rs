// Backend-generic ImageOps::resize checks for strided input views, `_into`
// channel slices, equal sizes and error cases.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{ConvLayout, ImageOps, ResizeConfig, ResizeFilter};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::resize_golden::{Block, Kind, blocks, from_layout, input, shape, to_layout};
use super::resize_suite::LAYOUTS;

const FILL: f32 = 7.5;

/// Blocks with a batch, channels, or a non-integer ratio.
fn view_blocks() -> Vec<Block> {
    blocks()
        .into_iter()
        .filter(|b| b.kind == Kind::F32 && (b.dims == [2, 3, 8, 9] || b.dims == [1, 1, 17, 23]))
        .collect()
}

fn dense<R: Runtime<DType = DType>>(
    x: &[f32],
    dims: [usize; 4],
    layout: ConvLayout,
    device: &R::Device,
) -> Tensor<R> {
    Tensor::<R>::from_slice(&to_layout(x, dims, layout), &shape(dims, layout), device)
        .expect("F32 tensor")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Row-major index of `coords` in a tensor of `shape`.
fn flat(coords: [usize; 4], shape: [usize; 4]) -> usize {
    ((coords[0] * shape[1] + coords[1]) * shape[2] + coords[2]) * shape[3] + coords[3]
}

/// A W slice of a wider buffer and an NHWC buffer read through an NCHW
/// permute give the same bits as a dense input.
pub fn check_resize_strided_input<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    for block in view_blocks() {
        let cfg = block.config();
        let x = input(block.dims);
        for layout in LAYOUTS {
            let label = format!("{} {layout:?} W slice", block.label());
            let want = client
                .resize(&dense::<R>(&x, block.dims, layout, device), &cfg, layout)
                .expect("dense resize")
                .to_vec::<f32>();
            let s = shape(block.dims, layout);
            let w_axis: usize = if layout == ConvLayout::Nchw { 3 } else { 2 };
            let mut wide = s;
            wide[w_axis] += 3;
            let laid = to_layout(&x, block.dims, layout);
            let mut host = vec![-9.0f32; wide.iter().product()];
            for n in 0..s[0] {
                for a in 0..s[1] {
                    for b in 0..s[2] {
                        for c in 0..s[3] {
                            let mut at = [n, a, b, c];
                            at[w_axis] += 2;
                            host[flat(at, wide)] = laid[flat([n, a, b, c], s)];
                        }
                    }
                }
            }
            let buf = Tensor::<R>::from_slice(&host, &wide, device).expect("wide buffer");
            let view = buf.narrow(w_axis as isize, 2, s[w_axis]).expect("narrow");
            let got = client
                .resize(&view, &cfg, layout)
                .unwrap_or_else(|e| panic!("{label}: {e}"))
                .to_vec::<f32>();
            assert_eq!(bits(&got), bits(&want), "{label}");
        }

        let label = format!("{} NCHW over a permuted NHWC buffer", block.label());
        let want = client
            .resize(
                &dense::<R>(&x, block.dims, ConvLayout::Nchw, device),
                &cfg,
                ConvLayout::Nchw,
            )
            .expect("dense resize")
            .to_vec::<f32>();
        let nhwc = dense::<R>(&x, block.dims, ConvLayout::Nhwc, device);
        let view = nhwc.permute(&[0, 3, 1, 2]).expect("permute");
        let got = client
            .resize(&view, &cfg, ConvLayout::Nchw)
            .unwrap_or_else(|e| panic!("{label}: {e}"))
            .to_vec::<f32>();
        assert_eq!(bits(&got), bits(&want), "{label}");
    }
}

/// `resize_into` a channel slice of a larger buffer gives the bits of
/// `resize`, and every other element keeps its value.
pub fn check_resize_into_channel_slice<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    for block in view_blocks() {
        let cfg = block.config();
        let x = input(block.dims);
        for layout in LAYOUTS {
            let label = format!("{} {layout:?} resize_into channel slice", block.label());
            let src = dense::<R>(&x, block.dims, layout, device);
            let want = from_layout(
                &client
                    .resize(&src, &cfg, layout)
                    .expect("resize")
                    .to_vec::<f32>(),
                block.out_dims(),
                layout,
            );
            let [n, c, h, w] = block.out_dims();
            let buf_dims = [n, c + 5, h, w];
            let buf_shape = shape(buf_dims, layout);
            let fill = vec![FILL; buf_shape.iter().product()];
            let buf = Tensor::<R>::from_slice(&fill, &buf_shape, device).expect("buffer");
            let c_axis: isize = if layout == ConvLayout::Nchw { 1 } else { 3 };
            let slice = buf.narrow(c_axis, 2, c).expect("narrow");
            client
                .resize_into(&slice, &src, &cfg, layout)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let got = from_layout(&buf.to_vec::<f32>(), buf_dims, layout);
            for ni in 0..n {
                for ci in 0..c + 5 {
                    for hi in 0..h {
                        for wi in 0..w {
                            let g = got[flat([ni, ci, hi, wi], buf_dims)];
                            let e = if (2..2 + c).contains(&ci) {
                                want[flat([ni, ci - 2, hi, wi], block.out_dims())]
                            } else {
                                FILL
                            };
                            assert_eq!(
                                g.to_bits(),
                                e.to_bits(),
                                "{label}: element ({ni}, {ci}, {hi}, {wi})"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Equal sizes. With antialias the input copies, `inf` included. Without
/// antialias the passes still run, as PyTorch's do: finite values come back
/// unchanged, and an `inf` meets a zero weight and becomes NaN.
pub fn check_resize_equal_sizes<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let dims = [2, 3, 5, 7];
    let mut x = input(dims);
    for layout in LAYOUTS {
        for filter in [ResizeFilter::Bilinear, ResizeFilter::Bicubic] {
            for aa in [false, true] {
                let cfg = ResizeConfig::new((5, 7), filter).with_antialias(aa);
                let label = format!("equal sizes {filter:?} antialias={aa} {layout:?}");
                let y = client
                    .resize(&dense::<R>(&x, dims, layout, device), &cfg, layout)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let got = from_layout(&y.to_vec::<f32>(), dims, layout);
                assert_eq!(bits(&got), bits(&x), "{label}");
            }
        }
    }

    let at = flat([0, 0, 2, 3], dims);
    x[at] = f32::INFINITY;
    let run = |filter, aa| {
        let cfg = ResizeConfig::new((5, 7), filter).with_antialias(aa);
        let y = client
            .resize(
                &dense::<R>(&x, dims, ConvLayout::Nchw, device),
                &cfg,
                ConvLayout::Nchw,
            )
            .expect("resize");
        y.to_vec::<f32>()
    };
    // PyTorch 2.8 CPU gives the same values and NaN counts.
    let nans = |v: &[f32]| v.iter().filter(|f| f.is_nan()).count();
    for filter in [ResizeFilter::Bilinear, ResizeFilter::Bicubic] {
        let aa = run(filter, true);
        assert_eq!(aa[at], f32::INFINITY, "{filter:?} antialias keeps inf");
        assert_eq!(nans(&aa), 0, "{filter:?} antialias NaN count");
    }
    let linear = run(ResizeFilter::Bilinear, false);
    assert!(linear[at].is_nan(), "bilinear: inf * 0 at the inf pixel");
    assert_eq!(nans(&linear), 1, "bilinear NaN count");
    let cubic = run(ResizeFilter::Bicubic, false);
    assert_eq!(
        cubic[at],
        f32::INFINITY,
        "bicubic keeps inf at the inf pixel"
    );
    assert!(
        cubic[flat([0, 0, 2, 4], dims)].is_nan(),
        "bicubic W neighbour"
    );
    assert!(
        cubic[flat([0, 0, 1, 3], dims)].is_nan(),
        "bicubic H neighbour"
    );
    assert_eq!(nans(&cubic), 15, "bicubic NaN count");
}

fn invalid<T: std::fmt::Debug>(result: numr::error::Result<T>, want: &str, label: &str) {
    match result {
        Err(Error::InvalidArgument { arg, .. }) => assert_eq!(arg, want, "{label}"),
        other => panic!("{label}: expected InvalidArgument({want}), got {other:?}"),
    }
}

/// Rank, size, dtype and `out` errors name what failed.
pub fn check_resize_errors<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let cfg = ResizeConfig::new((2, 2), ResizeFilter::Bilinear);
    let x = Tensor::<R>::from_slice(&[0.5f32; 16], &[1, 1, 4, 4], device).expect("input");

    let rank3 = Tensor::<R>::from_slice(&[0.5f32; 16], &[1, 4, 4], device).expect("rank 3");
    invalid(
        client.resize(&rank3, &cfg, ConvLayout::Nchw),
        "input",
        "rank 3",
    );
    let empty = Tensor::<R>::zeros(&[1, 1, 0, 4], DType::F32, device).expect("H = 0");
    invalid(
        client.resize(&empty, &cfg, ConvLayout::Nchw),
        "input",
        "H = 0",
    );
    for size in [(0, 2), (2, 0)] {
        let zero = ResizeConfig::new(size, ResizeFilter::Bicubic).with_antialias(true);
        invalid(
            client.resize(&x, &zero, ConvLayout::Nchw),
            "config",
            &format!("size {size:?}"),
        );
    }

    let ints = Tensor::<R>::from_slice(&[1i32; 16], &[1, 1, 4, 4], device).expect("I32");
    match client.resize(&ints, &cfg, ConvLayout::Nchw) {
        Err(Error::UnsupportedDType { dtype, op }) => {
            assert_eq!(dtype, DType::I32);
            assert_eq!(op, "resize");
        }
        other => panic!("I32: expected UnsupportedDType, got {other:?}"),
    }

    let wrong = Tensor::<R>::zeros(&[1, 1, 3, 2], DType::F32, device).expect("out");
    assert!(
        matches!(
            client.resize_into(&wrong, &x, &cfg, ConvLayout::Nchw),
            Err(Error::ShapeMismatch { .. })
        ),
        "resize_into with the wrong out shape"
    );
    let other = Tensor::<R>::zeros(&[1, 1, 2, 2], DType::I32, device).expect("out");
    assert!(
        matches!(
            client.resize_into(&other, &x, &cfg, ConvLayout::Nchw),
            Err(Error::DTypeMismatch { .. })
        ),
        "resize_into with an I32 out"
    );
    let alias = x.narrow(2, 0, 2).expect("narrow");
    let alias = alias.narrow(3, 0, 2).expect("narrow");
    assert!(
        matches!(
            client.resize_into(&alias, &x, &cfg, ConvLayout::Nchw),
            Err(Error::InvalidArgument { arg: "out", .. })
        ),
        "resize_into an out that overlaps the input"
    );
}

/// A batch large enough to cross the CPU parallel threshold gives the same
/// bits in NCHW and NHWC, since every element runs the same arithmetic.
pub fn check_resize_layouts_agree<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let dims = [2, 5, 37, 53];
    let x = input(dims);
    for size in [(61, 29), (12, 100)] {
        for filter in [ResizeFilter::Bilinear, ResizeFilter::Bicubic] {
            for aa in [false, true] {
                let cfg = ResizeConfig::new(size, filter).with_antialias(aa);
                let label = format!("{dims:?} -> {size:?} {filter:?} antialias={aa}");
                let out_dims = [dims[0], dims[1], size.0, size.1];
                let mut results = Vec::new();
                for layout in LAYOUTS {
                    let y = client
                        .resize(&dense::<R>(&x, dims, layout, device), &cfg, layout)
                        .unwrap_or_else(|e| panic!("{label} {layout:?}: {e}"));
                    results.push(bits(&from_layout(&y.to_vec::<f32>(), out_dims, layout)));
                }
                assert_eq!(results[0], results[1], "{label}: NCHW vs NHWC");
            }
        }
    }
}
