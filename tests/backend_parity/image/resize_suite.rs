// Backend-generic ImageOps::resize checks against the PyTorch references in
// resize_golden.bin. resize_cpu.rs, resize_cuda.rs and resize_wgpu.rs run
// them on their own client.

use numr::dtype::DType;
use numr::ops::{ConvLayout, ImageOps, ResizeFilter};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::resize_golden::{
    Block, Kind, assert_close, blocks, from_layout, input, shape, to_layout, tolerance,
};

pub use super::suite::LAYOUTS;

/// Resize NCHW data `x` of extents `block.dims` in `layout` with F32 tensors
/// and return the NCHW result.
pub fn run_f32<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    block: &Block,
    x: &[f32],
    layout: ConvLayout,
) -> Vec<f32> {
    let label = format!("{} {layout:?} F32", block.label());
    let t = Tensor::<R>::from_slice(
        &to_layout(x, block.dims, layout),
        &shape(block.dims, layout),
        device,
    )
    .expect("F32 input tensor");
    let y = client
        .resize(&t, &block.config(), layout)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(
        y.shape(),
        &shape(block.out_dims(), layout)[..],
        "{label}: shape"
    );
    assert_eq!(y.dtype(), DType::F32, "{label}: dtype");
    from_layout(&y.to_vec::<f32>(), block.out_dims(), layout)
}

/// Every F32 reference in both layouts, within `slack` times the bound of
/// [`tolerance`].
pub fn check_resize_golden<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    slack: f32,
) {
    for block in blocks().iter().filter(|b| b.kind == Kind::F32) {
        let x = input(block.dims);
        for layout in LAYOUTS {
            let got = run_f32::<R, C>(client, device, block, &x, layout);
            let label = format!("{} {layout:?} F32 vs torch", block.label());
            assert_close(
                &got,
                &block.want,
                |i| slack * tolerance(block, &x, block.want[i]),
                &label,
            );
        }
    }
}

/// The PoC's 4x -> 2x SRVGG downsample: an exact halving of 64x64 with
/// bicubic antialias.
pub fn check_resize_srvgg<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    slack: f32,
) {
    let all = blocks();
    let block = all
        .iter()
        .find(|b| {
            b.kind == Kind::F32
                && b.dims == [1, 1, 64, 64]
                && b.size == (32, 32)
                && b.filter == ResizeFilter::Bicubic
                && b.antialias
        })
        .expect("resize_golden.bin holds the SRVGG block");
    let x = input(block.dims);
    for layout in LAYOUTS {
        let got = run_f32::<R, C>(client, device, block, &x, layout);
        let label = format!("SRVGG 64x64 -> 32x32 bicubic AA {layout:?}");
        assert_close(
            &got,
            &block.want,
            |i| slack * tolerance(block, &x, block.want[i]),
            &label,
        );
    }
}

/// F64 on the F32 references, within twice the F32 bound.
///
/// F64 sits closer to the exact value than the F32 reference. The reference
/// also rounds the antialias center by up to `u·in`, which moves its weights,
/// and the bound has no term for that. The factor 2 covers it.
pub fn check_resize_f64<R: Runtime<DType = DType>, C: ImageOps<R>>(client: &C, device: &R::Device) {
    for block in blocks().iter().filter(|b| b.kind == Kind::F32) {
        let x = input(block.dims);
        for layout in LAYOUTS {
            let label = format!("{} {layout:?} F64 vs torch F32", block.label());
            let x64: Vec<f64> = to_layout(&x, block.dims, layout)
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let t = Tensor::<R>::from_slice(&x64, &shape(block.dims, layout), device)
                .expect("F64 input tensor");
            let y = client
                .resize(&t, &block.config(), layout)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(y.dtype(), DType::F64, "{label}: dtype");
            let got: Vec<f32> = y.to_vec::<f64>().iter().map(|&v| v as f32).collect();
            let got = from_layout(&got, block.out_dims(), layout);
            assert_close(
                &got,
                &block.want,
                |i| 2.0 * tolerance(block, &x, block.want[i]),
                &label,
            );
        }
    }
}

/// F16 and BF16 references.
///
/// - Against torch: one storage ulp plus the F32 bound, as the reference is
///   F32 on the same rounded input.
/// - Against the backend's own F32 result on the rounded input: equal bits
///   after one rounding, as half types accumulate in F32 and round once.
#[cfg(feature = "f16")]
pub fn check_resize_half<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
) {
    use super::resize_golden::storage_ulp;
    use half::{bf16, f16};

    /// Rounded input, half result and rounding function, each as f32.
    type HalfRun = (Vec<f32>, Vec<f32>, fn(f32) -> f32);

    for block in blocks().iter().filter(|b| b.kind != Kind::F32) {
        let raw = input(block.dims);
        for layout in LAYOUTS {
            let label = format!("{} {layout:?}", block.label());
            let shape_in = shape(block.dims, layout);
            let laid = to_layout(&raw, block.dims, layout);
            // Rounded input as f32, the half result as f32, and the F32
            // result rounded to the half type, all in `layout` order.
            let (rounded, got, own): HalfRun = match block.kind {
                Kind::F16 => {
                    let h: Vec<f16> = laid.iter().map(|&v| f16::from_f32(v)).collect();
                    let t = Tensor::<R>::from_slice(&h, &shape_in, device).expect("F16 input");
                    let y = client
                        .resize(&t, &block.config(), layout)
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(y.dtype(), DType::F16, "{label}: dtype");
                    let r: Vec<f32> = h.iter().map(|v| v.to_f32()).collect();
                    let got: Vec<f32> = y.to_vec::<f16>().iter().map(|v| v.to_f32()).collect();
                    (r, got, |v: f32| f16::from_f32(v).to_f32())
                }
                _ => {
                    let h: Vec<bf16> = laid.iter().map(|&v| bf16::from_f32(v)).collect();
                    let t = Tensor::<R>::from_slice(&h, &shape_in, device).expect("BF16 input");
                    let y = client
                        .resize(&t, &block.config(), layout)
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(y.dtype(), DType::BF16, "{label}: dtype");
                    let r: Vec<f32> = h.iter().map(|v| v.to_f32()).collect();
                    let got: Vec<f32> = y.to_vec::<bf16>().iter().map(|v| v.to_f32()).collect();
                    (r, got, |v: f32| bf16::from_f32(v).to_f32())
                }
            };
            let rounded = from_layout(&rounded, block.dims, layout);
            let got = from_layout(&got, block.out_dims(), layout);
            let f32_result = run_f32::<R, C>(client, device, block, &rounded, layout);
            let once: Vec<f32> = f32_result.iter().map(|&v| own(v)).collect();
            for (i, (&g, &o)) in got.iter().zip(&once).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    o.to_bits(),
                    "{label}: element {i}: half result {g:e} is not the F32 result {o:e} \
                     rounded once"
                );
            }
            let (mantissa, floor) = if block.kind == Kind::F16 {
                (10, 2f32.powi(-24))
            } else {
                (7, 0.0)
            };
            assert_close(
                &got,
                &block.want,
                |i| {
                    let want = block.want[i];
                    tolerance(block, &rounded, want) + storage_ulp(want, mantissa).max(floor)
                },
                &format!("{label} vs torch F32 on the rounded input"),
            );
        }
    }
}
