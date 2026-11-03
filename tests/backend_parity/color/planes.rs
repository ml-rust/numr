// Host helpers for ColorOps tests: plane and RGB tensors in and out of a
// backend, contiguous or as views into one pitched surface, as samples or as
// packed words (see packed.rs).

use numr::dtype::DType;
use numr::ops::{ConvLayout, PackedPlane, YuvPlanes};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::golden::Format;
use super::packed::{packed_form, unpack_rows, upload_packed};

/// Sample every byte of a surface starts as, outside the plane views.
pub const SENTINEL: u16 = 0xA5;

/// Columns a pitched surface adds past the widest plane.
pub const PITCH_PAD: usize = 6;

/// Plane tensors of one frame.
pub struct PlaneSet<R: Runtime> {
    pub format: Format,
    /// The pitched surface every plane views, or `None` for separate planes.
    pub surface: Option<Tensor<R>>,
    pub planes: Vec<Tensor<R>>,
    /// Columns per row of each plane when `planes` are packed U32 words.
    pub packed_widths: Option<Vec<usize>>,
}

impl<R: Runtime<DType = DType>> PlaneSet<R> {
    pub fn yuv(&self) -> YuvPlanes<'_, R> {
        let p = &self.planes;
        if let Some(w) = &self.packed_widths {
            let at = |i: usize| PackedPlane::new(&p[i], w[i]);
            return match self.format {
                Format::Nv12 => YuvPlanes::Nv12Packed {
                    y: at(0),
                    uv: at(1),
                },
                Format::I420 => YuvPlanes::I420Packed {
                    y: at(0),
                    u: at(1),
                    v: at(2),
                },
                Format::P010 => YuvPlanes::P010Packed {
                    y: at(0),
                    uv: at(1),
                },
            };
        }
        match self.format {
            Format::Nv12 => YuvPlanes::Nv12 {
                y: &p[0],
                uv: &p[1],
            },
            Format::I420 => YuvPlanes::I420 {
                y: &p[0],
                u: &p[1],
                v: &p[2],
            },
            Format::P010 => YuvPlanes::P010 {
                y: &p[0],
                uv: &p[1],
            },
        }
    }

    /// Samples of each plane in row-major order.
    pub fn download(&self) -> Vec<Vec<u16>> {
        let dtype = self.format.dtype();
        let Some(widths) = &self.packed_widths else {
            return self
                .planes
                .iter()
                .map(|t| samples(&t.to_bytes().expect("plane readback"), t.dtype()))
                .collect();
        };
        self.planes
            .iter()
            .zip(widths)
            .map(|(t, &cols)| {
                let row_bytes = cols * dtype.size_in_bytes();
                let bytes = t.to_bytes().expect("plane readback");
                samples(&unpack_rows(&bytes, t.shape(), row_bytes), dtype)
            })
            .collect()
    }

    /// Every sample of the surface outside the plane views keeps
    /// [`SENTINEL`].
    pub fn assert_padding_kept(&self, h: usize, w: usize, label: &str) {
        if self.packed_widths.is_some() {
            super::packed::assert_packed_padding_kept(self, label);
            return;
        }
        let Some(surface) = &self.surface else {
            return;
        };
        let all = samples(
            &surface.to_bytes().expect("surface readback"),
            surface.dtype(),
        );
        let pitch = surface.shape()[1];
        let mut row0 = 0;
        for [rows, cols] in self.format.plane_shapes(h, w) {
            for r in 0..rows {
                for c in cols..pitch {
                    let at = (row0 + r) * pitch + c;
                    assert_eq!(
                        all[at],
                        SENTINEL,
                        "{label}: padding at surface [{}, {c}]",
                        row0 + r
                    );
                }
            }
            row0 += rows;
        }
    }
}

/// Little-endian bytes of samples stored as `dtype` (U8 or U16).
pub fn sample_bytes(values: &[u16], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::U16 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => values.iter().map(|&v| v as u8).collect(),
    }
}

/// Samples of `dtype` (U8 or U16) from little-endian bytes.
pub fn samples(bytes: &[u8], dtype: DType) -> Vec<u16> {
    match dtype {
        DType::U16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect(),
        _ => bytes.iter().map(|&b| u16::from(b)).collect(),
    }
}

/// Upload `data` (one sample vector per plane) for an `h x w` frame.
///
/// With `pitched`, the planes stack top to bottom in one surface whose row
/// pitch exceeds the widest plane by [`PITCH_PAD`], and every other sample
/// holds [`SENTINEL`]. Inside `packed::with_packed` the planes upload as
/// packed U32 words instead.
pub fn upload<R: Runtime<DType = DType>>(
    format: Format,
    h: usize,
    w: usize,
    data: &[Vec<u16>],
    pitched: bool,
    device: &R::Device,
) -> PlaneSet<R> {
    if packed_form() {
        return upload_packed(format, h, w, data, pitched, device);
    }
    let dtype = format.dtype();
    let shapes = format.plane_shapes(h, w);
    if !pitched {
        let planes = shapes
            .iter()
            .zip(data)
            .map(|(shape, values)| {
                Tensor::<R>::from_bytes(&sample_bytes(values, dtype), shape, dtype, device)
                    .expect("plane upload")
            })
            .collect();
        return PlaneSet {
            format,
            surface: None,
            planes,
            packed_widths: None,
        };
    }
    let pitch = w + PITCH_PAD;
    let rows: usize = shapes.iter().map(|s| s[0]).sum();
    let mut all = vec![SENTINEL; rows * pitch];
    let mut row0 = 0;
    for ([r, c], values) in shapes.iter().zip(data) {
        for i in 0..*r {
            let dst = (row0 + i) * pitch;
            all[dst..dst + c].copy_from_slice(&values[i * c..(i + 1) * c]);
        }
        row0 += r;
    }
    let surface =
        Tensor::<R>::from_bytes(&sample_bytes(&all, dtype), &[rows, pitch], dtype, device)
            .expect("surface upload");
    let mut row0 = 0;
    let planes = shapes
        .iter()
        .map(|&[r, c]| {
            let view = surface
                .narrow(0, row0, r)
                .and_then(|t| t.narrow(1, 0, c))
                .expect("plane view");
            row0 += r;
            view
        })
        .collect();
    PlaneSet {
        format,
        surface: Some(surface),
        planes,
        packed_widths: None,
    }
}

/// Planes of an `h x w` frame filled with zeros.
pub fn blank<R: Runtime<DType = DType>>(
    format: Format,
    h: usize,
    w: usize,
    pitched: bool,
    device: &R::Device,
) -> PlaneSet<R> {
    let data: Vec<Vec<u16>> = format
        .plane_shapes(h, w)
        .iter()
        .map(|[r, c]| vec![0; r * c])
        .collect();
    upload(format, h, w, &data, pitched, device)
}

/// `[3, H, W]` values reordered into `layout` order.
pub fn to_layout(chw: &[f32], h: usize, w: usize, layout: ConvLayout) -> Vec<f32> {
    match layout {
        ConvLayout::Nchw => chw.to_vec(),
        ConvLayout::Nhwc => (0..h * w)
            .flat_map(|p| (0..3).map(move |c| chw[c * h * w + p]))
            .collect(),
    }
}

/// `layout`-order values reordered into `[3, H, W]`.
pub fn from_layout(vals: &[f32], h: usize, w: usize, layout: ConvLayout) -> Vec<f32> {
    match layout {
        ConvLayout::Nchw => vals.to_vec(),
        ConvLayout::Nhwc => (0..3)
            .flat_map(|c| (0..h * w).map(move |p| vals[p * 3 + c]))
            .collect(),
    }
}

/// RGB shape of an `h x w` frame in `layout`.
pub fn rgb_shape(h: usize, w: usize, layout: ConvLayout) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => [1, 3, h, w],
        ConvLayout::Nhwc => [1, h, w, 3],
    }
}

/// Upload `[3, H, W]` F32 values as an RGB tensor in `layout`.
pub fn rgb_tensor<R: Runtime<DType = DType>>(
    chw: &[f32],
    h: usize,
    w: usize,
    layout: ConvLayout,
    device: &R::Device,
) -> Tensor<R> {
    let vals = to_layout(chw, h, w, layout);
    Tensor::<R>::from_slice(&vals, &rgb_shape(h, w, layout), device).expect("rgb upload")
}

/// Read an F32 RGB tensor in `layout` back as `[3, H, W]` values.
pub fn rgb_values<R: Runtime<DType = DType>>(
    t: &Tensor<R>,
    h: usize,
    w: usize,
    layout: ConvLayout,
) -> Vec<f32> {
    assert_eq!(t.dtype(), DType::F32, "rgb_values reads F32");
    let bytes = t.to_bytes().expect("rgb readback");
    let vals: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    from_layout(&vals, h, w, layout)
}

/// Assert `|got - want| <= tol` elementwise.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{label}: element {i}: got {g:e}, want {w:e}, tolerance {tol:e}"
        );
    }
}
