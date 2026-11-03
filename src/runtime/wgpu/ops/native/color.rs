//! WebGPU dispatch of the `ColorOps` kernels on packed U32 words and F32 RGB.
//!
//! Planes and RGB24 frames pass as [`PackedPlane`] words, and float RGB as
//! a strided F32 view. Each operand passes to its kernel as a base word and
//! strides in words, so no operand is copied and nothing leaves the device.

use super::helpers::*;
use super::image::word;
use crate::error::{Error, Result};
use crate::ops::common::{YuvFrame, rgb_strides};
use crate::ops::{ConvLayout, PackedPlane, YuvPlanes};
use crate::runtime::wgpu::shaders::color::{ColorEntry, launch_color};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Pack a stride in words as a two's complement word.
fn stride_word(stride: isize, what: &str, op: &'static str) -> Result<u32> {
    let stride = i32::try_from(stride).map_err(|_| {
        Error::backend_limitation(
            "WebGPU",
            op,
            format!("{what} stride {stride} does not fit the 32-bit addressing of the kernel"),
        )
    })?;
    // The kernel adds strides as wrapping u32, which equals i32 addition.
    Ok(stride as u32)
}

/// Base word and row stride of the words of `p`.
fn plane_words(p: &PackedPlane<'_, WgpuRuntime>, op: &'static str) -> Result<(u32, u32)> {
    Ok((
        word(p.words.offset(), "plane offset", op)?,
        stride_word(p.words.strides()[0], "plane row", op)?,
    ))
}

/// Base word and `[C, H, W]` strides of the F32 RGB view `rgb`.
fn rgb_words(
    rgb: &Tensor<WgpuRuntime>,
    layout: ConvLayout,
    op: &'static str,
) -> Result<(u32, [u32; 3])> {
    let [c, h, w] = rgb_strides(rgb.strides(), layout);
    Ok((
        word(rgb.offset(), "RGB offset", op)?,
        [
            stride_word(c, "RGB channel", op)?,
            stride_word(h, "RGB row", op)?,
            stride_word(w, "RGB column", op)?,
        ],
    ))
}

/// Return `Error::BackendLimitation` unless `planes` are packed.
pub(crate) fn require_packed(planes: &YuvPlanes<'_, WgpuRuntime>, op: &'static str) -> Result<()> {
    if planes.is_packed() {
        return Ok(());
    }
    let format = planes.format_name();
    let (sample, variant) = match planes.sample_bytes() {
        2 => ("U16", "P010Packed"),
        _ if format == "I420" => ("U8", "I420Packed"),
        _ => ("U8", "Nv12Packed"),
    };
    Err(Error::backend_limitation(
        "WebGPU",
        op,
        format!(
            "{format} planes are {sample} tensors, and WebGPU kernels address 32-bit words only; \
             pass YuvPlanes::{variant} with each plane as U32 words [rows, pitch / 4]"
        ),
    ))
}

/// The `Error::BackendLimitation` of an unpacked RGB24 method on WebGPU.
pub(crate) fn packed_rgb24_only(op: &'static str, packed: &str) -> Error {
    Error::backend_limitation(
        "WebGPU",
        op,
        format!(
            "U8 RGB24 tensors are not addressable by WebGPU kernels, which read 32-bit words \
             only; call {packed} with the frame as U32 words [H, ceil(3·W / 4)]"
        ),
    )
}

/// Run a YCbCr kernel between the checked packed `planes` of `frame` and the
/// checked F32 RGB view `rgb`.
pub(crate) fn native_yuv(
    client: &WgpuClient,
    entry: ColorEntry,
    frame: &YuvFrame,
    planes: &YuvPlanes<'_, WgpuRuntime>,
    rgb: &Tensor<WgpuRuntime>,
    layout: ConvLayout,
) -> Result<()> {
    let op = entry.name();
    let (format, y, u, v) = match *planes {
        YuvPlanes::Nv12Packed { y, uv } => (0u32, y, uv, uv),
        YuvPlanes::I420Packed { y, u, v } => (1, y, u, v),
        YuvPlanes::P010Packed { y, uv } => (2, y, uv, uv),
        _ => return require_packed(planes, op),
    };
    let (h, w) = (frame.height, frame.width);
    // Encode groups: luma columns per chroma word of the format.
    let span = [4, 8, 2][format as usize];
    let groups = w.div_ceil(span);
    let total = match entry {
        ColorEntry::RgbToYuv => (h / 2) * groups,
        _ => h * w,
    };
    if total == 0 {
        return Ok(());
    }
    let (y_base, y_rs) = plane_words(&y, op)?;
    let (u_base, u_rs) = plane_words(&u, op)?;
    let (v_base, v_rs) = plane_words(&v, op)?;
    let (rgb_base, rgb_strides) = rgb_words(rgb, layout, op)?;
    let k = &frame.coeffs;
    let params = YuvParams {
        total: word(total, "invocation count", op)?,
        height: word(h, "frame height", op)?,
        width: word(w, "frame width", op)?,
        groups: word(groups, "column groups", op)?,
        format,
        center: u32::from(frame.center),
        y_base,
        y_rs,
        u_base,
        u_rs,
        v_base,
        v_rs,
        rgb_base,
        rgb_strides,
        y_off: k.y_off,
        c_off: k.c_off,
        coeffs: [
            k.c_div16, k.cr_r, k.cb_b, k.g_cb, k.g_cr, k.kr, k.kg, k.kb, k.y_mul, k.c_mul, k.y_lo,
            k.y_hi, k.c_lo, k.c_hi,
        ]
        .map(f32::to_bits),
        offsets: [(k.y_off as f32).to_bits(), (k.c_off as f32).to_bits()],
        _pad: [0; 2],
    };
    let params_buf = create_params_buffer(client, &params);
    let storages = [rgb, y.words, u.words, v.words].map(|t| t.storage().ptr());
    launch_color(client, entry, &storages, &params_buf, total)
}

/// Run an RGB24 kernel between the checked packed words `packed` and the
/// checked F32 RGB view `rgb`. `fresh` marks new packed words, whose bytes
/// past each row become 0.
pub(crate) fn native_rgb24(
    client: &WgpuClient,
    entry: ColorEntry,
    packed: &PackedPlane<'_, WgpuRuntime>,
    rgb: &Tensor<WgpuRuntime>,
    layout: ConvLayout,
    [h, w]: [usize; 2],
    fresh: bool,
) -> Result<()> {
    let op = entry.name();
    let row_words = (3 * w).div_ceil(4);
    let total = match entry {
        ColorEntry::FloatToRgb24 => h * row_words,
        _ => h * w * 3,
    };
    if total == 0 {
        return Ok(());
    }
    let (p_base, p_rs) = plane_words(packed, op)?;
    let (f_base, f_strides) = rgb_words(rgb, layout, op)?;
    let params = Rgb24Params {
        total: word(total, "invocation count", op)?,
        width: word(w, "frame width", op)?,
        row_words: word(row_words, "words per row", op)?,
        fresh: u32::from(fresh),
        p_base,
        p_rs,
        f_base,
        f_strides,
        _pad: [0; 2],
    };
    let params_buf = create_params_buffer(client, &params);
    let (packed_ptr, rgb_ptr) = (packed.words.storage().ptr(), rgb.storage().ptr());
    let storages = match entry {
        ColorEntry::FloatToRgb24 => [rgb_ptr, packed_ptr],
        _ => [packed_ptr, rgb_ptr],
    };
    launch_color(client, entry, &storages, &params_buf, total)
}
