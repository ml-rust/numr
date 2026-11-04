//! CPU dtype dispatch of the packed RGB24 `ColorOps` methods.
//!
//! Packed words are the bytes of the frame, so the RGB24 kernels read and
//! write them in place through byte strides.

use super::super::kernels::{Rgb24Geometry, float_to_rgb24_kernel, rgb24_to_float_kernel};
use super::super::{CpuClient, CpuRuntime};
use super::color::with_float;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::common::color_packed::packed_view;
use crate::ops::common::{PackedLumaPad, packed_rgb24_strides, rgb_strides};
use crate::ops::{ConvLayout, PackedPlane};
use crate::tensor::Tensor;

/// Unpack the checked packed RGB24 words `rgb24` into the float RGB view
/// `out`.
pub fn packed_rgb24_to_float_run(
    client: &CpuClient,
    rgb24: &PackedPlane<'_, CpuRuntime>,
    out: &Tensor<CpuRuntime>,
    layout: ConvLayout,
    size: [usize; 2],
) -> Result<()> {
    let op = "packed_rgb24_to_float";
    let g = Rgb24Geometry {
        size,
        src_strides: packed_rgb24_strides(rgb24, "rgb24", op)?,
        dst_strides: rgb_strides(out.strides(), layout),
    };
    let src = rgb24.words.ptr() as *const u8;
    with_float!(out.dtype(), op, T => {
        // SAFETY: the words hold `3·W` bytes per row from the first word of
        // each row, and `out` has the checked shape and passed the repeat
        // and overlap checks.
        unsafe { rgb24_to_float_kernel::<T>(client, src, out.ptr() as *mut T, g) }
    });
    Ok(())
}

/// Pack the float RGB view `rgb` into the checked packed RGB24 words `out`.
pub fn float_to_packed_rgb24_run(
    client: &CpuClient,
    rgb: &Tensor<CpuRuntime>,
    out: &PackedPlane<'_, CpuRuntime>,
    layout: ConvLayout,
    size: [usize; 2],
) -> Result<()> {
    let op = "float_to_packed_rgb24";
    let g = Rgb24Geometry {
        size,
        src_strides: rgb_strides(rgb.strides(), layout),
        dst_strides: packed_rgb24_strides(out, "out", op)?,
    };
    let dst = out.words.ptr() as *mut u8;
    with_float!(rgb.dtype(), op, T => {
        // SAFETY: the words hold `3·W` bytes per row from the first word of
        // each row, address each word once and share no byte with `rgb`.
        unsafe { float_to_rgb24_kernel::<T>(client, rgb.ptr() as *const T, dst, g) }
    });
    Ok(())
}

/// Replicate-pad the checked packed luma `y` into the zeroed words `out`,
/// which have the shape of `plan`.
pub fn pad_packed_luma_run(
    y: &PackedPlane<'_, CpuRuntime>,
    plan: &PackedLumaPad,
    out: &Tensor<CpuRuntime>,
) -> Result<()> {
    let view = packed_view(y, plan.sample_bytes, "pad_packed_luma")?;
    let [h, w] = plan.source;
    let [rows, columns] = plan.target;
    let sample = plan.sample_bytes;
    let src_base = view.addr as *const u8;
    let src_pitch = view.row_stride.wrapping_mul(sample as isize);
    let dst_base = out.ptr() as *mut u8;
    let dst_pitch = plan.row_words * 4;
    for r in 0..rows {
        let src_row = (r.min(h - 1) as isize).wrapping_mul(src_pitch);
        // SAFETY: the check found `W` samples in each of the `H` rows of `y`,
        // and `r.min(h - 1)` is one of those rows.
        let src =
            unsafe { std::slice::from_raw_parts(src_base.wrapping_offset(src_row), w * sample) };
        // SAFETY: `out` has `rows` rows of `dst_pitch` bytes, which hold
        // `columns` samples, and is a new tensor that shares no byte with `y`.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(dst_base.add(r * dst_pitch), columns * sample)
        };
        let (body, tail) = dst.split_at_mut(src.len());
        body.copy_from_slice(src);
        let last = &src[src.len() - sample..];
        for slot in tail.chunks_exact_mut(sample) {
            slot.copy_from_slice(last);
        }
    }
    Ok(())
}
