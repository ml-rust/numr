//! CPU kernels of the `ColorOps` RGB24 pack and unpack.
//!
//! Both sides are addressed through `[C, H, W]` strides, one row per task.

use super::color::clamp01;
use crate::dtype::Element;
use crate::runtime::cpu::CpuClient;

/// `[C, H, W]` strides and `[H, W]` extents of one conversion.
#[derive(Debug, Clone, Copy)]
pub struct Rgb24Geometry {
    /// Frame size `[H, W]`.
    pub size: [usize; 2],
    /// Source strides as `[C, H, W]`.
    pub src_strides: [isize; 3],
    /// Destination strides as `[C, H, W]`.
    pub dst_strides: [isize; 3],
}

/// Run `f(src_offset, dst_offset)` for each of the 3 channels of every
/// pixel, one row per task.
fn for_each_element<F: Fn(isize, isize) + Send + Sync>(client: &CpuClient, g: Rgb24Geometry, f: F) {
    let [h, w] = g.size;
    if h == 0 || w == 0 {
        return;
    }
    client.par_for_each(h, h * w * 3, move |row| {
        for col in 0..w {
            for c in 0..3isize {
                let src = c * g.src_strides[0]
                    + row as isize * g.src_strides[1]
                    + col as isize * g.src_strides[2];
                let dst = c * g.dst_strides[0]
                    + row as isize * g.dst_strides[1]
                    + col as isize * g.dst_strides[2];
                f(src, dst);
            }
        }
    });
}

/// Write `f32(v) / 255` for every U8 sample.
///
/// # Safety
///
/// `src` and `dst` point at element `[0, 0, 0]` of their views, every
/// address the strides reach lies inside its allocation, the destination
/// reaches no element twice, and the views share no byte.
pub unsafe fn rgb24_to_float_kernel<O: Element>(
    client: &CpuClient,
    src: *const u8,
    dst: *mut O,
    g: Rgb24Geometry,
) {
    let (src, dst) = (src as usize, dst as usize);
    for_each_element(client, g, move |s, d| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            let v = f32::from(*(src as *const u8).offset(s)) / 255.0;
            *(dst as *mut O).offset(d) = O::from_f32(v);
        }
    });
}

/// Write `round_ties_even(clamp(x, 0, 1)·255)` for every float sample.
///
/// # Safety
///
/// As [`rgb24_to_float_kernel`].
pub unsafe fn float_to_rgb24_kernel<I: Element>(
    client: &CpuClient,
    src: *const I,
    dst: *mut u8,
    g: Rgb24Geometry,
) {
    let (src, dst) = (src as usize, dst as usize);
    for_each_element(client, g, move |s, d| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            let x = clamp01((*(src as *const I).offset(s)).to_f32());
            *(dst as *mut u8).offset(d) = (x * 255.0).round_ties_even() as u8;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn every_byte_round_trips() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let src: Vec<u8> = (0..=255).flat_map(|v| [v, v, v]).collect();
        let mut mid = vec![0.0f32; src.len()];
        let mut back = vec![0u8; src.len()];
        let g = Rgb24Geometry {
            size: [1, 256],
            src_strides: [1, 768, 3],
            dst_strides: [1, 768, 3],
        };
        unsafe {
            rgb24_to_float_kernel(&client, src.as_ptr(), mid.as_mut_ptr(), g);
            float_to_rgb24_kernel(&client, mid.as_ptr(), back.as_mut_ptr(), g);
        }
        assert_eq!(back, src);
        assert_eq!(mid[3 * 255], 1.0);
    }
}
