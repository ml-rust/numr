//! Separable pass kernel of `resize` on the CPU.
//!
//! One pass interpolates along one canonical axis (H or W) and keeps the
//! other three coordinates. The kernel walks the destination in rows: an
//! NCHW row is one `(n, c, h)` run along W, an NHWC row is one `(n, h, w)`
//! run along C. Both sides are addressed through canonical `[N, C, H, W]`
//! strides, so strided views need no copy.
//!
//! Tap sums use `mul_add`. On x86-64 with FMA the rows run in a
//! `#[target_feature(enable = "fma")]` function, so `mul_add` is one
//! instruction. Elsewhere `mul_add` is the correctly rounded library call,
//! so every path gives the same bits.

use super::image_resize_acc::ResizeAcc;
use super::image_resize_taps::{AxisTaps, sum_taps};
#[cfg(target_arch = "x86_64")]
use super::simd::detect_simd;
use crate::dtype::Element;
use crate::ops::common::ResizeMode;
use crate::runtime::cpu::CpuClient;

/// Extents and canonical strides of one pass.
#[derive(Debug, Clone, Copy)]
pub struct ResizePassGeometry {
    /// Destination extents as `[N, C, H, W]`.
    pub dst_dims: [usize; 4],
    /// Source strides in elements, as `[N, C, H, W]`.
    pub src_strides: [isize; 4],
    /// Destination strides in elements, as `[N, C, H, W]`.
    pub dst_strides: [isize; 4],
    /// Interpolated canonical axis: 2 for H, 3 for W.
    pub axis: usize,
    /// Walk rows along C (NHWC order) instead of along W.
    pub channels_last: bool,
}

impl ResizePassGeometry {
    /// Row count and row length of the walk.
    fn rows(&self) -> (usize, usize) {
        let [n, c, h, w] = self.dst_dims;
        if self.channels_last {
            (n * h * w, c)
        } else {
            (n * c * h, w)
        }
    }
}

#[inline(always)]
fn dot(strides: [isize; 4], coords: [usize; 4]) -> isize {
    coords[0] as isize * strides[0]
        + coords[1] as isize * strides[1]
        + coords[2] as isize * strides[2]
        + coords[3] as isize * strides[3]
}

/// Write one destination element.
///
/// # Safety
///
/// Every address the strides reach from `src` and `dst` lies inside its
/// allocation, and `taps` holds indices below the source extent of the axis.
#[inline(always)]
unsafe fn resize_element<I: Element, O: Element, A: ResizeAcc>(
    src: *const I,
    dst: *mut O,
    g: &ResizePassGeometry,
    mode: ResizeMode,
    taps: &AxisTaps<A>,
    coords: [usize; 4],
) {
    let o = coords[g.axis];
    let mut base = coords;
    base[g.axis] = 0;
    let base = dot(g.src_strides, base);
    let step = g.src_strides[g.axis];
    // SAFETY: forwarded from this function's contract.
    unsafe {
        let value = if mode == ResizeMode::Copy {
            A::load(*src.offset(base + o as isize * step))
        } else {
            let (index, weights) = taps.get(o);
            sum_taps(mode.is_antialias(), weights, |j| {
                A::load(*src.offset(base + index[j] as isize * step))
            })
        };
        *dst.offset(dot(g.dst_strides, coords)) = value.store::<O>();
    }
}

/// Write one destination row.
///
/// # Safety
///
/// See [`resize_element`]. No two rows write the same element.
#[inline(always)]
unsafe fn resize_row<I: Element, O: Element, A: ResizeAcc>(
    src: *const I,
    dst: *mut O,
    g: &ResizePassGeometry,
    mode: ResizeMode,
    taps: &AxisTaps<A>,
    row: usize,
) {
    let [_, c_dim, h_dim, w_dim] = g.dst_dims;
    // SAFETY: forwarded from this function's contract.
    unsafe {
        if g.channels_last {
            let (w, h, n) = (row % w_dim, (row / w_dim) % h_dim, row / (w_dim * h_dim));
            for c in 0..c_dim {
                resize_element(src, dst, g, mode, taps, [n, c, h, w]);
            }
        } else {
            let (h, c, n) = (row % h_dim, (row / h_dim) % c_dim, row / (h_dim * c_dim));
            for w in 0..w_dim {
                resize_element(src, dst, g, mode, taps, [n, c, h, w]);
            }
        }
    }
}

/// [`resize_row`] compiled with FMA, so `mul_add` is one instruction.
///
/// # Safety
///
/// See [`resize_row`]. The CPU supports FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn resize_row_fma<I: Element, O: Element, A: ResizeAcc>(
    src: *const I,
    dst: *mut O,
    g: &ResizePassGeometry,
    mode: ResizeMode,
    taps: &AxisTaps<A>,
    row: usize,
) {
    // SAFETY: forwarded from this function's contract.
    unsafe { resize_row(src, dst, g, mode, taps, row) }
}

/// Run one pass, rows spread over the client's pool.
///
/// `taps` comes from [`AxisTaps::build`] for `mode` along `g.axis`.
///
/// # Safety
///
/// - `src` and `dst` point at the first element of their views, and every
///   address the strides reach lies inside the allocation.
/// - The destination view addresses no element twice and shares no element
///   with the source view.
pub unsafe fn resize_pass_kernel<I: Element, O: Element, A: ResizeAcc>(
    client: &CpuClient,
    src: *const I,
    dst: *mut O,
    g: &ResizePassGeometry,
    mode: ResizeMode,
    taps: &AxisTaps<A>,
) {
    let (rows, run) = g.rows();
    if rows == 0 || run == 0 {
        return;
    }
    let src_addr = src as usize;
    let dst_addr = dst as usize;
    let geometry = *g;
    let taps_per_output = match mode {
        ResizeMode::Copy => 1,
        _ => taps.get(0).1.len().max(1),
    };
    #[cfg(target_arch = "x86_64")]
    let fma = detect_simd().has_avx2();
    client.par_for_each(rows, rows * run * taps_per_output, move |row| {
        let src = src_addr as *const I;
        let dst = dst_addr as *mut O;
        // SAFETY: the caller's contract covers every row, distinct rows
        // write distinct elements, and `has_avx2` levels include FMA.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if fma {
                resize_row_fma(src, dst, &geometry, mode, taps, row);
                return;
            }
            resize_row(src, dst, &geometry, mode, taps, row);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn w_pass_on_nchw_matches_the_taps() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let src: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let mut dst = vec![0.0f32; 4];
        let taps = AxisTaps::<f32>::build(ResizeMode::Linear, 4, 2);
        let g = ResizePassGeometry {
            dst_dims: [1, 1, 2, 2],
            src_strides: [8, 8, 4, 1],
            dst_strides: [4, 4, 2, 1],
            axis: 3,
            channels_last: false,
        };
        unsafe {
            resize_pass_kernel(
                &client,
                src.as_ptr(),
                dst.as_mut_ptr(),
                &g,
                ResizeMode::Linear,
                &taps,
            );
        }
        assert_eq!(dst, vec![0.5, 2.5, 4.5, 6.5]);
    }

    #[test]
    fn copy_pass_moves_elements_through_strides() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        // NHWC [1, 1, 2, 3] read as canonical [N, C, H, W] = [1, 3, 1, 2].
        let src: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let mut dst = vec![0.0f32; 6];
        let taps = AxisTaps::<f32>::build(ResizeMode::Copy, 2, 2);
        let g = ResizePassGeometry {
            dst_dims: [1, 3, 1, 2],
            src_strides: [6, 1, 6, 3],
            dst_strides: [6, 2, 2, 1],
            axis: 3,
            channels_last: false,
        };
        unsafe {
            resize_pass_kernel(
                &client,
                src.as_ptr(),
                dst.as_mut_ptr(),
                &g,
                ResizeMode::Copy,
                &taps,
            );
        }
        assert_eq!(dst, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
