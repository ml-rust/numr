//! Gather kernel behind `pixel_shuffle`, `pixel_unshuffle` and
//! `upsample_nearest2d`.
//!
//! The kernel walks the output in rows. An NCHW row is one `(n, c, h)` run
//! along W. An NHWC row is one `(n, h, w)` run along C. Every output element
//! reads one source element at the canonical coordinate the source map gives.
//! Both sides are addressed through canonical `[N, C, H, W]` strides, so
//! strided views on either side need no copy. Elements move as raw words of
//! the element size, so one kernel covers every dtype.

use crate::runtime::cpu::CpuClient;

/// Canonical source coordinate map of one op, resolved for the CPU.
#[derive(Debug, Clone, Copy)]
pub enum GatherMap<'a> {
    /// Pixel shuffle with factor `r`.
    Shuffle(usize),
    /// Pixel unshuffle with factor `r`.
    Unshuffle(usize),
    /// Nearest upsampling. `rows[h]` and `cols[w]` are the source H and W
    /// indices of output H index `h` and W index `w`.
    Nearest {
        /// Source H index per output H index.
        rows: &'a [usize],
        /// Source W index per output W index.
        cols: &'a [usize],
    },
}

/// Output extents and canonical strides of one gather.
#[derive(Debug, Clone, Copy)]
pub struct GatherGeometry {
    /// Output extents as `[N, C, H, W]`.
    pub out_dims: [usize; 4],
    /// Source strides in elements, as `[N, C, H, W]`.
    pub src_strides: [isize; 4],
    /// Destination strides in elements, as `[N, C, H, W]`.
    pub dst_strides: [isize; 4],
    /// Walk rows along C (NHWC order) instead of along W.
    pub channels_last: bool,
}

impl GatherGeometry {
    /// Row count and row length of the walk.
    fn rows(&self) -> (usize, usize) {
        let [n, c, h, w] = self.out_dims;
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

/// Copy one row, with `source` mapping canonical output `(c, h, w)` to
/// canonical source `(c, h, w)`.
///
/// # Safety
///
/// Every address `dot(strides, coords)` reaches from `src` and `dst` must lie
/// inside its allocation, and no two rows write the same element.
#[inline(always)]
unsafe fn copy_row<T: Copy, F: Fn(usize, usize, usize) -> [usize; 3]>(
    src: *const T,
    dst: *mut T,
    g: &GatherGeometry,
    row: usize,
    source: F,
) {
    let [_, c_dim, h_dim, w_dim] = g.out_dims;
    let emit = |n: usize, c: usize, h: usize, w: usize| {
        let [sc, sh, sw] = source(c, h, w);
        let s = dot(g.src_strides, [n, sc, sh, sw]);
        let d = dot(g.dst_strides, [n, c, h, w]);
        // SAFETY: forwarded from this function's contract.
        unsafe { *dst.offset(d) = *src.offset(s) };
    };
    if g.channels_last {
        let (w, h, n) = (row % w_dim, (row / w_dim) % h_dim, row / (w_dim * h_dim));
        for c in 0..c_dim {
            emit(n, c, h, w);
        }
    } else {
        let (h, c, n) = (row % h_dim, (row / h_dim) % c_dim, row / (h_dim * c_dim));
        for w in 0..w_dim {
            emit(n, c, h, w);
        }
    }
}

/// Copy one row under `map`.
///
/// # Safety
///
/// See [`copy_row`].
unsafe fn gather_row<T: Copy>(
    src: *const T,
    dst: *mut T,
    g: &GatherGeometry,
    map: GatherMap<'_>,
    row: usize,
) {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        match map {
            GatherMap::Shuffle(r) => copy_row(src, dst, g, row, |c, h, w| {
                [c * r * r + (h % r) * r + w % r, h / r, w / r]
            }),
            GatherMap::Unshuffle(r) => copy_row(src, dst, g, row, |c, h, w| {
                let rem = c % (r * r);
                [c / (r * r), h * r + rem / r, w * r + rem % r]
            }),
            GatherMap::Nearest { rows, cols } => {
                copy_row(src, dst, g, row, |c, h, w| [c, rows[h], cols[w]])
            }
        }
    }
}

/// Gather every output element with elements of type `T`, rows spread over
/// the client's pool.
///
/// # Safety
///
/// - `src` and `dst` point at the first element of their views, and every
///   address the strides reach lies inside the allocation.
/// - The destination view addresses no element twice and shares no element
///   with the source view.
/// - `T` has the element size and at most the alignment of the dtype.
pub unsafe fn image_gather_kernel<T: Copy>(
    client: &CpuClient,
    src: *const T,
    dst: *mut T,
    g: &GatherGeometry,
    map: GatherMap<'_>,
) {
    let (rows, run) = g.rows();
    if rows == 0 || run == 0 {
        return;
    }
    let src_addr = src as usize;
    let dst_addr = dst as usize;
    let geometry = *g;
    client.par_for_each(rows, rows * run, move |row| {
        // SAFETY: the caller's contract covers every row, and distinct rows
        // write distinct output elements.
        unsafe {
            gather_row(
                src_addr as *const T,
                dst_addr as *mut T,
                &geometry,
                map,
                row,
            )
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn contiguous(dims: [usize; 4]) -> [isize; 4] {
        let [_, c, h, w] = dims;
        [(c * h * w) as isize, (h * w) as isize, w as isize, 1]
    }

    #[test]
    fn shuffle_matches_the_index_formula() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let r = 2;
        let in_dims = [1, 8, 2, 3];
        let out_dims = [1, 2, 4, 6];
        let src: Vec<u32> = (0..48).collect();
        let mut dst = vec![0u32; 48];
        let g = GatherGeometry {
            out_dims,
            src_strides: contiguous(in_dims),
            dst_strides: contiguous(out_dims),
            channels_last: false,
        };
        unsafe {
            image_gather_kernel(
                &client,
                src.as_ptr(),
                dst.as_mut_ptr(),
                &g,
                GatherMap::Shuffle(r),
            );
        }
        for c in 0..2 {
            for oh in 0..4 {
                for ow in 0..6 {
                    let ic = c * 4 + (oh % 2) * 2 + ow % 2;
                    let want = (ic * 6 + (oh / 2) * 3 + ow / 2) as u32;
                    assert_eq!(dst[(c * 4 + oh) * 6 + ow], want);
                }
            }
        }
    }

    #[test]
    fn unshuffle_inverts_shuffle_in_nhwc_order() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        // NHWC [1, 2, 3, 8] shuffled by 2 into NHWC [1, 4, 6, 2].
        let src: Vec<u16> = (0..48).collect();
        let mut mid = vec![0u16; 48];
        let mut back = vec![0u16; 48];
        let nhwc = |n_h_w_c: [usize; 4]| {
            let [_, h, w, c] = n_h_w_c;
            [(h * w * c) as isize, 1, (w * c) as isize, c as isize]
        };
        let shuffle = GatherGeometry {
            out_dims: [1, 2, 4, 6],
            src_strides: nhwc([1, 2, 3, 8]),
            dst_strides: nhwc([1, 4, 6, 2]),
            channels_last: true,
        };
        let unshuffle = GatherGeometry {
            out_dims: [1, 8, 2, 3],
            src_strides: nhwc([1, 4, 6, 2]),
            dst_strides: nhwc([1, 2, 3, 8]),
            channels_last: true,
        };
        unsafe {
            image_gather_kernel(
                &client,
                src.as_ptr(),
                mid.as_mut_ptr(),
                &shuffle,
                GatherMap::Shuffle(2),
            );
            image_gather_kernel(
                &client,
                mid.as_ptr(),
                back.as_mut_ptr(),
                &unshuffle,
                GatherMap::Unshuffle(2),
            );
        }
        assert_eq!(back, src);
    }
}
