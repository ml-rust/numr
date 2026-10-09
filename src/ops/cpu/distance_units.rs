//! Parallel units of work for CPU cdist and pdist.
//!
//! The units depend on the shape alone, never on the thread count or chunk
//! size: a fixed-width block of columns of one cdist row, or one pdist row.
//! Each output element comes from one distance call on fixed inputs, so every
//! schedule of the units writes the same bits.

use crate::dtype::Element;
use crate::ops::DistanceMetric;
use crate::runtime::cpu::{CpuClient, kernels};
use num_traits::{Float, FromPrimitive};

/// Target scalar operations per cdist unit.
const CDIST_UNIT_WORK: usize = 32 * 1024;

/// Columns per cdist unit. It depends on `d` alone, never on the thread count.
#[inline]
fn cdist_col_block(d: usize) -> usize {
    (CDIST_UNIT_WORK / d.max(1)).max(1)
}

/// cdist over contiguous `x [n, d]` and `y [m, d]` into `out [n, m]`.
///
/// Unit `u` covers row `u / blocks_per_row` and column block
/// `u % blocks_per_row`.
///
/// # Safety
/// `x`, `y` and `out` must point to contiguous buffers of `n * d`, `m * d` and
/// `n * m` elements. `out` must alias neither input.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn cdist_units<T: Element + Float + FromPrimitive>(
    client: &CpuClient,
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
    metric: DistanceMetric,
) {
    let block = cdist_col_block(d);
    let blocks_per_row = m.div_ceil(block);
    let count = n * blocks_per_row;
    let work = n.saturating_mul(m).saturating_mul(d);
    // Addresses, not pointers: the closure must be `Sync`. Units write
    // disjoint ranges of `out`.
    let (x_addr, y_addr, out_addr) = (x as usize, y as usize, out as usize);
    client.par_for_each(count, work, |u| {
        let row = u / blocks_per_row;
        let col_start = (u % blocks_per_row) * block;
        let col_end = (col_start + block).min(m);
        // SAFETY: the addresses come from the caller's valid pointers. Unit `u`
        // writes only `out[row * m + col_start..row * m + col_end]`, and no
        // other unit covers that range.
        unsafe {
            kernels::cdist_block_kernel::<T>(
                x_addr as *const T,
                y_addr as *const T,
                out_addr as *mut T,
                row,
                col_start,
                col_end,
                m,
                d,
                metric,
            );
        }
    });
}

/// pdist over contiguous `x [n, d]` into the condensed `out [n * (n - 1) / 2]`.
///
/// Unit `u` covers row `u`.
///
/// # Safety
/// `x` and `out` must point to contiguous buffers of `n * d` and
/// `n * (n - 1) / 2` elements. `out` must not alias `x`.
pub(super) unsafe fn pdist_units<T: Element + Float + FromPrimitive>(
    client: &CpuClient,
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
    metric: DistanceMetric,
) {
    let pairs = n * n.saturating_sub(1) / 2;
    let work = pairs.saturating_mul(d);
    let (x_addr, out_addr) = (x as usize, out as usize);
    // Addresses, not pointers: the closure must be `Sync`. Unit `row` writes
    // only the pairs of `row`, so units never overlap.
    // SAFETY: the addresses come from the caller's valid pointers.
    client.par_for_each(n, work, |row| unsafe {
        kernels::pdist_row_kernel::<T>(x_addr as *const T, out_addr as *mut T, n, d, row, metric);
    });
}
