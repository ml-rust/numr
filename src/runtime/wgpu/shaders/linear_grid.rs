//! Element counts and 2-D dispatch grids for one-thread-per-item kernels.
//!
//! A 1-D dispatch holds at most 65535 workgroups, which caps a 256-thread
//! kernel near 16.7M items. [`linear_grid`] spreads the workgroups over x and y.
//! A shader on this grid computes its item index as
//!
//! ```wgsl
//! fn kernel(@builtin(global_invocation_id) gid: vec3<u32>,
//!           @builtin(num_workgroups) nwg: vec3<u32>) {
//!     let idx = gid.x + gid.y * nwg.x * 256u;
//!     if (idx >= params.numel) { return; }
//! }
//! ```
//!
//! For any item count up to `u32::MAX`, every thread index of the grid fits in
//! a `u32`, so that sum never wraps. [`element_count_u32`] rejects larger counts.

use crate::error::{Error, Result};

use super::pipeline::WORKGROUP_SIZE;

/// The WebGPU minimum for `maxComputeWorkgroupsPerDimension`.
const MAX_GROUPS_PER_DIM: usize = 65535;

/// Workgroups along x once a grid needs a second row. A power of two divides
/// `2^32 / WORKGROUP_SIZE`, so the padded grid never passes `2^32` threads.
const WIDE_GRID_X: usize = 32768;

/// An item count as a shader's `u32` count.
///
/// Returns `InvalidArgument` naming `op` when `count` exceeds `u32::MAX`.
pub fn element_count_u32(count: usize, op: &'static str) -> Result<u32> {
    u32::try_from(count).map_err(|_| Error::InvalidArgument {
        arg: "shape",
        reason: format!("{op}: {count} elements exceed the u32 count a WebGPU shader indexes"),
    })
}

/// The `(x, y)` workgroup grid for `threads` threads, one per item.
///
/// Up to 65535 workgroups the grid is `(groups, 1)`. Past that it is
/// `(32768, ceil(groups / 32768))`. For `threads <= u32::MAX` the grid holds at
/// most `2^32` threads, so `gid.x + gid.y * nwg.x * WORKGROUP_SIZE` fits in a `u32`.
pub fn linear_grid(threads: usize) -> (u32, u32) {
    let groups = threads.div_ceil(WORKGROUP_SIZE as usize).max(1);
    if groups <= MAX_GROUPS_PER_DIM {
        return (groups as u32, 1);
    }
    let rows = groups.div_ceil(WIDE_GRID_X);
    (WIDE_GRID_X as u32, rows as u32)
}

/// The `(x, y)` workgroup grid for `groups` workgroups, one per row.
///
/// Up to 65535 workgroups the grid is `(groups, 1)`. Past that it is
/// `(65535, ceil(groups / 65535))`. A shader recovers its row as
/// `wid.x + wid.y * nwg.x` and returns early past the last one.
///
/// Returns `InvalidArgument` naming `op` past `65535 * 65535` rows.
pub fn group_grid(groups: usize, op: &'static str) -> Result<(u32, u32)> {
    let groups = groups.max(1);
    if groups <= MAX_GROUPS_PER_DIM {
        return Ok((groups as u32, 1));
    }
    let rows = groups.div_ceil(MAX_GROUPS_PER_DIM);
    if rows > MAX_GROUPS_PER_DIM {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: format!(
                "{op}: {groups} rows exceed the {} workgroups a WebGPU grid holds",
                MAX_GROUPS_PER_DIM * MAX_GROUPS_PER_DIM
            ),
        });
    }
    Ok((MAX_GROUPS_PER_DIM as u32, rows as u32))
}

/// The `(x, y, z)` workgroup grid for `batch` rows of `threads` threads each.
///
/// x covers one row with `ceil(threads / WORKGROUP_SIZE)` workgroups. The rows
/// spread over y and z: `y = min(batch, 65535)` and `z = ceil(batch / y)`. A
/// shader recovers its row as `gid.y + gid.z * nwg.y` and returns early past
/// the last one.
///
/// Returns `InvalidArgument` naming `op` when one row needs more than 65535
/// workgroups, or `batch` needs more than `65535 * 65535` rows.
pub fn row_batch_grid(threads: usize, batch: usize, op: &'static str) -> Result<(u32, u32, u32)> {
    let x = threads.div_ceil(WORKGROUP_SIZE as usize).max(1);
    if x > MAX_GROUPS_PER_DIM {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: format!(
                "{op}: a row of {threads} elements exceeds the {} threads a WebGPU grid row holds",
                MAX_GROUPS_PER_DIM * WORKGROUP_SIZE as usize
            ),
        });
    }
    let batch = batch.max(1);
    let y = batch.min(MAX_GROUPS_PER_DIM);
    let z = batch.div_ceil(y);
    if z > MAX_GROUPS_PER_DIM {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: format!(
                "{op}: {batch} rows exceed the {} rows a WebGPU grid holds",
                MAX_GROUPS_PER_DIM * MAX_GROUPS_PER_DIM
            ),
        });
    }
    Ok((x as u32, y as u32, z as u32))
}

/// Dispatch `pass` over [`row_batch_grid`].
pub fn dispatch_row_batch(
    pass: &mut wgpu::ComputePass<'_>,
    threads: usize,
    batch: usize,
    op: &'static str,
) -> Result<()> {
    let (x, y, z) = row_batch_grid(threads, batch, op)?;
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

/// Dispatch `pass` over [`linear_grid`] for `threads` threads.
pub fn dispatch_linear(pass: &mut wgpu::ComputePass<'_>, threads: usize) {
    let (x, y) = linear_grid(threads);
    pass.dispatch_workgroups(x, y, 1);
}

/// Dispatch `pass` with one workgroup per row, over [`group_grid`].
pub fn dispatch_rows(
    pass: &mut wgpu::ComputePass<'_>,
    rows: usize,
    op: &'static str,
) -> Result<()> {
    let (x, y) = group_grid(rows, op)?;
    pass.dispatch_workgroups(x, y, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_threads(grid: (u32, u32)) -> u64 {
        grid.0 as u64 * grid.1 as u64 * WORKGROUP_SIZE as u64
    }

    #[test]
    fn element_count_u32_accepts_counts_up_to_u32_max() {
        assert_eq!(element_count_u32(0, "rand").unwrap(), 0);
        assert_eq!(
            element_count_u32(u32::MAX as usize, "rand").unwrap(),
            u32::MAX
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn element_count_u32_rejects_counts_past_u32_max() {
        let count = u32::MAX as usize + 1;
        match element_count_u32(count, "bernoulli") {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "shape");
                assert!(reason.starts_with("bernoulli: "), "{reason}");
                assert!(reason.contains(&count.to_string()), "{reason}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn small_counts_use_one_row() {
        assert_eq!(linear_grid(0), (1, 1));
        assert_eq!(linear_grid(1), (1, 1));
        assert_eq!(linear_grid(256), (1, 1));
        assert_eq!(linear_grid(257), (2, 1));
        assert_eq!(linear_grid(65535 * 256), (65535, 1));
    }

    #[test]
    fn large_counts_spill_into_rows() {
        let threads = 65535 * 256 + 1;
        let grid = linear_grid(threads);
        assert_eq!(grid, (32768, 2));
        assert!(total_threads(grid) >= threads as u64);
    }

    #[test]
    fn group_grid_covers_every_row() {
        assert_eq!(group_grid(0, "sort").unwrap(), (1, 1));
        assert_eq!(group_grid(65535, "sort").unwrap(), (65535, 1));
        assert_eq!(group_grid(65536, "sort").unwrap(), (65535, 2));
        let max_rows = MAX_GROUPS_PER_DIM * MAX_GROUPS_PER_DIM;
        for rows in [1usize, 70_000, 1 << 24, max_rows] {
            let (x, y) = group_grid(rows, "sort").unwrap();
            assert!(x as usize <= MAX_GROUPS_PER_DIM && y as usize <= MAX_GROUPS_PER_DIM);
            assert!(x as u64 * y as u64 >= rows as u64, "{rows}: ({x}, {y})");
        }
    }

    #[test]
    fn group_grid_rejects_rows_past_the_grid() {
        let rows = MAX_GROUPS_PER_DIM * MAX_GROUPS_PER_DIM + 1;
        match group_grid(rows, "topk") {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "shape");
                assert!(reason.starts_with("topk: "), "{reason}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn row_batch_grid_spreads_rows_over_y_and_z() {
        assert_eq!(row_batch_grid(1, 1, "fft").unwrap(), (1, 1, 1));
        assert_eq!(row_batch_grid(512, 3, "fft").unwrap(), (2, 3, 1));
        assert_eq!(row_batch_grid(256, 65535, "fft").unwrap(), (1, 65535, 1));
        assert_eq!(row_batch_grid(256, 65536, "fft").unwrap(), (1, 65535, 2));
        let (_, y, z) = row_batch_grid(256, 200_000, "fft").unwrap();
        assert!(y as u64 * z as u64 >= 200_000);
    }

    #[test]
    fn row_batch_grid_rejects_oversized_rows_and_batches() {
        let long_row = MAX_GROUPS_PER_DIM * WORKGROUP_SIZE as usize + 1;
        assert!(matches!(
            row_batch_grid(long_row, 1, "fft"),
            Err(Error::InvalidArgument { arg: "shape", .. })
        ));
        let rows = MAX_GROUPS_PER_DIM * MAX_GROUPS_PER_DIM + 1;
        assert!(matches!(
            row_batch_grid(1, rows, "fft"),
            Err(Error::InvalidArgument { arg: "shape", .. })
        ));
    }

    #[test]
    fn grid_covers_every_count_without_passing_u32() {
        let counts = [
            1u64,
            255,
            256,
            65535 * 256,
            65535 * 256 + 1,
            1 << 31,
            (1 << 32) - 256,
            (1 << 32) - 1,
        ];
        for count in counts {
            let grid = linear_grid(count as usize);
            assert!(grid.0 as usize <= MAX_GROUPS_PER_DIM, "{count}: {grid:?}");
            assert!(grid.1 as usize <= MAX_GROUPS_PER_DIM, "{count}: {grid:?}");
            let total = total_threads(grid);
            assert!(total >= count, "{count}: {grid:?} covers {total}");
            assert!(total <= 1 << 32, "{count}: {grid:?} holds {total}");
        }
    }
}
